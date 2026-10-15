"""
Fast trial execution framework for scene discovery experiments.

This module provides utilities to run many fast training trials to evaluate
different scene combinations for memory bank selection.
"""

import os
import time
import tempfile
import shutil
import logging
from typing import Dict, List, Optional, Any, Tuple
import json

import torch
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import train_model, single_gpu_test
from mmdet.datasets import build_dataloader
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel


class FastTrialRunner:
    """
    Runs fast training trials for scene selection experiments.
    
    This class is designed to run many short training trials (1-2 epochs)
    to evaluate which scenes are most valuable for preventing catastrophic
    forgetting in incremental learning.
    """
    
    def __init__(self, 
                 base_config_path: str,
                 stage1_checkpoint: str,
                 device: str = 'cuda:0',
                 cleanup_trials: bool = True,
                 max_gpu_memory_gb: float = 8.0):
        """
        Initialize the fast trial runner.
        
        Args:
            base_config_path: Path to base fast trial configuration
            stage1_checkpoint: Path to Stage 1 checkpoint for warm start
            device: Device for training ('cuda:0', 'cuda:1', etc.)
            cleanup_trials: Whether to cleanup trial directories after completion
            max_gpu_memory_gb: Maximum GPU memory usage (for monitoring)
        """
        self.base_config_path = base_config_path
        self.stage1_checkpoint = stage1_checkpoint
        self.device = device
        self.cleanup_trials = cleanup_trials
        self.max_gpu_memory_gb = max_gpu_memory_gb
        
        # Load base configuration
        self.base_cfg = Config.fromfile(base_config_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FastTrialRunner initialized:")
        self.logger.info(f"  - Base config: {base_config_path}")
        self.logger.info(f"  - Stage 1 checkpoint: {stage1_checkpoint}")
        self.logger.info(f"  - Device: {device}")
        
        # Verify Stage 1 checkpoint exists
        if not os.path.exists(stage1_checkpoint):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
        
        # Trial tracking
        self.trial_count = 0
        self.successful_trials = 0
        self.failed_trials = 0
        self.total_trial_time = 0.0
    
    def run_single_scene_trial(self,
                              scene_ids: List[str],
                              trial_epochs: int = 1,
                              work_dir: Optional[str] = None,
                              timeout_minutes: int = 10) -> Dict[str, Any]:
        """
        Run a single trial with specific scenes in memory bank.
        
        Args:
            scene_ids: List of scene IDs to include in memory bank
            trial_epochs: Number of epochs to train
            work_dir: Working directory (temporary if None)  
            timeout_minutes: Maximum time for trial
            
        Returns:
            Dictionary with trial results
        """
        self.trial_count += 1
        start_time = time.time()
        
        # Create work directory
        if work_dir is None:
            # Ensure trial_temp directory exists
            trial_temp_dir = './trial_temp'
            os.makedirs(trial_temp_dir, exist_ok=True)
            
            work_dir = tempfile.mkdtemp(
                prefix=f'scene_trial_{self.trial_count}_',
                dir=trial_temp_dir
            )
        
        os.makedirs(work_dir, exist_ok=True)
        
        self.logger.info(f"Trial {self.trial_count}: Testing {len(scene_ids)} scenes")
        self.logger.debug(f"  Work dir: {work_dir}")
        self.logger.debug(f"  Scenes: {scene_ids[:5]}{'...' if len(scene_ids) > 5 else ''}")
        
        try:
            # Create trial configuration
            cfg = self._create_trial_config(scene_ids, trial_epochs, work_dir)
            
            # Check memory before starting
            if torch.cuda.is_available():
                self._check_gpu_memory()
            
            # Run the trial with timeout
            result = self._run_trial_with_timeout(
                cfg, scene_ids, work_dir, timeout_minutes * 60
            )
            
            # Calculate trial time
            trial_time = time.time() - start_time
            self.total_trial_time += trial_time
            
            # Add metadata
            result.update({
                'scene_ids': scene_ids,
                'trial_epochs': trial_epochs,
                'trial_time_seconds': trial_time,
                'work_dir': work_dir,
                'trial_number': self.trial_count,
                'success': True
            })
            
            self.successful_trials += 1
            self.logger.info(f"Trial {self.trial_count} completed successfully in {trial_time:.1f}s")
            
            return result
            
        except TimeoutError:
            self.failed_trials += 1
            self.logger.warning(f"Trial {self.trial_count} timed out after {timeout_minutes} minutes")
            return {
                'scene_ids': scene_ids,
                'error': 'Trial timed out',
                'success': False,
                'trial_number': self.trial_count
            }
            
        except Exception as e:
            self.failed_trials += 1
            self.logger.error(f"Trial {self.trial_count} failed: {e}")
            return {
                'scene_ids': scene_ids,
                'error': str(e),
                'success': False,
                'trial_number': self.trial_count
            }
            
        finally:
            # Cleanup
            if self.cleanup_trials and work_dir and os.path.exists(work_dir):
                try:
                    shutil.rmtree(work_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup {work_dir}: {e}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def run_single_trial(self, 
                        config: Dict[str, Any], 
                        trial_name: str, 
                        timeout: int = 30) -> Dict[str, Any]:
        """
        Run a single trial with the given configuration.
        
        This is the main interface expected by subset search algorithms.
        Extracts scene information from config and delegates to run_single_scene_trial.
        
        Args:
            config: Configuration dictionary (should contain scene information)
            trial_name: Name identifier for the trial
            timeout: Timeout in minutes
            
        Returns:
            Trial results dictionary with performance metrics
        """
        try:
            # GPU context management and memory cleanup
            if 'cuda' in self.device.lower():
                torch.cuda.empty_cache()
                device_id = int(self.device.split(':')[1]) if ':' in self.device else 0
                torch.cuda.set_device(device_id)
                
            self.logger.info(f"DEBUG: Starting trial execution for {trial_name} on {self.device}")
            
            # Extract scene IDs from config
            scene_ids = []
            
            # Check if forced_scene_list is in scene_memory_config
            if ('scene_memory_config' in config and 
                config['scene_memory_config'] is not None and
                'forced_scene_list' in config['scene_memory_config']):
                scene_ids = config['scene_memory_config']['forced_scene_list']
            
            # Fallback: check if memory_budget_ratio is 0.0 (baseline case)
            elif ('scene_memory_config' in config and 
                  config['scene_memory_config'] is not None and
                  config['scene_memory_config'].get('memory_budget_ratio', 0.0) == 0.0):
                scene_ids = []  # No memory scenes for baseline
            
            else:
                self.logger.warning(f"Could not extract scene_ids from config in trial {trial_name}")
                return {
                    'trial_name': trial_name,
                    'scene_ids': [],
                    'error': 'Could not extract scene_ids from config',
                    'success': False
                }
            
            self.logger.info(f"Running trial '{trial_name}' with {len(scene_ids)} scenes: {scene_ids}")
            
            # Delegate to existing method
            result = self.run_single_scene_trial(
                scene_ids=scene_ids,
                trial_epochs=1,
                work_dir=None,  # Auto-generate work_dir
                timeout_minutes=timeout
            )
            
            # Add trial name to result
            result['trial_name'] = trial_name
            return result
            
        except Exception as e:
            self.logger.error(f"Error in run_single_trial '{trial_name}': {e}")
            import traceback
            full_traceback = traceback.format_exc()
            self.logger.error(f"Full traceback for trial '{trial_name}': {full_traceback}")
            
            # Check if this is the DataContainer error
            if "DataContainer" in str(e) and "iterable" in str(e):
                self.logger.error("CRITICAL: DataContainer iteration error detected in run_single_trial")
                self.logger.error(f"Error details: {str(e)}")
            
            return {
                'trial_name': trial_name,
                'scene_ids': [],
                'error': str(e),
                'traceback': full_traceback,
                'success': False
            }
    
    def run_add_one_evaluation(self,
                             candidate_scenes: List[str],
                             base_memory_scenes: List[str] = None,
                             max_scenes: int = 100,
                             progress_callback: Optional[callable] = None) -> Dict[str, float]:
        """
        Run Add-One evaluation for candidate scenes.
        
        This tests each candidate scene individually to determine its marginal
        utility for preventing catastrophic forgetting.
        
        Args:
            candidate_scenes: List of candidate scene IDs to test
            base_memory_scenes: Base scenes to always include
            max_scenes: Maximum number of scenes to evaluate
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary mapping scene_id -> utility_score
        """
        if base_memory_scenes is None:
            base_memory_scenes = []
        
        # Limit candidates for computational feasibility
        scenes_to_test = candidate_scenes[:max_scenes]
        scene_scores = {}
        
        self.logger.info(f"Starting Add-One evaluation:")
        self.logger.info(f"  - Candidate scenes: {len(scenes_to_test)}")
        self.logger.info(f"  - Base memory scenes: {len(base_memory_scenes)}")
        
        for i, scene_id in enumerate(scenes_to_test):
            if progress_callback:
                progress_callback(i, len(scenes_to_test), scene_id)
            
            self.logger.info(f"  Testing scene {i+1}/{len(scenes_to_test)}: {scene_id}")
            
            # Create memory bank with base scenes + this scene
            test_scenes = base_memory_scenes + [scene_id]
            
            # Run trial
            result = self.run_single_scene_trial(
                scene_ids=test_scenes,
                trial_epochs=1
            )
            
            if result.get('success', False):
                # Compute utility score
                utility_score = self._compute_utility_score(result)
                scene_scores[scene_id] = utility_score
                
                self.logger.info(f"    Scene {scene_id}: utility = {utility_score:.4f}")
            else:
                scene_scores[scene_id] = 0.0
                self.logger.warning(f"    Scene {scene_id}: trial failed")
        
        self.logger.info(f"Add-One evaluation completed: {len(scene_scores)} scenes evaluated")
        return scene_scores
    
    def run_scene_set_evaluation(self,
                                scene_set: List[str],
                                epochs: int = 3,
                                detailed: bool = False) -> Dict[str, Any]:
        """
        Run evaluation on a specific set of scenes with more epochs.
        
        This is used for final validation of selected scene combinations.
        
        Args:
            scene_set: List of scene IDs to evaluate together
            epochs: Number of epochs to train (more than trials)
            detailed: Whether to collect detailed metrics
            
        Returns:
            Detailed evaluation results
        """
        self.logger.info(f"Running detailed evaluation on {len(scene_set)} scenes for {epochs} epochs")
        
        result = self.run_single_scene_trial(
            scene_ids=scene_set,
            trial_epochs=epochs,
            timeout_minutes=30  # Longer timeout for detailed evaluation
        )
        
        if result.get('success', False):
            # Add detailed analysis if requested
            if detailed:
                result.update(self._analyze_detailed_results(result))
        
        return result
    
    def _create_trial_config(self,
                           scene_ids: List[str],
                           trial_epochs: int,
                           work_dir: str) -> Config:
        """Create configuration for a specific trial."""
        cfg = self.base_cfg.deepcopy()
        
        # Set basic parameters
        cfg.work_dir = work_dir
        cfg.runner.max_epochs = trial_epochs
        
        # Add required gpu_ids field for training
        if not hasattr(cfg, 'gpu_ids'):
            cfg.gpu_ids = [0]  # Default to GPU 0
        
        # Add other required training fields
        if not hasattr(cfg, 'seed'):
            cfg.seed = 42
        if not hasattr(cfg, 'deterministic'):
            cfg.deterministic = False
        
        # CRITICAL: Configure model for Stage 1 (7 classes) initially
        # We need to match the checkpoint size first, then expand after loading
        if hasattr(cfg.model, 'head'):
            cfg.model.head.n_classes = 7  # Start with 7 classes to match Stage 1 checkpoint
            self.logger.info(f"Configured model head for 7 classes (matching Stage 1 checkpoint)")
        
        # CRITICAL: Configure datasets for Stage 2 incremental learning
        # This ensures proper class filtering and label mapping
        stage_2_classes = list(range(14))  # Classes 0-13 for Stage 2
        
        # Load Stage 2 definition from dynamic head mapping - use relative path for portability
        import sys
        import os
        trial_runner_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.dirname(trial_runner_dir)  # mmdet3d/utils
        mmdet3d_dir = os.path.dirname(utils_dir)  # mmdet3d
        root_dir = os.path.dirname(mmdet3d_dir)  # TR3D root
        sys.path.append(os.path.join(root_dir, 'configs/_base_/class_mappings'))
        from scannet_dynamic_head_mapping import DYNAMIC_HEAD_STAGE_DEFINITIONS
        
        # Get Stage 2 definition (stage_id=2 means index 1)
        stage_2_def = DYNAMIC_HEAD_STAGE_DEFINITIONS[1]  # Stage 2
        
        # Configure training dataset for Stage 2
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'train'):
            cfg.data.train.use_sequential_gci = True  # Enable sequential GCI mapping
            cfg.data.train.stage_definition = stage_2_def  # Provide Stage 2 definition
            cfg.data.train.all_stage_definitions = DYNAMIC_HEAD_STAGE_DEFINITIONS  # Provide all stages
            cfg.data.train.evaluation_mode = False  # Training mode (not evaluation)
            
            # The dataset will compute all_seen_classes internally based on stage_definitions
            self.logger.info(f"Configured training dataset for Stage 2: classes {stage_2_def['class_indices']}")
        
        # Configure validation dataset for Stage 2
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'val'):
            # For validation, we only set the classes for evaluation (not the full incremental config)
            # The validation dataset uses the standard ScanNet35ClassBinFileDataset
            cfg.data.val.seen_classes_for_eval = stage_2_classes  # Evaluate on classes 0-13
            self.logger.info(f"Configured validation dataset for Stage 2: eval classes {stage_2_classes}")
        
        # Configure memory bank with specific scenes
        # This assumes the config has scene memory bank settings
        if hasattr(cfg, 'scene_memory_config') and cfg.scene_memory_config:
            cfg.scene_memory_config.max_memory_scenes = len(scene_ids)
            cfg.scene_memory_config.force_scene_list = scene_ids
            cfg.scene_memory_config.debug_mode = False  # Reduce logging in trials
        
        # Enable scene metrics collection in model
        if hasattr(cfg.model, 'head') and hasattr(cfg.model.head, 'train_cfg'):
            if cfg.model.head.train_cfg is None:
                cfg.model.head.train_cfg = {}
            cfg.model.head.train_cfg['collect_scene_metrics'] = True
        
        # Configure custom hooks - remove non-existent SceneMetricsHook for now
        # We'll focus on getting basic training to work first
        if not hasattr(cfg, 'custom_hooks'):
            cfg.custom_hooks = []
        
        # Remove any SceneMetricsHook references to avoid registry errors
        if hasattr(cfg, 'custom_hooks') and cfg.custom_hooks:
            cfg.custom_hooks = [hook for hook in cfg.custom_hooks 
                               if hook.get('type') != 'SceneMetricsHook']
        
        return cfg
    
    def _run_trial_with_timeout(self,
                               cfg: Config,
                               scene_ids: List[str],
                               work_dir: str,
                               timeout_seconds: int) -> Dict[str, Any]:
        """Run a trial with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Trial execution timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            self.logger.info(f"Building model with {cfg.model.head.n_classes} classes")
            
            # Build model
            model = build_model(
                cfg.model,
                train_cfg=cfg.get('train_cfg'),
                test_cfg=cfg.get('test_cfg')
            )
            
            self.logger.info(f"Loading Stage 1 checkpoint: {self.stage1_checkpoint}")
            # Load Stage 1 checkpoint - should now have matching 7-class head
            checkpoint_info = load_checkpoint(
                model, 
                self.stage1_checkpoint, 
                map_location=self.device,
                strict=False  # Still use False for safety
            )
            
            # CRITICAL: Expand model from Stage 1 (7 classes) to Stage 2 (14 classes)
            # This is required for Stage 2 discovery trials
            self.logger.info("Expanding model head from Stage 1 (7 classes) to Stage 2 (14 classes)")
            self._expand_model_head_to_stage2(model)
            self.logger.info("Model head expansion completed successfully")
            
            model = model.to(self.device)
            self.logger.info("Model and checkpoint loaded successfully")
            
            # Build datasets
            self.logger.info("Building train and validation datasets")
            train_dataset = build_dataset(cfg.data.train)
            val_dataset = build_dataset(cfg.data.val)
            self.logger.info(f"Datasets built - train: {len(train_dataset)}, val: {len(val_dataset)}")
            
            # Ensure work_dir is set in config
            if not hasattr(cfg, 'work_dir') or not cfg.work_dir:
                cfg.work_dir = work_dir
                self.logger.info(f"Set cfg.work_dir = {work_dir}")
            
            # Run training with enhanced debugging
            self.logger.info(f"Starting training with {len(scene_ids)} memory scenes")
            
            # Pre-training validation
            self.logger.info(f"PRE-TRAINING VALIDATION:")
            self.logger.info(f"  Model device: {next(model.parameters()).device}")
            self.logger.info(f"  Model head classes: {getattr(model.head, 'n_classes', 'N/A')}")
            self.logger.info(f"  Train dataset size: {len(train_dataset)}")
            self.logger.info(f"  Val dataset size: {len(val_dataset)}")
            self.logger.info(f"  Work dir: {work_dir}")
            self.logger.info(f"  Config total_epochs: {getattr(cfg, 'total_epochs', 'N/A')}")
            self.logger.info(f"  Config runner max_epochs: {getattr(cfg.runner, 'max_epochs', 'N/A') if hasattr(cfg, 'runner') else 'No runner config'}")
            
            # Create comprehensive log file for this trial
            trial_log_file = os.path.join(work_dir, f'training_debug_{self.trial_count}.log')
            self.logger.info(f"Trial training log will be saved to: {trial_log_file}")
            
            # Capture training output
            import sys
            import contextlib
            from io import StringIO
            
            training_output = StringIO()
            training_success = False
            training_error = None
            
            try:
                # Redirect stdout/stderr during training
                with contextlib.redirect_stdout(training_output), contextlib.redirect_stderr(training_output):
                    self.logger.info("CALLING train_model() - START")
                    train_model(
                        model,
                        train_dataset,
                        cfg,
                        distributed=False,
                        validate=True,
                        timestamp=str(int(time.time())),
                        meta=dict(scene_ids=scene_ids)
                    )
                    training_success = True
                    self.logger.info("CALLING train_model() - SUCCESS")
                    
            except Exception as training_exception:
                training_error = str(training_exception)
                self.logger.error(f"CALLING train_model() - FAILED: {training_error}")
                import traceback
                training_traceback = traceback.format_exc()
                self.logger.error(f"Training traceback: {training_traceback}")
            
            # Save captured output to file
            captured_output = training_output.getvalue()
            try:
                with open(trial_log_file, 'w') as f:
                    f.write(f"=== TRIAL {self.trial_count} TRAINING LOG ===\n")
                    f.write(f"Scene IDs: {scene_ids}\n")
                    f.write(f"Work Dir: {work_dir}\n")
                    f.write(f"Success: {training_success}\n")
                    f.write(f"Error: {training_error}\n")
                    f.write(f"=== CAPTURED OUTPUT ===\n")
                    f.write(captured_output)
                    f.write(f"\n=== END LOG ===\n")
                self.logger.info(f"Training output saved to: {trial_log_file}")
            except Exception as log_save_error:
                self.logger.error(f"Failed to save training log: {log_save_error}")
            
            # Log summary of captured output
            if captured_output:
                lines = captured_output.split('\n')
                self.logger.info(f"Captured {len(lines)} lines of training output")
                # Log first few lines
                for i, line in enumerate(lines[:5]):
                    if line.strip():
                        self.logger.info(f"  Line {i+1}: {line.strip()[:100]}...")
            else:
                self.logger.warning("No training output was captured!")
            
            # Post-training validation
            self.logger.info(f"POST-TRAINING VALIDATION:")
            self.logger.info(f"  Training success: {training_success}")
            self.logger.info(f"  Training error: {training_error}")
            
            if not training_success:
                # Return error immediately if training failed
                return {'success': False, 'error': f'Training failed: {training_error}', 'trial_log': trial_log_file}
            
            self.logger.info("Training completed - checking for outputs")
            
            # Evaluate the result
            self.logger.info("Looking for checkpoint to evaluate")
            latest_checkpoint = os.path.join(work_dir, 'latest.pth')
            if not os.path.exists(latest_checkpoint):
                # Try epoch checkpoints
                latest_checkpoint = os.path.join(work_dir, 'epoch_1.pth')
                self.logger.info(f"latest.pth not found, trying epoch_1.pth")
            
            if os.path.exists(latest_checkpoint):
                self.logger.info(f"Evaluating checkpoint: {latest_checkpoint}")
                results = self._evaluate_checkpoint(latest_checkpoint, val_dataset, cfg)
                self.logger.info(f"Evaluation completed, success: {results.get('success', False)}")
            else:
                self.logger.warning(f"No checkpoint found in {work_dir}")
                # List available files for debugging
                if os.path.exists(work_dir):
                    files = os.listdir(work_dir)
                    self.logger.warning(f"Available files in work_dir: {files}")
                results = {'success': False, 'error': 'No checkpoint found'}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Trial execution failed: {e}")
            import traceback
            full_traceback = traceback.format_exc()
            self.logger.error(f"Full traceback: {full_traceback}")
            
            # Check if this is the DataContainer error
            if "DataContainer" in str(e) and "iterable" in str(e):
                self.logger.error("CRITICAL: DataContainer iteration error detected in run_single_scene_trial")
                self.logger.error(f"This error is preventing discovery evaluation from working")
                
            return {'success': False, 'error': str(e), 'traceback': full_traceback}
            
        finally:
            # Cancel timeout
            signal.alarm(0)
    
    def _expand_model_head_to_stage2(self, model):
        """
        Expand model head from Stage 1 (7 classes) to Stage 2 (14 classes).
        
        Uses the proper TR3D head expansion method which creates a new MinkowskiConvolution
        layer and correctly preserves the trained weights.
        
        Args:
            model: The model with Stage 1 checkpoint loaded
        """
        head = model.head
        current_classes = head.n_classes
        target_classes = 14  # Target Stage 2 classes
        
        self.logger.info(f"Current head classes: {current_classes}")
        self.logger.info(f"Target head classes: {target_classes}")
        
        if current_classes == target_classes:
            self.logger.info("Model already has 14 classes - no expansion needed")
            return
        elif current_classes == 7 and target_classes == 14:
            # Use the proper TR3D head expansion method
            self.logger.info("Using TR3D head expansion method for proper weight preservation")
            success = head.expand_classification_head(target_classes)
            
            if success:
                self.logger.info(f"✅ Head expansion successful: {current_classes} -> {head.n_classes} classes")
                self.logger.info(f"✅ Stage 1 knowledge preserved in classes 0-6")
                self.logger.info(f"✅ Stage 2 classes 7-13 randomly initialized")
            else:
                self.logger.error(f"❌ Head expansion failed!")
                raise RuntimeError(f"Failed to expand head from {current_classes} to {target_classes} classes")
        else:
            raise ValueError(f"Unsupported head expansion: {current_classes} -> {target_classes} classes")
        
        self.logger.info("Model head expansion completed:")
        self.logger.info(f"  - Preserved Stage 1 classes: 0-6")
        self.logger.info(f"  - Added Stage 2 classes: 7-13")
        self.logger.info(f"  - Total classes: {head.n_classes}")
    
    def _evaluate_checkpoint(self,
                           checkpoint_path: str,
                           val_dataset: Any,
                           cfg: Config) -> Dict[str, Any]:
        """Evaluate a checkpoint and return stage-aware metrics."""
        if not os.path.exists(checkpoint_path):
            return {'success': False, 'error': 'Checkpoint not found'}
        
        try:
            # Create a copy of config for evaluation
            eval_cfg = cfg.deepcopy()
            
            # CRITICAL: Check what's actually saved in the checkpoint first
            temp_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            saved_head_shape = temp_checkpoint['state_dict'].get('head.cls_conv.kernel', None)
            if saved_head_shape is not None:
                saved_classes = saved_head_shape.shape[1]
                self.logger.info(f"Detected checkpoint has {saved_classes} classes")
            else:
                saved_classes = 7  # Default assumption
                self.logger.warning(f"Could not detect checkpoint head shape, assuming 7 classes")
            
            # Configure model to match saved checkpoint
            if hasattr(eval_cfg.model, 'head'):
                eval_cfg.model.head.n_classes = saved_classes
                self.logger.info(f"Configured evaluation model for {saved_classes} classes (matching checkpoint)")
            
            # Load model with matching architecture 
            model = build_model(
                eval_cfg.model,
                train_cfg=eval_cfg.get('train_cfg'),
                test_cfg=eval_cfg.get('test_cfg')
            )
            
            # Determine target device for loading and set CUDA context BEFORE loading
            if 'cuda' in self.device.lower() and torch.cuda.is_available():
                # Respect CUDA_VISIBLE_DEVICES: use device 0 in the visible device list
                cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                visible_device = int(cuda_devices.split(',')[0]) if cuda_devices else 0
                device_id = 0  # Always use the first visible device (mapped to index 0)
                torch.cuda.set_device(device_id)
                self.logger.info(f"Set CUDA device context to {device_id} for MinkowskiEngine (CUDA_VISIBLE_DEVICES={cuda_devices})")
                map_location = f'cuda:{device_id}'
            else:
                map_location = 'cpu'
            
            # Load checkpoint directly to target device for MinkowskiEngine compatibility
            load_checkpoint(model, checkpoint_path, map_location=map_location, strict=False)
            self.logger.info(f"Loaded {saved_classes}-class checkpoint to {map_location}")
            
            # If saved model has 7 classes but we need 14 for evaluation, expand it
            if saved_classes == 7:
                self.logger.info("Expanding model head from 7 to 14 classes for Stage 2 evaluation")
                self._expand_model_head_to_stage2(model)
                self.logger.info("Model head expansion completed for evaluation")
            
            # Wrap model with MMDataParallel for evaluation (following eval_incremental.py pattern)
            target_device = map_location if 'cuda' in map_location else self.device
            if 'cuda' in target_device:
                # Use MMDataParallel for GPU evaluation (required for MinkowskiEngine)
                model = MMDataParallel(model, device_ids=[0])
                self.logger.info(f"Model wrapped with MMDataParallel for GPU evaluation")
            else:
                # CPU evaluation - just move model to device
                model = model.to(target_device)
                self.logger.info(f"Model moved to CPU for evaluation")
            
            model.eval()
            self.logger.info(f"Model ready for evaluation on device: {target_device}")
            
            # Build dataloader
            dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=1,
                workers_per_gpu=1,
                dist=False,
                shuffle=False
            )
            
            # Configure dataset for Stage 2 evaluation 
            # CRITICAL: After Stage 2 training, we evaluate on BOTH Stage 1 and Stage 2 classes
            # to measure both old class retention AND new class learning
            stage1_and_stage2_classes = list(range(14))  # Classes 0-13 (both stages)
            if hasattr(val_dataset, 'seen_classes_for_eval'):
                val_dataset.seen_classes_for_eval = stage1_and_stage2_classes
                self.logger.info(f"Set dataset seen_classes_for_eval to {stage1_and_stage2_classes} (Stage 1+2 evaluation)")
            else:
                self.logger.warning("Dataset doesn't support seen_classes_for_eval - evaluation may be inaccurate")
            
            # Run evaluation with DataContainer handling
            from mmcv.parallel import DataContainer
            
            # Add DataContainer unwrapping helper
            def unwrap_data_containers(data_batch):
                """Unwrap DataContainer objects to prevent iteration errors."""
                for key in data_batch:
                    if isinstance(data_batch[key], DataContainer):
                        data_batch[key] = data_batch[key].data
                    elif isinstance(data_batch[key], list):
                        data_batch[key] = [
                            item.data if isinstance(item, DataContainer) else item 
                            for item in data_batch[key]
                        ]
                return data_batch
            
            # Patch dataloader to unwrap DataContainers
            original_iter = dataloader.__iter__
            def patched_iter():
                for batch in original_iter():
                    yield unwrap_data_containers(batch)
            dataloader.__iter__ = patched_iter
            
            outputs = single_gpu_test(model, dataloader, show=False)
            
            # Get evaluation results (evaluates on both Stage 1 and Stage 2 classes 0-13)
            eval_results = val_dataset.evaluate(outputs, metric='mAP')
            
            # Extract results for discovery evaluation after Stage 2 training
            stage1_classes = [0, 1, 2, 3, 4, 5, 6]      # Stage 1 classes (retention measurement)
            stage2_classes = [7, 8, 9, 10, 11, 12, 13]  # Stage 2 classes (learning measurement)
            
            old_class_results = self._extract_class_group_results(eval_results, stage1_classes)
            new_class_results = self._extract_class_group_results(eval_results, stage2_classes)
            
            # Compute discovery utility score: old_class_mAP + 0.3 * new_class_mAP  
            old_map = old_class_results.get('mAP_0.25', 0.0)
            new_map = new_class_results.get('mAP_0.25', 0.0)
            discovery_utility = old_map + 0.3 * new_map
            
            self.logger.info(f"Stage 2 Discovery Evaluation - Stage 1 classes (0-6): {old_map:.4f}, Stage 2 classes (7-13): {new_map:.4f}")
            self.logger.info(f"Discovery utility score: {discovery_utility:.4f}")
            
            # GPU memory cleanup before returning
            if 'cuda' in self.device.lower():
                torch.cuda.empty_cache()
            
            return {
                'success': True,
                'old_class_mAP_0.25': old_map,
                'old_class_mAP_0.50': old_class_results.get('mAP_0.50', 0.0), 
                'new_class_mAP_0.25': new_map,
                'new_class_mAP_0.50': new_class_results.get('mAP_0.50', 0.0),
                'discovery_utility': discovery_utility,  # Key metric for discovery
                'overall_mAP_0.25': eval_results.get('mAP_0.25', 0.0),
                'overall_mAP_0.50': eval_results.get('mAP_0.50', 0.0),
                'detailed_results': eval_results
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    def _extract_class_group_results(self, 
                                   eval_results: Dict,
                                   class_indices: List[int]) -> Dict[str, float]:
        """Extract mAP results for a specific group of classes."""
        # TR3D evaluation returns results in format like 'cabinet_AP_0.25', 'chair_AP_0.25', etc.
        # We need to extract these per-class results and compute group averages
        
        # Map class indices to class names (Sequential GCI mapping)
        # Legacy hardcoded ordering removed Aug 2025 - now imports from central mapping
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../../configs/_base_/class_mappings'))
        try:
            from scannet_dynamic_head_mappings import get_stage_definitions
            stage_defs = get_stage_definitions('frequency')
            # Build full class name list from stage definitions
            class_names = []
            for stage in stage_defs:
                class_names.extend(stage['class_names'])
        except ImportError:
            # Fallback if import fails - use corrected frequency ordering
            print("WARNING: Could not import frequency ordering, using hardcoded fallback")
            class_names = ['chair', 'door', 'otherfurniture', 'books', 'cabinet', 'table', 'window',
                          'pillow', 'picture', 'box', 'desk', 'shelves', 'towel', 'sofa',
                          'sink', 'clothes', 'lamp', 'bed', 'bookshelf', 'curtain', 'mirror',
                          'bag', 'whiteboard', 'counter', 'toilet', 'nightstand', 'refrigerator', 'television',
                          'dresser', 'shower_curtain', 'bathtub', 'paper', 'person', 'floor_mat', 'blinds']
        
        # Extract per-class AP@0.25 results
        class_maps_25 = []
        class_maps_50 = []
        
        for class_idx in class_indices:
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                ap_25_key = f'{class_name}_AP_0.25'
                ap_50_key = f'{class_name}_AP_0.50'
                
                ap_25 = eval_results.get(ap_25_key, 0.0)
                ap_50 = eval_results.get(ap_50_key, 0.0)
                
                class_maps_25.append(ap_25)
                class_maps_50.append(ap_50)
                
                self.logger.debug(f"Class {class_idx} ({class_name}): AP@0.25={ap_25:.4f}, AP@0.50={ap_50:.4f}")
        
        # Compute group averages
        group_map_25 = np.mean(class_maps_25) if class_maps_25 else 0.0
        group_map_50 = np.mean(class_maps_50) if class_maps_50 else 0.0
        
        self.logger.debug(f"Group average - AP@0.25: {group_map_25:.4f}, AP@0.50: {group_map_50:.4f}")
        
        return {
            'mAP_0.25': group_map_25,
            'mAP_0.50': group_map_50
        }
    
    def _compute_utility_score(self, result: Dict[str, Any]) -> float:
        """Compute utility score from trial results."""
        if not result.get('success', False):
            return 0.0
        
        # Weight old class retention higher than new class learning
        old_class_map = result.get('old_class_mAP_0.25', 0.0)
        new_class_map = result.get('new_class_mAP_0.25', 0.0)
        
        # Utility function: emphasize old class retention
        utility = old_class_map + 0.3 * new_class_map
        return utility
    
    def _analyze_detailed_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed results for comprehensive evaluation."""
        analysis = {}
        
        if result.get('success', False):
            # Compute forgetting metrics
            old_map = result.get('old_class_mAP_0.25', 0.0)
            new_map = result.get('new_class_mAP_0.25', 0.0)
            
            # Estimate forgetting (this would need baseline comparison)
            analysis.update({
                'forgetting_estimate': max(0.0, 0.4 - old_map),  # Assume 0.4 was Stage 1 performance
                'new_learning_rate': new_map,
                'balance_score': old_map / (old_map + new_map + 1e-6),
                'overall_performance': (old_map + new_map) / 2.0
            })
        
        return analysis
    
    def _check_gpu_memory(self):
        """Check GPU memory usage and warn if approaching limits."""
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            cached_gb = torch.cuda.memory_reserved() / (1024**3)
            
            if allocated_gb > self.max_gpu_memory_gb * 0.8:
                self.logger.warning(f"High GPU memory usage: {allocated_gb:.1f}GB allocated, "
                                   f"{cached_gb:.1f}GB cached")
            
            if allocated_gb > self.max_gpu_memory_gb:
                self.logger.error(f"GPU memory exceeded limit: {allocated_gb:.1f}GB > {self.max_gpu_memory_gb}GB")
                torch.cuda.empty_cache()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runner statistics."""
        avg_trial_time = self.total_trial_time / max(1, self.trial_count)
        success_rate = self.successful_trials / max(1, self.trial_count)
        
        return {
            'total_trials': self.trial_count,
            'successful_trials': self.successful_trials,
            'failed_trials': self.failed_trials,
            'success_rate': success_rate,
            'total_trial_time': self.total_trial_time,
            'average_trial_time': avg_trial_time,
            'trials_per_hour': 3600 / max(avg_trial_time, 1)
        }
    
    def save_statistics(self, filepath: str):
        """Save runner statistics to file."""
        stats = self.get_statistics()
        stats['stage1_checkpoint'] = self.stage1_checkpoint
        stats['base_config'] = self.base_config_path
        stats['device'] = self.device
        
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            self.logger.info(f"Statistics saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")