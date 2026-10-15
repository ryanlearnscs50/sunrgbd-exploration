"""
Subset search algorithms for discovering optimal scene combinations.

This module implements greedy forward selection and beam search algorithms
to find the best combination of scenes for memory bank replay, optimizing
for catastrophic forgetting prevention in incremental learning.
"""

import os
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

from .trial_runner import FastTrialRunner
from .gradient_alignment import GradientAlignmentScorer


@dataclass
class SceneCandidate:
    """Represents a scene candidate with its utility metrics."""
    scene_id: str
    alignment_score: float = 0.0
    loss_improvement: float = 0.0
    gradient_norm: float = 0.0
    diversity_score: float = 0.0
    combined_score: float = 0.0
    
    def update_combined_score(self, 
                            alignment_weight: float = 0.4,
                            loss_weight: float = 0.3,
                            gradient_weight: float = 0.2,
                            diversity_weight: float = 0.1):
        """Update combined utility score with weighted factors."""
        self.combined_score = (
            alignment_weight * self.alignment_score +
            loss_weight * self.loss_improvement +
            gradient_weight * self.gradient_norm +
            diversity_weight * self.diversity_score
        )


class GreedySubsetSearch:
    """
    Greedy forward selection algorithm for scene subset discovery.
    
    Incrementally adds scenes that provide maximum marginal improvement
    to the validation performance or gradient alignment score.
    """
    
    def __init__(self, 
                 trial_runner: FastTrialRunner,
                 gradient_scorer: Optional[GradientAlignmentScorer] = None,
                 max_scenes: int = 50,
                 min_improvement_threshold: float = 0.001,
                 timeout_per_trial: int = 300,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize greedy subset search.
        
        Args:
            trial_runner: Fast trial execution framework
            gradient_scorer: Optional gradient alignment scorer
            max_scenes: Maximum number of scenes to select
            min_improvement_threshold: Minimum improvement to continue
            timeout_per_trial: Timeout for each trial in seconds
            logger: Logger instance
        """
        self.trial_runner = trial_runner
        self.gradient_scorer = gradient_scorer
        self.max_scenes = max_scenes
        self.min_improvement_threshold = min_improvement_threshold
        self.timeout_per_trial = timeout_per_trial
        self.logger = logger or logging.getLogger(__name__)
        
        # Search state
        self.selected_scenes = []
        self.candidate_scenes = []
        self.baseline_performance = None
        self.search_history = []
        
        # Performance tracking
        self.best_performance = 0.0
        self.iterations_without_improvement = 0
        self.max_stagnant_iterations = 5
        
    def initialize_candidates(self, 
                            scene_pool: List[str],
                            target_classes: List[int]) -> List[SceneCandidate]:
        """
        Initialize candidate scenes with utility scores.
        
        Args:
            scene_pool: Pool of available scene IDs
            target_classes: Target classes for alignment computation
            
        Returns:
            List of SceneCandidate objects with initial scores
        """
        self.logger.info(f"Initializing {len(scene_pool)} scene candidates")
        
        candidates = []
        
        # Compute gradient alignment scores if scorer available
        if self.gradient_scorer:
            self.logger.info("Computing gradient alignment scores...")
            alignment_scores = self.gradient_scorer.compute_alignment_scores_batch(
                scene_pool, target_classes
            )
        else:
            alignment_scores = {scene_id: 0.0 for scene_id in scene_pool}
        
        # Create candidate objects
        for scene_id in scene_pool:
            candidate = SceneCandidate(
                scene_id=scene_id,
                alignment_score=alignment_scores.get(scene_id, 0.0)
            )
            candidates.append(candidate)
        
        # Sort by alignment score initially
        candidates.sort(key=lambda x: x.alignment_score, reverse=True)
        
        self.logger.info(f"Created {len(candidates)} candidates")
        self.logger.info(f"Top candidate: {candidates[0].scene_id} "
                        f"(alignment={candidates[0].alignment_score:.4f})")
        
        return candidates
    
    def evaluate_baseline(self, stage_config: Dict[str, Any]) -> float:
        """
        Evaluate baseline performance without any replay scenes.
        
        Args:
            stage_config: Configuration for the training stage
            
        Returns:
            Baseline validation performance
        """
        self.logger.info("Evaluating baseline performance (no replay scenes)")
        
        # Create config with empty memory bank
        baseline_config = stage_config.copy()
        baseline_config['scene_memory_config'] = dict(
            memory_budget_ratio=0.0,  # No replay scenes
            selection_strategy='random'
        )
        
        try:
            # Run baseline trial
            results = self.trial_runner.run_single_trial(
                config=baseline_config,
                trial_name="baseline_evaluation",
                timeout=self.timeout_per_trial
            )
            
            # Extract performance metric (validation mAP)
            performance = self._extract_performance_metric(results)
            
            self.baseline_performance = performance
            self.logger.info(f"Baseline performance: {performance:.4f}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Baseline evaluation failed: {e}")
            return 0.0
    
    def evaluate_add_one_gain(self, 
                            candidate: SceneCandidate,
                            current_scenes: List[str],
                            stage_config: Dict[str, Any]) -> float:
        """
        Evaluate the marginal gain of adding one candidate scene.
        
        Args:
            candidate: Scene candidate to evaluate
            current_scenes: Currently selected scenes
            stage_config: Training stage configuration
            
        Returns:
            Performance gain from adding the candidate
        """
        # Create trial config with candidate added
        trial_scenes = current_scenes + [candidate.scene_id]
        trial_config = stage_config.copy()
        
        # Force specific scenes in memory bank
        trial_config['scene_memory_config'] = dict(
            forced_scene_list=trial_scenes,
            selection_strategy='forced',
            dedup_strategy='merge_labels'
        )
        
        trial_name = f"add_one_{candidate.scene_id}_{len(trial_scenes)}"
        
        try:
            # Run trial with candidate added
            results = self.trial_runner.run_single_trial(
                config=trial_config,
                trial_name=trial_name,
                timeout=self.timeout_per_trial
            )
            
            # Extract performance
            performance = self._extract_performance_metric(results)
            
            # Compute marginal gain
            if current_scenes:
                # Compare to current subset performance
                current_performance = self.search_history[-1]['performance'] if self.search_history else self.baseline_performance
                gain = performance - current_performance
            else:
                # Compare to baseline
                gain = performance - (self.baseline_performance or 0.0)
            
            self.logger.info(f"  Scene {candidate.scene_id}: performance={performance:.4f}, gain={gain:.4f}")
            
            return gain
            
        except Exception as e:
            self.logger.warning(f"Trial failed for scene {candidate.scene_id}: {e}")
            return 0.0
    
    def run_greedy_search(self,
                         scene_pool: List[str],
                         stage_config: Dict[str, Any],
                         target_classes: List[int]) -> Dict[str, Any]:
        """
        Run greedy forward selection to find optimal scene subset.
        
        Args:
            scene_pool: Pool of candidate scenes
            stage_config: Training configuration for trials
            target_classes: Target classes for gradient alignment
            
        Returns:
            Dictionary with search results and selected scenes
        """
        self.logger.info(f"Starting greedy subset search with {len(scene_pool)} scenes")
        self.logger.info(f"Target: max {self.max_scenes} scenes, "
                        f"min improvement {self.min_improvement_threshold}")
        
        start_time = time.time()
        
        # Initialize candidates
        candidates = self.initialize_candidates(scene_pool, target_classes)
        self.candidate_scenes = candidates
        
        # Evaluate baseline
        baseline_perf = self.evaluate_baseline(stage_config)
        current_performance = baseline_perf
        
        # Greedy selection loop
        for iteration in range(self.max_scenes):
            self.logger.info(f"\n--- Greedy Iteration {iteration + 1} ---")
            self.logger.info(f"Current subset size: {len(self.selected_scenes)}")
            self.logger.info(f"Current performance: {current_performance:.4f}")
            
            # Find remaining candidates
            selected_ids = {scene.scene_id for scene in self.selected_scenes} if hasattr(self, 'selected_scenes') else set()
            remaining_candidates = [c for c in candidates if c.scene_id not in selected_ids]
            
            if not remaining_candidates:
                self.logger.info("No more candidates available")
                break
            
            # Evaluate marginal gain for each remaining candidate
            best_candidate = None
            best_gain = 0.0
            
            self.logger.info(f"Evaluating {len(remaining_candidates)} remaining candidates...")
            
            for candidate in remaining_candidates:
                gain = self.evaluate_add_one_gain(
                    candidate, 
                    [s if isinstance(s, str) else s.scene_id for s in self.selected_scenes],
                    stage_config
                )
                
                # Update candidate with performance info
                candidate.loss_improvement = gain
                candidate.update_combined_score()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
            
            # Check improvement threshold
            if best_gain < self.min_improvement_threshold:
                self.logger.info(f"Best gain {best_gain:.6f} below threshold "
                                f"{self.min_improvement_threshold:.6f}. Stopping.")
                break
            
            # Add best candidate to selection
            if best_candidate:
                self.selected_scenes.append(best_candidate)
                current_performance += best_gain
                self.best_performance = max(self.best_performance, current_performance)
                
                # Record search step
                search_step = {
                    'iteration': iteration + 1,
                    'added_scene': best_candidate.scene_id,
                    'marginal_gain': best_gain,
                    'performance': current_performance,
                    'total_scenes': len(self.selected_scenes),
                    'timestamp': time.time()
                }
                self.search_history.append(search_step)
                
                self.logger.info(f"✅ Added scene {best_candidate.scene_id}")
                self.logger.info(f"   Marginal gain: {best_gain:.6f}")
                self.logger.info(f"   New performance: {current_performance:.4f}")
                
                # Reset stagnation counter
                self.iterations_without_improvement = 0
                
            else:
                self.iterations_without_improvement += 1
                if self.iterations_without_improvement >= self.max_stagnant_iterations:
                    self.logger.info(f"No improvement for {self.max_stagnant_iterations} iterations. Stopping.")
                    break
        
        # Compile results
        total_time = time.time() - start_time
        
        results = {
            'algorithm': 'greedy_forward_selection',
            'selected_scenes': [s.scene_id if hasattr(s, 'scene_id') else s for s in self.selected_scenes],
            'num_selected': len(self.selected_scenes),
            'baseline_performance': baseline_perf,
            'final_performance': current_performance,
            'total_improvement': current_performance - baseline_perf,
            'search_history': self.search_history,
            'total_time_seconds': total_time,
            'iterations_completed': len(self.search_history),
            'candidates_evaluated': len(scene_pool),
            'config': {
                'max_scenes': self.max_scenes,
                'min_improvement_threshold': self.min_improvement_threshold,
                'timeout_per_trial': self.timeout_per_trial
            }
        }
        
        self.logger.info(f"\n🎯 Greedy search completed!")
        self.logger.info(f"   Selected scenes: {len(self.selected_scenes)}")
        self.logger.info(f"   Final performance: {current_performance:.4f}")
        self.logger.info(f"   Total improvement: {current_performance - baseline_perf:.4f}")
        self.logger.info(f"   Search time: {total_time:.1f} seconds")
        
        return results
    
    def _extract_performance_metric(self, trial_results: Dict[str, Any]) -> float:
        """
        Extract the primary performance metric from trial results.
        
        Args:
            trial_results: Results from trial execution
            
        Returns:
            Performance metric (validation mAP)
        """
        if not trial_results:
            self.logger.debug("Empty trial results")
            return 0.0
        
        # Log the structure for debugging
        self.logger.debug(f"Trial results keys: {list(trial_results.keys())}")
        self.logger.info(f"Extracting performance from trial with keys: {list(trial_results.keys())}")
        
        # Check if trial was successful
        if not trial_results.get('success', False):
            error_msg = trial_results.get('error', 'Unknown error')
            self.logger.warning(f"Trial failed: {error_msg}")
            return 0.0
        
        # Try to extract validation mAP
        try:
            # CRITICAL: For discovery trials, look for the discovery utility score FIRST
            if 'discovery_utility' in trial_results:
                utility_score = float(trial_results['discovery_utility'])
                self.logger.debug(f"Found discovery utility score: {utility_score:.4f}")
                return utility_score
            
            # Fallback: Try to compute discovery utility from old/new class scores
            if 'old_class_mAP_0.25' in trial_results and 'new_class_mAP_0.25' in trial_results:
                old_map = float(trial_results['old_class_mAP_0.25'])
                new_map = float(trial_results['new_class_mAP_0.25'])
                discovery_utility = old_map + 0.3 * new_map
                self.logger.debug(f"Computed discovery utility: old={old_map:.4f}, new={new_map:.4f}, utility={discovery_utility:.4f}")
                return discovery_utility
            
            # Look for validation results
            if 'validation_results' in trial_results:
                val_results = trial_results['validation_results']
                self.logger.debug(f"Found validation_results: {val_results}")
                if isinstance(val_results, dict) and 'mAP' in val_results:
                    return float(val_results['mAP'])
                elif isinstance(val_results, dict) and 'mAP_0.25' in val_results:
                    return float(val_results['mAP_0.25'])
            
            # Look for final metrics
            if 'final_metrics' in trial_results:
                metrics = trial_results['final_metrics']
                self.logger.debug(f"Found final_metrics: {metrics}")
                if 'mAP' in metrics:
                    return float(metrics['mAP'])
                elif 'mAP_0.25' in metrics:
                    return float(metrics['mAP_0.25'])
            
            # Try direct access to other performance keys (legacy fallback)
            performance_keys = ['mAP', 'mAP_0.25', 'overall_mAP_0.25', 'old_class_mAP_0.25', 'detailed_results']
            for key in performance_keys:
                if key in trial_results:
                    value = trial_results[key]
                    self.logger.debug(f"Found {key}: {value}")
                    if key == 'detailed_results' and isinstance(value, dict):
                        # Extract from detailed results
                        if 'mAP_0.25' in value:
                            return float(value['mAP_0.25'])
                        elif 'mAP' in value:
                            return float(value['mAP'])
                    elif isinstance(value, (int, float)):
                        return float(value)
            
            # Look for training loss improvement (fallback)
            if 'training_loss' in trial_results:
                return 1.0 / (1.0 + trial_results['training_loss'])
            
            # Default fallback - add more detailed debugging
            self.logger.warning("No valid performance metric found, returning 0.0")
            self.logger.debug(f"Available keys in trial_results: {list(trial_results.keys()) if trial_results else 'None'}")
            if trial_results:
                self.logger.debug(f"Trial success status: {trial_results.get('success', 'Not found')}")
                if 'error' in trial_results:
                    self.logger.warning(f"Trial error details: {trial_results['error']}")
                if 'traceback' in trial_results:
                    self.logger.warning(f"Trial traceback: {trial_results['traceback']}")
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Error extracting performance metric: {e}")
            self.logger.debug(f"Full trial_results: {trial_results}")
            return 0.0
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save search results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Greedy search results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


class BeamSubsetSearch:
    """
    Beam search algorithm for scene subset discovery.
    
    Maintains multiple candidate subsets simultaneously and explores
    the most promising paths to find better solutions than greedy search.
    """
    
    def __init__(self,
                 trial_runner: FastTrialRunner,
                 gradient_scorer: Optional[GradientAlignmentScorer] = None,
                 beam_width: int = 3,
                 max_depth: int = 20,
                 min_improvement_threshold: float = 0.001,
                 timeout_per_trial: int = 300,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize beam search.
        
        Args:
            trial_runner: Fast trial execution framework
            gradient_scorer: Optional gradient alignment scorer
            beam_width: Number of candidates to maintain at each level
            max_depth: Maximum search depth (number of scenes)
            min_improvement_threshold: Minimum improvement threshold
            timeout_per_trial: Timeout for each trial
            logger: Logger instance
        """
        self.trial_runner = trial_runner
        self.gradient_scorer = gradient_scorer
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_improvement_threshold = min_improvement_threshold
        self.timeout_per_trial = timeout_per_trial
        self.logger = logger or logging.getLogger(__name__)
        
        # Beam search state
        self.beam = []  # List of (scenes, performance) tuples
        self.explored_subsets = set()  # Cache of evaluated subsets
        self.best_subset = None
        self.best_performance = 0.0
        
    def run_beam_search(self,
                       scene_pool: List[str],
                       stage_config: Dict[str, Any],
                       target_classes: List[int]) -> Dict[str, Any]:
        """
        Run beam search to find optimal scene subsets.
        
        Args:
            scene_pool: Pool of candidate scenes
            stage_config: Training configuration for trials
            target_classes: Target classes for alignment
            
        Returns:
            Dictionary with search results
        """
        self.logger.info(f"Starting beam search with {len(scene_pool)} scenes")
        self.logger.info(f"Beam width: {self.beam_width}, Max depth: {self.max_depth}")
        
        start_time = time.time()
        
        # Initialize with empty subset (baseline)
        baseline_config = stage_config.copy()
        baseline_config['scene_memory_config'] = dict(memory_budget_ratio=0.0)
        
        baseline_results = self.trial_runner.run_single_trial(
            config=baseline_config,
            trial_name="beam_baseline",
            timeout=self.timeout_per_trial
        )
        baseline_performance = self._extract_performance_metric(baseline_results)
        
        # Initialize beam with empty subset
        self.beam = [([], baseline_performance)]
        self.best_subset = []
        self.best_performance = baseline_performance
        
        # Beam search iterations
        for depth in range(self.max_depth):
            self.logger.info(f"\n--- Beam Search Depth {depth + 1} ---")
            self.logger.info(f"Current beam size: {len(self.beam)}")
            
            new_candidates = []
            
            # Expand each beam member
            for current_subset, current_performance in self.beam:
                # Find available scenes (not in current subset)
                available_scenes = [s for s in scene_pool if s not in current_subset]
                
                # Expand with each available scene
                for scene_id in available_scenes:
                    new_subset = current_subset + [scene_id]
                    subset_key = tuple(sorted(new_subset))
                    
                    # Skip if already explored
                    if subset_key in self.explored_subsets:
                        continue
                    
                    # Evaluate new subset
                    performance = self._evaluate_subset(new_subset, stage_config)
                    self.explored_subsets.add(subset_key)
                    
                    # Check if this is an improvement
                    gain = performance - current_performance
                    if gain >= self.min_improvement_threshold:
                        new_candidates.append((new_subset, performance))
                        
                        # Update best if better
                        if performance > self.best_performance:
                            self.best_performance = performance
                            self.best_subset = new_subset.copy()
                            self.logger.info(f"🔥 New best subset found: "
                                           f"{len(new_subset)} scenes, "
                                           f"performance={performance:.4f}")
            
            # Select top beam_width candidates for next iteration
            if new_candidates:
                # Sort by performance and take top candidates
                new_candidates.sort(key=lambda x: x[1], reverse=True)
                self.beam = new_candidates[:self.beam_width]
                
                self.logger.info(f"Generated {len(new_candidates)} new candidates")
                self.logger.info(f"Beam performance range: "
                                f"{self.beam[-1][1]:.4f} - {self.beam[0][1]:.4f}")
            else:
                self.logger.info("No improving candidates found. Search terminated.")
                break
        
        # Compile results
        total_time = time.time() - start_time
        
        results = {
            'algorithm': 'beam_search',
            'selected_scenes': self.best_subset,
            'num_selected': len(self.best_subset),
            'baseline_performance': baseline_performance,
            'final_performance': self.best_performance,
            'total_improvement': self.best_performance - baseline_performance,
            'subsets_evaluated': len(self.explored_subsets),
            'total_time_seconds': total_time,
            'config': {
                'beam_width': self.beam_width,
                'max_depth': self.max_depth,
                'min_improvement_threshold': self.min_improvement_threshold
            }
        }
        
        self.logger.info(f"\n🎯 Beam search completed!")
        self.logger.info(f"   Best subset: {len(self.best_subset)} scenes")
        self.logger.info(f"   Best performance: {self.best_performance:.4f}")
        self.logger.info(f"   Total improvement: {self.best_performance - baseline_performance:.4f}")
        self.logger.info(f"   Subsets evaluated: {len(self.explored_subsets)}")
        self.logger.info(f"   Search time: {total_time:.1f} seconds")
        
        return results
    
    def _evaluate_subset(self, scene_subset: List[str], stage_config: Dict[str, Any]) -> float:
        """Evaluate a specific scene subset."""
        # Create trial config with forced scene list
        trial_config = stage_config.copy()
        trial_config['scene_memory_config'] = dict(
            forced_scene_list=scene_subset,
            selection_strategy='forced'
        )
        
        trial_name = f"beam_eval_{len(scene_subset)}_{hash(tuple(sorted(scene_subset))) % 10000}"
        
        try:
            results = self.trial_runner.run_single_trial(
                config=trial_config,
                trial_name=trial_name,
                timeout=self.timeout_per_trial
            )
            
            return self._extract_performance_metric(results)
            
        except Exception as e:
            self.logger.warning(f"Subset evaluation failed: {e}")
            return 0.0
    
    def _extract_performance_metric(self, trial_results: Dict[str, Any]) -> float:
        """Extract performance metric from trial results."""
        if not trial_results:
            return 0.0
        
        # Check if trial was successful
        if not trial_results.get('success', False):
            return 0.0
        
        try:
            # CRITICAL: For discovery trials, look for the discovery utility score FIRST
            if 'discovery_utility' in trial_results:
                utility_score = float(trial_results['discovery_utility'])
                return utility_score
            
            # Fallback: Try to compute discovery utility from old/new class scores
            if 'old_class_mAP_0.25' in trial_results and 'new_class_mAP_0.25' in trial_results:
                old_map = float(trial_results['old_class_mAP_0.25'])
                new_map = float(trial_results['new_class_mAP_0.25'])
                discovery_utility = old_map + 0.3 * new_map
                return discovery_utility
            
            # Look for validation results
            if 'validation_results' in trial_results:
                val_results = trial_results['validation_results']
                if isinstance(val_results, dict) and 'mAP' in val_results:
                    return float(val_results['mAP'])
                elif isinstance(val_results, dict) and 'mAP_0.25' in val_results:
                    return float(val_results['mAP_0.25'])
            
            # Look for final metrics
            if 'final_metrics' in trial_results:
                metrics = trial_results['final_metrics']
                if 'mAP' in metrics:
                    return float(metrics['mAP'])
                elif 'mAP_0.25' in metrics:
                    return float(metrics['mAP_0.25'])
            
            return 0.0
            
        except Exception:
            return 0.0


class RandomBaselineSearch:
    """
    Random baseline for comparison with intelligent search algorithms.
    
    Randomly selects scene subsets to establish a baseline performance
    for evaluating the effectiveness of greedy and beam search.
    """
    
    def __init__(self,
                 trial_runner: FastTrialRunner,
                 num_trials: int = 10,
                 subset_sizes: List[int] = None,
                 timeout_per_trial: int = 300,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize random baseline search.
        
        Args:
            trial_runner: Trial execution framework
            num_trials: Number of random trials per subset size
            subset_sizes: List of subset sizes to evaluate
            timeout_per_trial: Timeout for each trial
            logger: Logger instance
        """
        self.trial_runner = trial_runner
        self.num_trials = num_trials
        self.subset_sizes = subset_sizes or [120]
        self.timeout_per_trial = timeout_per_trial
        self.logger = logger or logging.getLogger(__name__)
        
    def run_random_baseline(self,
                           scene_pool: List[str],
                           stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run random subset selection baseline.
        
        Args:
            scene_pool: Pool of candidate scenes
            stage_config: Training configuration
            
        Returns:
            Dictionary with baseline results
        """
        self.logger.info(f"Running random baseline with {len(scene_pool)} scenes")
        self.logger.info(f"Subset sizes: {self.subset_sizes}")
        self.logger.info(f"Trials per size: {self.num_trials}")
        
        results = {
            'algorithm': 'random_baseline',
            'scene_pool_size': len(scene_pool),
            'subset_sizes': self.subset_sizes,
            'num_trials_per_size': self.num_trials,
            'results_by_size': {}
        }
        
        for subset_size in self.subset_sizes:
            if subset_size > len(scene_pool):
                continue
                
            self.logger.info(f"\n--- Random baseline: {subset_size} scenes ---")
            
            size_results = []
            
            for trial in range(self.num_trials):
                # Random subset selection
                random_subset = np.random.choice(
                    scene_pool, size=subset_size, replace=False
                ).tolist()
                
                # Evaluate subset
                trial_config = stage_config.copy()
                trial_config['scene_memory_config'] = dict(
                    forced_scene_list=random_subset,
                    selection_strategy='forced'
                )
                
                trial_name = f"random_{subset_size}_{trial}"
                
                try:
                    trial_results = self.trial_runner.run_single_trial(
                        config=trial_config,
                        trial_name=trial_name,
                        timeout=self.timeout_per_trial
                    )
                    
                    performance = self._extract_performance_metric(trial_results)
                    
                    size_results.append({
                        'trial': trial + 1,
                        'scenes': random_subset,
                        'performance': performance
                    })
                    
                    self.logger.info(f"  Trial {trial + 1}: {performance:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Random trial {trial + 1} failed: {e}")
            
            # Compute statistics for this subset size
            performances = [r['performance'] for r in size_results if 'performance' in r]
            
            if performances:
                size_stats = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'min_performance': np.min(performances),
                    'max_performance': np.max(performances),
                    'num_successful_trials': len(performances),
                    'trials': size_results
                }
                
                results['results_by_size'][subset_size] = size_stats
                
                self.logger.info(f"Size {subset_size} results: "
                                f"mean={size_stats['mean_performance']:.4f} ± "
                                f"{size_stats['std_performance']:.4f}")
        
        return results
    
    def _extract_performance_metric(self, trial_results: Dict[str, Any]) -> float:
        """Extract performance metric from trial results."""
        if not trial_results:
            return 0.0
        
        # Check if trial was successful
        if not trial_results.get('success', False):
            return 0.0
        
        try:
            # CRITICAL: For discovery trials, look for the discovery utility score FIRST
            if 'discovery_utility' in trial_results:
                utility_score = float(trial_results['discovery_utility'])
                return utility_score
            
            # Fallback: Try to compute discovery utility from old/new class scores
            if 'old_class_mAP_0.25' in trial_results and 'new_class_mAP_0.25' in trial_results:
                old_map = float(trial_results['old_class_mAP_0.25'])
                new_map = float(trial_results['new_class_mAP_0.25'])
                discovery_utility = old_map + 0.3 * new_map
                return discovery_utility
            
            # Look for validation results
            if 'validation_results' in trial_results:
                val_results = trial_results['validation_results']
                if isinstance(val_results, dict) and 'mAP' in val_results:
                    return float(val_results['mAP'])
                elif isinstance(val_results, dict) and 'mAP_0.25' in val_results:
                    return float(val_results['mAP_0.25'])
            
            # Look for final metrics
            if 'final_metrics' in trial_results:
                metrics = trial_results['final_metrics']
                if 'mAP' in metrics:
                    return float(metrics['mAP'])
                elif 'mAP_0.25' in metrics:
                    return float(metrics['mAP_0.25'])
            
            return 0.0
            
        except Exception:
            return 0.0