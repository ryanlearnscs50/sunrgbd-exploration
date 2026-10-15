#!/usr/bin/env python3
"""
Scene Ranking and Diversity Selection for Optimal Memory Bank Discovery

This module implements comprehensive scene ranking based on preservation metrics
and semantic diversity selection algorithms for memory bank construction.

Key Features:
1. Full scene ranking using preservation metrics
2. Semantic diversity selection to avoid redundant scenes
3. Configurable selection strategies (greedy, beam search, etc.)
4. Integration with preservation metrics from Milestone 2

Date: August 2025
"""

import torch
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Set, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import heapq

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.utils.scene_discovery.preservation_metrics import PreservationMetrics


@dataclass
class SceneScore:
    """Container for scene scoring information."""
    scene_id: str
    preservation_score: float
    component_scores: Dict[str, float]
    metadata: Dict
    rank: int = -1


@dataclass 
class SelectionConfig:
    """Configuration for scene selection strategies."""
    strategy: str = 'greedy'  # 'greedy', 'beam', 'diversity_greedy'
    max_scenes: int = 30
    diversity_weight: float = 0.3  # Balance preservation vs diversity
    min_diversity_threshold: float = 0.1
    beam_width: int = 5  # For beam search
    semantic_feature_dim: int = 256  # Feature dimension for diversity


class SceneRankingSystem:
    """
    Comprehensive scene ranking and selection system.
    
    This system ranks all available training scenes using preservation metrics
    and implements various selection strategies to build diverse memory banks.
    """
    
    def __init__(
        self,
        stage1_checkpoint: str,
        base_config_path: str,
        device: str = 'cuda:0',
        output_dir: str = './scene_ranking_results',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize scene ranking system.
        
        Args:
            stage1_checkpoint: Path to Stage 1 model checkpoint
            base_config_path: Path to base configuration
            device: Device for computation
            output_dir: Directory to save ranking results
            logger: Optional logger
        """
        self.stage1_checkpoint = stage1_checkpoint
        self.base_config_path = base_config_path
        self.device = device
        self.output_dir = output_dir
        
        # Setup logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = logger
        
        # Initialize preservation metrics calculator
        self.metrics_calculator = PreservationMetrics(
            stage1_checkpoint=stage1_checkpoint,
            base_config_path=base_config_path,
            device=device,
            logger=self.logger
        )
        
        # Scene data storage
        self.scene_scores: Dict[str, SceneScore] = {}
        self.ranked_scenes: List[SceneScore] = []
        self.scene_features: Dict[str, np.ndarray] = {}  # For diversity calculation
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"SceneRankingSystem initialized, output: {output_dir}")
    
    def get_all_training_scenes(self) -> List[str]:
        """
        Get all available training scene IDs from the dataset.
        
        Returns:
            List of scene IDs found in training dataset
        """
        self.logger.info("Discovering all training scenes...")
        
        # Build training dataset to discover scene IDs
        cfg = Config.fromfile(self.base_config_path)
        train_cfg = cfg.data.train.copy()
        train_cfg.type = 'ScanNetDataset'
        if hasattr(train_cfg, 'variant'):
            train_cfg.variant = 'dynamic_head'
        train_cfg.test_mode = True
        
        # Remove incremental-specific parameters
        incremental_params = ['stage_definition', 'mappings', 'memory_bank', 
                             'evaluation_mode', 'all_stage_definitions', 'use_sequential_gci']
        for param in incremental_params:
            if hasattr(train_cfg, param):
                delattr(train_cfg, param)
        
        dataset = build_dataset(train_cfg)
        self.logger.info(f"Built training dataset: {len(dataset)} samples")
        
        # Extract unique scene IDs
        scene_ids = set()
        for i in range(len(dataset)):
            sample_info = dataset.get_data_info(i)
            
            # Extract scene ID from pts_filename
            pts_filename = sample_info.get('pts_filename', '')
            if pts_filename and 'scene' in pts_filename:
                basename = os.path.basename(pts_filename)
                if '_' in basename and basename.startswith('scene'):
                    try:
                        # Extract scene ID: e.g., scene0568_00.bin -> scene0568_00
                        scene_part = basename.split('_')[0] + '_' + basename.split('_')[1].split('.')[0]
                        scene_ids.add(scene_part)
                    except IndexError:
                        continue
        
        scene_list = sorted(list(scene_ids))
        self.logger.info(f"Discovered {len(scene_list)} unique training scenes")
        return scene_list
    
    def rank_all_scenes(
        self,
        metric_type: str = 'combined',
        batch_size: int = 50,
        save_intermediate: bool = True
    ) -> List[SceneScore]:
        """
        Rank all training scenes using preservation metrics.
        
        Args:
            metric_type: Preservation metric to use ('combined', 'max_confidence', 'entropy')
            batch_size: Number of scenes to process in each batch
            save_intermediate: Whether to save intermediate results
            
        Returns:
            List of SceneScore objects sorted by preservation score (descending)
        """
        self.logger.info(f"Starting comprehensive scene ranking with {metric_type} metric")
        
        # Get all training scenes
        all_scenes = self.get_all_training_scenes()
        self.logger.info(f"Ranking {len(all_scenes)} scenes in batches of {batch_size}")
        
        # Process scenes in batches
        all_results = {}
        for i in range(0, len(all_scenes), batch_size):
            batch_scenes = all_scenes[i:i + batch_size]
            batch_idx = i // batch_size + 1
            total_batches = (len(all_scenes) + batch_size - 1) // batch_size
            
            self.logger.info(f"Processing batch {batch_idx}/{total_batches}: {len(batch_scenes)} scenes")
            
            try:
                # Analyze batch with preservation metrics
                batch_results = self.metrics_calculator.analyze_scene_batch(
                    scene_ids=batch_scenes,
                    metric_type=metric_type,
                    output_dir=os.path.join(self.output_dir, 'batch_results'),
                    batch_name=f'ranking_batch_{batch_idx:03d}'
                )
                
                # Store results
                if batch_results['scene_results']:
                    for scene_id, scene_result in batch_results['scene_results'].items():
                        all_results[scene_id] = scene_result
                
                # Save intermediate results
                if save_intermediate:
                    intermediate_file = os.path.join(self.output_dir, f'intermediate_ranking_batch_{batch_idx:03d}.json')
                    with open(intermediate_file, 'w') as f:
                        json.dump(batch_results, f, indent=2, default=str)
                
                self.logger.info(f"Batch {batch_idx} complete: {len(batch_results.get('scene_results', {}))} scenes processed")
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        self.logger.info(f"Scene ranking complete: {len(all_results)} scenes processed")
        
        # Convert to SceneScore objects
        scene_scores = []
        for scene_id, result in all_results.items():
            # Extract component scores based on metric type
            component_scores = {}
            if metric_type == 'combined' and 'component_scores' in result:
                component_scores = result['component_scores']
            elif metric_type in ['max_confidence', 'entropy']:
                component_scores[metric_type] = result['preservation_score']
            
            scene_score = SceneScore(
                scene_id=scene_id,
                preservation_score=result['preservation_score'],
                component_scores=component_scores,
                metadata={
                    'samples_processed': result.get('samples_processed', 0),
                    'metric_type': result.get('metric_type', metric_type),
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                }
            )
            scene_scores.append(scene_score)
        
        # Sort by preservation score (descending - higher is better)
        scene_scores.sort(key=lambda x: x.preservation_score, reverse=True)
        
        # Assign ranks
        for rank, scene in enumerate(scene_scores, 1):
            scene.rank = rank
        
        self.scene_scores = {scene.scene_id: scene for scene in scene_scores}
        self.ranked_scenes = scene_scores
        
        # Save final ranking
        self.save_scene_ranking(scene_scores, metric_type)
        
        self.logger.info(f"Scene ranking completed: {len(scene_scores)} scenes ranked")
        if scene_scores:
            self.logger.info(f"Top scene: {scene_scores[0].scene_id} (score: {scene_scores[0].preservation_score:.4f})")
            self.logger.info(f"Score range: {scene_scores[-1].preservation_score:.4f} - {scene_scores[0].preservation_score:.4f}")
        
        return scene_scores
    
    def save_scene_ranking(self, scene_scores: List[SceneScore], metric_type: str):
        """Save scene ranking results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'scene_ranking_{metric_type}_{timestamp}.json'
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to serializable format
        ranking_data = {
            'metadata': {
                'metric_type': metric_type,
                'total_scenes': len(scene_scores),
                'timestamp': datetime.now().isoformat(),
                'stage1_checkpoint': self.stage1_checkpoint,
                'base_config': self.base_config_path
            },
            'ranking': [
                {
                    'rank': scene.rank,
                    'scene_id': scene.scene_id,
                    'preservation_score': scene.preservation_score,
                    'component_scores': scene.component_scores,
                    'metadata': scene.metadata
                }
                for scene in scene_scores
            ],
            'statistics': {
                'mean_score': np.mean([s.preservation_score for s in scene_scores]) if scene_scores else 0.0,
                'std_score': np.std([s.preservation_score for s in scene_scores]) if scene_scores else 0.0,
                'min_score': min(s.preservation_score for s in scene_scores) if scene_scores else 0.0,
                'max_score': max(s.preservation_score for s in scene_scores) if scene_scores else 0.0,
                'top_10_scenes': [s.scene_id for s in scene_scores[:10]] if scene_scores else []
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(ranking_data, f, indent=2, default=str)
        
        self.logger.info(f"Scene ranking saved to: {filepath}")
        return filepath
    
    def compute_scene_diversity(
        self,
        scene_scores: List[SceneScore],
        diversity_method: str = 'semantic_similarity'
    ) -> Dict[str, np.ndarray]:
        """
        Compute diversity features for scenes to enable diversity-based selection.
        
        Args:
            scene_scores: List of SceneScore objects
            diversity_method: Method for computing diversity ('semantic_similarity', 'spatial_distribution')
            
        Returns:
            Dictionary mapping scene_id to diversity feature vector
        """
        self.logger.info(f"Computing scene diversity using {diversity_method}")
        
        if diversity_method == 'semantic_similarity':
            return self._compute_semantic_diversity(scene_scores)
        elif diversity_method == 'spatial_distribution':
            return self._compute_spatial_diversity(scene_scores)
        else:
            raise ValueError(f"Unknown diversity method: {diversity_method}")
    
    def _compute_semantic_diversity(self, scene_scores: List[SceneScore]) -> Dict[str, np.ndarray]:
        """
        Compute semantic diversity based on component score patterns.
        
        For now, we use the component scores as diversity features.
        Future improvements could use actual scene semantic features.
        """
        diversity_features = {}
        
        for scene in scene_scores:
            # Use component scores as basic semantic features
            feature_vector = []
            
            if 'max_confidence_score' in scene.component_scores:
                feature_vector.append(scene.component_scores['max_confidence_score'])
            if 'entropy_score' in scene.component_scores:
                feature_vector.append(scene.component_scores['entropy_score'])
            
            # Add preservation score
            feature_vector.append(scene.preservation_score)
            
            # Add rank-based features (normalized)
            rank_norm = scene.rank / len(scene_scores) if scene_scores else 0.5
            feature_vector.append(rank_norm)
            
            # Pad to fixed dimension
            while len(feature_vector) < 8:
                feature_vector.append(0.0)
            
            diversity_features[scene.scene_id] = np.array(feature_vector[:8])
        
        self.scene_features = diversity_features
        self.logger.info(f"Computed semantic diversity for {len(diversity_features)} scenes")
        return diversity_features
    
    def _compute_spatial_diversity(self, scene_scores: List[SceneScore]) -> Dict[str, np.ndarray]:
        """
        Compute spatial diversity based on scene ID patterns.
        
        This is a simplified approach using scene ID numeric values.
        """
        diversity_features = {}
        
        for scene in scene_scores:
            # Extract numeric part from scene ID (e.g., scene0568_00 -> [568, 0])
            try:
                scene_num = int(scene.scene_id.split('_')[0].replace('scene', ''))
                scene_sub = int(scene.scene_id.split('_')[1])
                
                # Simple spatial features based on scene numbering
                spatial_vector = [
                    scene_num / 1000.0,  # Normalized scene number
                    scene_sub / 10.0,    # Normalized sub-scene number
                    (scene_num % 100) / 100.0,  # Local clustering
                    scene.preservation_score  # Include quality
                ]
                
                # Pad to 8 dimensions
                while len(spatial_vector) < 8:
                    spatial_vector.append(0.0)
                
                diversity_features[scene.scene_id] = np.array(spatial_vector)
                
            except (ValueError, IndexError):
                # Fallback for scenes with non-standard naming
                diversity_features[scene.scene_id] = np.random.random(8) * 0.1
        
        self.scene_features = diversity_features
        self.logger.info(f"Computed spatial diversity for {len(diversity_features)} scenes")
        return diversity_features
    
    def diversity_greedy_selection(
        self,
        scene_scores: List[SceneScore],
        config: SelectionConfig
    ) -> Tuple[List[SceneScore], Dict]:
        """
        Greedy selection balancing preservation score and diversity.
        
        Args:
            scene_scores: Ranked list of SceneScore objects
            config: Selection configuration
            
        Returns:
            Tuple of (selected_scenes, selection_metadata)
        """
        self.logger.info(f"Diversity greedy selection: max_scenes={config.max_scenes}, diversity_weight={config.diversity_weight}")
        
        # Compute diversity features if not already done
        if not self.scene_features:
            self.compute_scene_diversity(scene_scores)
        
        selected_scenes = []
        remaining_scenes = scene_scores.copy()
        selection_history = []
        
        for iteration in range(config.max_scenes):
            best_scene = None
            best_score = -float('inf')
            best_breakdown = {}
            
            for candidate in remaining_scenes:
                # Base preservation score (normalized)
                preservation_component = candidate.preservation_score
                
                # Diversity component (average distance to selected scenes)
                if selected_scenes and candidate.scene_id in self.scene_features:
                    candidate_features = self.scene_features[candidate.scene_id]
                    
                    distances = []
                    for selected in selected_scenes:
                        if selected.scene_id in self.scene_features:
                            selected_features = self.scene_features[selected.scene_id]
                            # Cosine distance
                            dist = 1.0 - np.dot(candidate_features, selected_features) / (
                                np.linalg.norm(candidate_features) * np.linalg.norm(selected_features) + 1e-8
                            )
                            distances.append(dist)
                    
                    diversity_component = np.mean(distances) if distances else 1.0
                else:
                    diversity_component = 1.0  # Maximum diversity if no selected scenes yet
                
                # Combined score
                combined_score = (
                    (1 - config.diversity_weight) * preservation_component +
                    config.diversity_weight * diversity_component
                )
                
                breakdown = {
                    'preservation_component': preservation_component,
                    'diversity_component': diversity_component,
                    'combined_score': combined_score,
                    'iteration': iteration + 1
                }
                
                if combined_score > best_score:
                    best_scene = candidate
                    best_score = combined_score
                    best_breakdown = breakdown
            
            if best_scene is None:
                self.logger.warning(f"No valid scene found at iteration {iteration + 1}")
                break
            
            # Add best scene to selection
            selected_scenes.append(best_scene)
            remaining_scenes.remove(best_scene)
            
            # Record selection history
            selection_entry = {
                'iteration': iteration + 1,
                'selected_scene': best_scene.scene_id,
                'selection_score': best_score,
                'breakdown': best_breakdown,
                'total_selected': len(selected_scenes)
            }
            selection_history.append(selection_entry)
            
            self.logger.info(f"Iteration {iteration + 1}: Selected {best_scene.scene_id} (score: {best_score:.4f})")
        
        # Selection metadata
        metadata = {
            'selection_strategy': 'diversity_greedy',
            'config': {
                'max_scenes': config.max_scenes,
                'diversity_weight': config.diversity_weight,
                'diversity_method': 'semantic_similarity'
            },
            'results': {
                'scenes_selected': len(selected_scenes),
                'selection_history': selection_history,
                'final_scenes': [s.scene_id for s in selected_scenes]
            },
            'statistics': {
                'avg_preservation_score': np.mean([s.preservation_score for s in selected_scenes]),
                'preservation_score_std': np.std([s.preservation_score for s in selected_scenes]),
                'rank_distribution': [s.rank for s in selected_scenes]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Diversity greedy selection complete: {len(selected_scenes)} scenes selected")
        return selected_scenes, metadata
    
    def simple_greedy_selection(
        self,
        scene_scores: List[SceneScore],
        max_scenes: int
    ) -> Tuple[List[SceneScore], Dict]:
        """
        Simple greedy selection: take top N scenes by preservation score.
        
        Args:
            scene_scores: Ranked list of SceneScore objects
            max_scenes: Maximum number of scenes to select
            
        Returns:
            Tuple of (selected_scenes, selection_metadata)
        """
        selected_scenes = scene_scores[:max_scenes]
        
        metadata = {
            'selection_strategy': 'simple_greedy',
            'config': {'max_scenes': max_scenes},
            'results': {
                'scenes_selected': len(selected_scenes),
                'final_scenes': [s.scene_id for s in selected_scenes]
            },
            'statistics': {
                'avg_preservation_score': np.mean([s.preservation_score for s in selected_scenes]),
                'preservation_score_std': np.std([s.preservation_score for s in selected_scenes]),
                'score_range': [selected_scenes[-1].preservation_score, selected_scenes[0].preservation_score]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Simple greedy selection: {len(selected_scenes)} scenes selected")
        return selected_scenes, metadata
    
    def select_scenes(
        self,
        scene_scores: Optional[List[SceneScore]] = None,
        config: Optional[SelectionConfig] = None
    ) -> Tuple[List[SceneScore], Dict]:
        """
        Select optimal scenes using specified strategy.
        
        Args:
            scene_scores: Pre-ranked scene scores (uses self.ranked_scenes if None)
            config: Selection configuration (uses default if None)
            
        Returns:
            Tuple of (selected_scenes, selection_metadata)
        """
        if scene_scores is None:
            scene_scores = self.ranked_scenes
            
        if config is None:
            config = SelectionConfig()
        
        if not scene_scores:
            raise ValueError("No scene scores available. Run rank_all_scenes() first.")
        
        self.logger.info(f"Scene selection using strategy: {config.strategy}")
        
        if config.strategy == 'simple_greedy':
            return self.simple_greedy_selection(scene_scores, config.max_scenes)
        elif config.strategy == 'diversity_greedy':
            return self.diversity_greedy_selection(scene_scores, config)
        else:
            raise ValueError(f"Unknown selection strategy: {config.strategy}")
    
    def save_selection_results(
        self,
        selected_scenes: List[SceneScore],
        metadata: Dict,
        filename_prefix: str = 'scene_selection'
    ) -> str:
        """Save scene selection results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{filename_prefix}_{metadata["selection_strategy"]}_{timestamp}.json'
        filepath = os.path.join(self.output_dir, filename)
        
        selection_data = {
            'metadata': metadata,
            'selected_scenes': [
                {
                    'rank': scene.rank,
                    'scene_id': scene.scene_id,
                    'preservation_score': scene.preservation_score,
                    'component_scores': scene.component_scores
                }
                for scene in selected_scenes
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(selection_data, f, indent=2, default=str)
        
        self.logger.info(f"Selection results saved to: {filepath}")
        return filepath


def load_scene_ranking(filepath: str) -> List[SceneScore]:
    """
    Load scene ranking from saved JSON file.
    
    Args:
        filepath: Path to saved ranking file
        
    Returns:
        List of SceneScore objects
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    scene_scores = []
    for item in data['ranking']:
        scene_score = SceneScore(
            scene_id=item['scene_id'],
            preservation_score=item['preservation_score'],
            component_scores=item.get('component_scores', {}),
            metadata=item.get('metadata', {}),
            rank=item['rank']
        )
        scene_scores.append(scene_score)
    
    return scene_scores
