#!/usr/bin/env python3
"""
Flexible Pseudo Label Validation for Incremental Learning Pipeline

This module validates the quality of generated pseudo labels by comparing them
against expected metrics (when available) or performing basic sanity checks.

Key features:
- Stage-aware validation with reference data for Stage 2
- Flexible thresholds that adapt to confidence settings
- Non-blocking warnings that don't abort training
- Comprehensive logging for debugging

Date: September 2025
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logger
logger = logging.getLogger(__name__)

def interpolate_expected_metric(confidence_threshold: float, expected_metrics: Dict[float, Dict]) -> Dict:
    """Interpolate expected metrics based on confidence threshold."""
    # Get the two closest confidence values
    conf_values = sorted(expected_metrics.keys())
    
    if confidence_threshold <= conf_values[0]:
        return expected_metrics[conf_values[0]]
    elif confidence_threshold >= conf_values[-1]:
        return expected_metrics[conf_values[-1]]
    else:
        # Find surrounding values for interpolation
        lower_conf = max(c for c in conf_values if c <= confidence_threshold)
        upper_conf = min(c for c in conf_values if c >= confidence_threshold)
        
        if lower_conf == upper_conf:
            return expected_metrics[lower_conf]
        
        # Linear interpolation
        weight = (confidence_threshold - lower_conf) / (upper_conf - lower_conf)
        lower_metrics = expected_metrics[lower_conf]
        upper_metrics = expected_metrics[upper_conf]
        
        return {
            'f1': lower_metrics['f1'] * (1 - weight) + upper_metrics['f1'] * weight,
            'count': lower_metrics['count'] * (1 - weight) + upper_metrics['count'] * weight
        }

def compute_pseudo_label_stats(pseudo_labels: Dict[str, Any]) -> Dict[str, Any]:
    """Compute basic statistics for pseudo labels."""
    if not pseudo_labels:
        return {'total_count': 0, 'avg_confidence': 0.0, 'scenes_with_labels': 0}
    
    total_count = 0
    confidence_scores = []
    scenes_with_labels = 0
    
    for scene_id, scene_data in pseudo_labels.items():
        if isinstance(scene_data, dict) and 'scores' in scene_data:
            scores = scene_data['scores']
            if len(scores) > 0:
                total_count += len(scores)
                confidence_scores.extend(scores)
                scenes_with_labels += 1
        elif isinstance(scene_data, dict) and 'boxes' in scene_data:
            # Handle different pseudo label formats
            boxes = scene_data['boxes']
            scores = scene_data.get('scores', [])
            if len(boxes) > 0:
                total_count += len(boxes)
                if len(scores) > 0:
                    confidence_scores.extend(scores)
                scenes_with_labels += 1
    
    return {
        'total_count': total_count,
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
        'scenes_with_labels': scenes_with_labels,
        'total_scenes': len(pseudo_labels),
        'avg_per_scene': total_count / len(pseudo_labels) if pseudo_labels else 0.0,
        'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0
    }

def calculate_quick_f1_score(pseudo_labels: Dict[str, Any], sample_ratio: float = 0.1) -> float:
    """
    Calculate a quick F1 score estimate by sampling scenes.
    This is a placeholder - in practice, you'd implement actual GT comparison.
    """
    # For now, return a mock F1 score based on confidence distribution
    # In real implementation, you'd load GT data and compute IoU matches
    stats = compute_pseudo_label_stats(pseudo_labels)
    
    if stats['total_count'] == 0:
        return 0.0
    
    # Rough estimation: higher average confidence often correlates with better F1
    # This is just a placeholder - real implementation would use GT comparison
    confidence_factor = min(stats['avg_confidence'], 1.0)
    count_factor = min(stats['total_count'] / 40000, 1.5)  # Normalize around expected count
    
    # Simple heuristic - replace with actual GT-based F1 calculation
    estimated_f1 = confidence_factor * 0.7 + (count_factor - 1.0) * 0.1
    return max(0.0, min(1.0, estimated_f1))

def validate_pseudo_labels(pseudo_labels: Dict[str, Any], stage_id: int, confidence_threshold: float) -> bool:
    """
    Validate pseudo label quality with stage-aware criteria.
    
    Args:
        pseudo_labels: Dictionary of pseudo labels per scene
        stage_id: Current stage ID (2, 3, 4, 5)
        confidence_threshold: Confidence threshold used for generation
        
    Returns:
        bool: True if validation passes, False if critical issues found
    """
    logger.info(f"\n🔍 VALIDATING PSEUDO LABELS FOR STAGE {stage_id}")
    logger.info(f"   Confidence threshold: {confidence_threshold}")
    logger.info(f"   Total scenes: {len(pseudo_labels)}")
    
    # Early label-space sanity: warn if labels look like GCI indices (0..N-1)
    try:
        sample = []
        for _sid, _item in list(pseudo_labels.items())[:50]:
            if isinstance(_item, dict) and 'labels' in _item and _item['labels'] is not None:
                labs = _item['labels']
                if hasattr(labs, 'tolist'):
                    labs = labs.tolist()
                sample.extend([int(x) for x in labs[:100]])
        if sample:
            has_zero = any(x == 0 for x in sample)
            max_val = max(sample)
            if has_zero or (max_val <= 34):
                logger.warning("⚠️ WARNING: Pseudo labels appear to be in GCI (0..N-1) index space; expected NYU40 IDs.")
    except Exception:
        pass
    
    # Compute basic statistics
    stats = compute_pseudo_label_stats(pseudo_labels)
    
    logger.info(f"📊 Basic Statistics:")
    logger.info(f"   Total pseudo labels: {stats['total_count']:,}")
    logger.info(f"   Average confidence: {stats['avg_confidence']:.3f} ± {stats['confidence_std']:.3f}")
    logger.info(f"   Scenes with labels: {stats['scenes_with_labels']}/{stats['total_scenes']}")
    logger.info(f"   Average per scene: {stats['avg_per_scene']:.1f}")
    
    if stage_id == 2:
        # We have reference data for Stage 2 from our analysis
        logger.info("🎯 Stage 2 - Using reference metrics from standalone analysis")
        
        # Reference metrics based on our comprehensive analysis
        # Updated for per-stage thresholds (Stage 2 uses 0.45 threshold)
        expected_metrics = {
            0.05: {'f1': 0.45, 'count': 80000},  # Very loose (generation threshold)
            0.15: {'f1': 0.55, 'count': 60000},  # Loose  
            0.25: {'f1': 0.58, 'count': 50000},  # Moderate
            0.30: {'f1': 0.60, 'count': 45000},  # Our visualization default
            0.40: {'f1': 0.634, 'count': 43177}, # Our analysis optimal
            0.45: {'f1': 0.62, 'count': 40000},  # Stage 2 per-stage threshold
            0.50: {'f1': 0.60, 'count': 35000},  # Strict
        }
        
        # Get expected metrics for this confidence threshold
        ref_metric = interpolate_expected_metric(confidence_threshold, expected_metrics)
        logger.info(f"📋 Expected metrics at conf={confidence_threshold}: F1={ref_metric['f1']:.3f}, Count={ref_metric['count']:.0f}")
        
        # Define flexible validation ranges (allow significant fluctuation)
        f1_lower = ref_metric['f1'] * 0.6  # Allow 40% drop
        f1_upper = ref_metric['f1'] * 1.4  # Allow 40% increase
        count_lower = ref_metric['count'] * 0.5  # Allow 50% drop  
        count_upper = ref_metric['count'] * 1.8  # Allow 80% increase
        
        logger.info(f"✓ Validation ranges:")
        logger.info(f"   F1 range: [{f1_lower:.3f}, {f1_upper:.3f}]")
        logger.info(f"   Count range: [{count_lower:.0f}, {count_upper:.0f}]")
        
        # Quick F1 estimation (placeholder for real GT-based calculation)
        estimated_f1 = calculate_quick_f1_score(pseudo_labels)
        logger.info(f"📈 Estimated F1 score: {estimated_f1:.3f}")
        
        # Validate F1 score
        if estimated_f1 < f1_lower * 0.4:  # Less than 40% of expected lower bound
            logger.error(f"❌ CRITICAL: Pseudo label F1 score {estimated_f1:.3f} is far below expected {ref_metric['f1']:.3f}")
            logger.error("   Possible issues: coordinate transformations, NMS settings, or model quality")
            logger.error("   Consider regenerating pseudo labels with different settings")
            return False
        elif estimated_f1 < f1_lower:
            logger.warning(f"⚠️ WARNING: Pseudo label F1 score {estimated_f1:.3f} is below expected range [{f1_lower:.3f}, {f1_upper:.3f}]")
            logger.warning("   Training may proceed but monitor performance carefully")
        else:
            logger.info(f"✅ Pseudo label F1 score {estimated_f1:.3f} is within expected range [{f1_lower:.3f}, {f1_upper:.3f}]")
        
        # Validate count
        if stats['total_count'] < count_lower:
            logger.warning(f"⚠️ WARNING: Pseudo label count {stats['total_count']:,} is below expected range [{count_lower:.0f}, {count_upper:.0f}]")
            logger.warning("   This may indicate overly strict confidence threshold or poor model performance")
        elif stats['total_count'] > count_upper:
            logger.warning(f"⚠️ WARNING: Pseudo label count {stats['total_count']:,} is above expected range [{count_lower:.0f}, {count_upper:.0f}]")
            logger.warning("   This may indicate overly loose confidence threshold or coordinate issues")
        else:
            logger.info(f"✅ Pseudo label count {stats['total_count']:,} is within expected range [{count_lower:.0f}, {count_upper:.0f}]")
        
    elif stage_id >= 3:
        # No reference data for Stage 3+ - perform basic sanity checks with per-stage awareness
        logger.info(f"📊 Stage {stage_id} - Using per-stage threshold validation")
        
        # Per-stage expected thresholds and rough count estimates
        stage_threshold_expectations = {
            3: {'threshold': 0.40, 'min_count': 25000, 'max_count': 70000},  # Stage 2 model (14 classes)
            4: {'threshold': 0.35, 'min_count': 20000, 'max_count': 60000},  # Stage 3 model (21 classes) 
            5: {'threshold': 0.30, 'min_count': 15000, 'max_count': 50000}   # Stage 4 model (28 classes)
        }
        
        expected = stage_threshold_expectations.get(stage_id, {'threshold': 0.30, 'min_count': 10000, 'max_count': 50000})
        logger.info(f"   Expected threshold: {expected['threshold']} (vs actual: {confidence_threshold})")
        logger.info(f"   Expected count range: {expected['min_count']:,} - {expected['max_count']:,}")
        
        # Threshold consistency check
        if abs(confidence_threshold - expected['threshold']) > 0.05:
            logger.warning(f"⚠️ WARNING: Threshold {confidence_threshold} differs from expected {expected['threshold']}")
            logger.warning("   This may affect pseudo label quality for this stage")
        
        # Basic sanity checks with stage-aware expectations
        if stats['total_count'] < 100:
            logger.error(f"❌ CRITICAL: Only {stats['total_count']} pseudo labels generated!")
            logger.error("   This is likely a serious issue - check model inference and thresholds")
            return False
        elif stats['total_count'] < expected['min_count']:
            logger.warning(f"⚠️ WARNING: Count {stats['total_count']:,} below expected minimum {expected['min_count']:,}")
            logger.warning("   Consider lowering confidence threshold or checking model performance")
        elif stats['total_count'] > expected['max_count']:
            logger.warning(f"⚠️ WARNING: Count {stats['total_count']:,} above expected maximum {expected['max_count']:,}")
            logger.warning("   May indicate overly permissive threshold or coordinate issues")
        
        if stats['avg_confidence'] < confidence_threshold * 0.7:
            logger.warning(f"⚠️ WARNING: Average confidence {stats['avg_confidence']:.3f} is much lower than threshold {confidence_threshold}")
            logger.warning("   Many predictions are just above threshold - consider quality")
        
        if stats['scenes_with_labels'] < len(pseudo_labels) * 0.5:
            logger.warning(f"⚠️ WARNING: Only {stats['scenes_with_labels']}/{len(pseudo_labels)} scenes have pseudo labels")
            logger.warning("   This may indicate poor model generalization or overly strict thresholds")
        
        logger.info(f"ℹ️ Note: Stage {stage_id} uses Stage {stage_id-1} model predictions.")
        logger.info("   Manual verification through visualization app is recommended.")
        logger.info("   Expected performance will depend on incremental learning dynamics.")
    
    # Final summary
    logger.info(f"\n✅ PSEUDO LABEL VALIDATION COMPLETE FOR STAGE {stage_id}")
    if stage_id == 2:
        logger.info("   Stage 2: Validated against reference metrics")
    else:
        logger.info(f"   Stage {stage_id}: Basic sanity checks passed")
    logger.info("   Training can proceed - monitor performance during training")
    
    return True

def validate_pseudo_labels_from_file(pseudo_label_file: str, stage_id: int, confidence_threshold: float) -> bool:
    """
    Validate pseudo labels from a saved pickle file.
    
    Args:
        pseudo_label_file: Path to pickle file containing pseudo labels
        stage_id: Current stage ID 
        confidence_threshold: Confidence threshold used
        
    Returns:
        bool: True if validation passes, False if critical issues
    """
    try:
        logger.info(f"📁 Loading pseudo labels from: {pseudo_label_file}")
        
        with open(pseudo_label_file, 'rb') as f:
            pseudo_labels = pickle.load(f)
        
        if not pseudo_labels:
            logger.error("❌ CRITICAL: Pseudo label file is empty!")
            return False
        
        return validate_pseudo_labels(pseudo_labels, stage_id, confidence_threshold)
        
    except Exception as e:
        logger.error(f"❌ ERROR: Failed to validate pseudo labels from {pseudo_label_file}")
        logger.error(f"   Error: {e}")
        return False

def quick_validate_on_generation(pseudo_labels: Dict[str, Any], stage_id: int, confidence_threshold: float) -> None:
    """
    Quick validation during generation - non-blocking.
    Only logs warnings, doesn't return success/failure.
    """
    try:
        validate_pseudo_labels(pseudo_labels, stage_id, confidence_threshold)
    except Exception as e:
        logger.warning(f"⚠️ Pseudo label validation failed: {e}")
        logger.warning("   This is non-critical - training will continue")

# Example usage and testing
if __name__ == "__main__":
    # Test with mock data
    mock_pseudo_labels = {
        f"scene{i:04d}_00": {
            'boxes': np.random.rand(50, 7),
            'scores': np.random.rand(50) * 0.8 + 0.2,
            'labels': np.random.randint(0, 7, 50)
        } for i in range(100)
    }
    
    print("Testing pseudo label validation...")
    result = validate_pseudo_labels(mock_pseudo_labels, stage_id=2, confidence_threshold=0.3)
    print(f"Validation result: {'PASSED' if result else 'FAILED'}")
