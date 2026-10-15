"""
Incremental Learning Evaluation for Indoor 3D Object Detection

Modified evaluation protocol that only evaluates on classes seen so far
in the incremental learning process.
"""

import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .indoor_eval import indoor_eval, _format_eval_banner_tags


def _as_int_list(values: Sequence[int]) -> List[int]:
    return [int(x) for x in values]


def _filter_gt_anno_by_seen_classes(gt_anno: Dict[str, Any],
                                   seen_classes: Sequence[int]) -> Dict[str, Any]:
    """Filter a single indoor-format GT annotation dict by seen classes.

    This supports the GT format consumed by `mmdet3d.core.evaluation.indoor_eval`:
      - 'gt_num': int
      - 'class': np.ndarray[int]
      - 'gt_boxes_upright_depth': np.ndarray[float]
    """
    gt_num = int(gt_anno.get('gt_num', 0) or 0)
    if gt_num <= 0 or 'class' not in gt_anno:
        return gt_anno

    labels = np.asarray(gt_anno['class']).reshape(-1).astype(np.int64)
    if labels.shape[0] != gt_num:
        gt_num = int(labels.shape[0])
    if gt_num <= 0:
        return {**gt_anno, 'gt_num': 0, 'class': labels}

    keep = np.isin(labels, np.asarray(_as_int_list(seen_classes), dtype=np.int64))
    if keep.all():
        return gt_anno

    filtered = dict(gt_anno)
    filtered['class'] = labels[keep]
    filtered['gt_num'] = int(keep.sum())

    boxes = gt_anno.get('gt_boxes_upright_depth', None)
    if isinstance(boxes, np.ndarray) and boxes.shape[0] == labels.shape[0]:
        filtered['gt_boxes_upright_depth'] = boxes[keep]

    # Filter any other per-object arrays aligned with gt_num.
    for key, value in gt_anno.items():
        if key in {'gt_num', 'class', 'gt_boxes_upright_depth'}:
            continue
        if isinstance(value, np.ndarray) and value.shape[0] == labels.shape[0]:
            filtered[key] = value[keep]

    return filtered


def _filter_det_anno_by_seen_classes(det_anno: Dict[str, Any],
                                    seen_classes: Sequence[int]) -> Dict[str, Any]:
    """Filter a single detection dict by seen classes."""
    if 'labels_3d' not in det_anno:
        return det_anno

    labels = det_anno['labels_3d']
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy().reshape(-1).astype(np.int64)
    else:
        labels_np = np.asarray(labels).reshape(-1).astype(np.int64)

    keep_np = np.isin(labels_np, np.asarray(_as_int_list(seen_classes), dtype=np.int64))
    if keep_np.all():
        return det_anno

    filtered = dict(det_anno)
    if isinstance(labels, torch.Tensor):
        keep = torch.from_numpy(keep_np).to(device=labels.device)
        filtered['labels_3d'] = labels[keep]
        if 'scores_3d' in det_anno and isinstance(det_anno['scores_3d'], torch.Tensor):
            filtered['scores_3d'] = det_anno['scores_3d'][keep]
        if 'boxes_3d' in det_anno:
            filtered['boxes_3d'] = det_anno['boxes_3d'][keep]
    else:
        keep = keep_np.astype(bool)
        filtered['labels_3d'] = labels_np[keep]
        if 'scores_3d' in det_anno:
            filtered['scores_3d'] = np.asarray(det_anno['scores_3d']).reshape(-1)[keep]
        if 'boxes_3d' in det_anno:
            try:
                filtered['boxes_3d'] = det_anno['boxes_3d'][keep]
            except Exception:
                filtered['boxes_3d'] = det_anno['boxes_3d']

    return filtered


def _build_stage_cohort_table(ret_dict: Dict[str, float],
                              iou_thrs: Sequence[float],
                              seen_classes: Sequence[int],
                              class_names: Sequence[str],
                              class_meta: Optional[Dict[int, Dict[str, Any]]]
                              ) -> Tuple[Dict[str, float], Optional[str]]:
    """Compute per-stage cohort averages and return (metrics, ascii_table)."""
    if not isinstance(class_meta, dict) or not class_meta:
        return {}, None

    stage_to_classes: Dict[int, List[int]] = {}
    for cls_idx in seen_classes:
        meta = class_meta.get(int(cls_idx), {})
        stage = meta.get('stage', None)
        if stage in (None, '', -1):
            continue
        try:
            stage_id = int(stage)
        except Exception:
            continue
        stage_to_classes.setdefault(stage_id, []).append(int(cls_idx))

    if not stage_to_classes:
        return {}, None

    stage_ids = sorted(stage_to_classes.keys())
    min_stage = int(min(stage_ids))
    max_stage = int(max(stage_ids))

    header = ['Cohort', '#Cls']
    for thr in iou_thrs:
        header += [f'mAP_{float(thr):.2f}', f'mAR_{float(thr):.2f}']

    metrics: Dict[str, float] = {}
    rows: List[List[str]] = []

    def _mean(keys: List[str]) -> float:
        vals = [float(ret_dict.get(k, 0.0)) for k in keys]
        return float(np.mean(vals)) if vals else 0.0

    for sid in stage_ids:
        cohort_classes = stage_to_classes[sid]
        row: List[str] = [f'Stage {sid}', str(len(cohort_classes))]
        for thr in iou_thrs:
            ap_keys = [f'{class_names[c]}_AP_{float(thr):.2f}' for c in cohort_classes
                       if 0 <= int(c) < len(class_names)]
            rec_keys = [f'{class_names[c]}_rec_{float(thr):.2f}' for c in cohort_classes
                        if 0 <= int(c) < len(class_names)]

            mean_ap = _mean(ap_keys)
            mean_rec = _mean(rec_keys)
            metrics[f'cohort_stage_{sid}_mAP_{float(thr):.2f}'] = mean_ap
            metrics[f'cohort_stage_{sid}_mAR_{float(thr):.2f}'] = mean_rec
            row += [f'{mean_ap:.4f}', f'{mean_rec:.4f}']
        rows.append(row)

    seen_row: List[str] = [f'Seen (Stage {min_stage}-{max_stage})', str(len(seen_classes))]
    for thr in iou_thrs:
        ap_keys = [f'{class_names[c]}_AP_{float(thr):.2f}' for c in seen_classes
                   if 0 <= int(c) < len(class_names)]
        rec_keys = [f'{class_names[c]}_rec_{float(thr):.2f}' for c in seen_classes
                    if 0 <= int(c) < len(class_names)]
        mean_ap = _mean(ap_keys)
        mean_rec = _mean(rec_keys)
        metrics[f'cohort_seen_mAP_{float(thr):.2f}'] = mean_ap
        metrics[f'cohort_seen_mAR_{float(thr):.2f}'] = mean_rec
        seen_row += [f'{mean_ap:.4f}', f'{mean_rec:.4f}']
    rows.append(seen_row)

    table = AsciiTable([header] + rows)
    table.inner_footing_row_border = True
    return metrics, table.table


def incremental_indoor_eval(gt_annos: List[Dict],
                           dt_annos: List[Dict], 
                           metric: List[str],
                           seen_classes: List[int],
                           class_names: List[str],
                           stage_idx: int = 0,
                           logger=None,
                           box_type_3d=None,
                           box_mode_3d=None,
                           class_meta: Dict[int, Dict[str, Any]] = None,
                           eval_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Evaluate 3D detection results for incremental learning.
    
    Only evaluates on classes that have been seen so far in the incremental learning process.
    
    Args:
        gt_annos (List[Dict]): Ground truth annotations
        dt_annos (List[Dict]): Detection results
        metric (List[str]): Evaluation metrics ('mAP', 'AR', etc.)
        seen_classes (List[int]): List of class indices seen so far
        class_names (List[str]): List of all class names
        stage_idx (int): Current stage index (for logging)
        logger: Logger for outputting results
        box_type_3d: 3D box type for evaluation
        box_mode_3d: 3D box mode for evaluation
        
    Returns:
        Dict[str, float]: Evaluation results
    """
    
    if logger:
        logger.info(f"Incremental Evaluation - Stage {stage_idx + 1}")
        logger.info(f"Evaluating on {len(seen_classes)} seen classes: {seen_classes}")

    seen_classes = sorted({int(x) for x in seen_classes})
    
    # Filter both GT and detections to seen classes (keeps aggregates consistent and
    # avoids undefined per-class AP for unseen classes with 0 GT).
    filtered_gt_annos = [_filter_gt_anno_by_seen_classes(g, seen_classes) for g in gt_annos]
    filtered_dt_annos = [_filter_det_anno_by_seen_classes(d, seen_classes) for d in dt_annos]
    
    # CRITICAL FIX: Create label2cat with proper class indices to ensure consistent ordering
    # Use actual class indices as keys (not enumeration) to maintain proper order
    label2cat = {}
    for class_idx in seen_classes:
        if 0 <= int(class_idx) < len(class_names):
            label2cat[int(class_idx)] = class_names[int(class_idx)]
    
    # Sort by class index to ensure consistent ordering across all epochs
    # This ensures classes always appear in order: 0, 1, 2, ..., 13, 14, 15, ...
    label2cat = dict(sorted(label2cat.items()))
    
    if logger:
        logger.info(f"Evaluation class ordering (indices): {list(label2cat.keys())}")
        logger.info(f"Evaluation class ordering (names): {list(label2cat.values())}")
    
    # Run standard indoor evaluation on filtered data
    try:
        ctx = None
        if isinstance(eval_context, dict):
            ctx = dict(eval_context)
            try:
                ctx.setdefault('stage_id', max(0, int(stage_idx)) + 1)
            except Exception:
                pass

        ret_dict = indoor_eval(
            filtered_gt_annos,
            filtered_dt_annos,
            metric,
            label2cat,
            logger=logger,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            class_meta=class_meta,
            eval_context=ctx,
        )

        cohort_metrics, cohort_table = _build_stage_cohort_table(
            ret_dict,
            iou_thrs=metric,
            seen_classes=seen_classes,
            class_names=class_names,
            class_meta=class_meta,
        )
        if cohort_metrics:
            ret_dict.update(cohort_metrics)
            if logger and cohort_table:
                if ctx is not None:
                    banner = _format_eval_banner_tags(ctx)
                    banner = f"{banner} Cohort summary table:"
                    print_log(banner + '\n' + cohort_table, logger=logger)
                else:
                    print_log('\n' + cohort_table, logger=logger)
        
        # Add incremental learning specific information to results
        incremental_results = {}
        for key, value in ret_dict.items():
            incremental_results[f"stage_{stage_idx}_{key}"] = value
        
        # Add summary metrics
        incremental_results[f"stage_{stage_idx}_seen_classes"] = len(seen_classes)
        incremental_results[f"stage_{stage_idx}_class_list"] = seen_classes
        
        if logger:
            logger.info(f"Stage {stage_idx + 1} Evaluation Results:")
            # Keep only aggregate metrics here; detailed per-class metrics are shown in the table above
            for key in sorted(ret_dict.keys()):
                if isinstance(ret_dict[key], float) and (
                    key.startswith('mAP_') or key.startswith('mAR_') or key.startswith('AR@')
                    or key.startswith('cohort_')
                ):
                    logger.info(f"  {key}: {ret_dict[key]:.4f}")
        
        return incremental_results
        
    except Exception as e:
        if logger:
            logger.error(f"Evaluation failed for stage {stage_idx}: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return {}


def aggregate_incremental_results(stage_results: List[Dict[str, float]], 
                                 logger=None) -> Dict[str, float]:
    """Aggregate evaluation results across all incremental stages.
    
    Args:
        stage_results (List[Dict]): Results from each stage
        logger: Logger for output
        
    Returns:
        Dict[str, float]: Aggregated results
    """
    if not stage_results:
        return {}
    
    aggregated = {}
    
    # Track metrics across stages
    metric_history = {}
    
    for stage_idx, results in enumerate(stage_results):
        for key, value in results.items():
            if isinstance(value, (int, float)):
                metric_name = key.replace(f"stage_{stage_idx}_", "")
                if metric_name not in metric_history:
                    metric_history[metric_name] = []
                metric_history[metric_name].append(value)
    
    # Calculate aggregate metrics
    for metric_name, values in metric_history.items():
        if metric_name in ['seen_classes', 'class_list']:
            continue  # Skip non-numeric metrics
            
        if values:
            aggregated[f"final_{metric_name}"] = values[-1]  # Final stage value
            aggregated[f"avg_{metric_name}"] = np.mean(values)  # Average across stages
            
            # Calculate forgetting measure (difference between max and final)
            if len(values) > 1:
                aggregated[f"forgetting_{metric_name}"] = max(values) - values[-1]
    
    # Calculate total classes learned
    total_classes = 0
    for results in stage_results:
        seen_classes_key = [k for k in results.keys() if k.endswith('_seen_classes')]
        if seen_classes_key:
            total_classes = max(total_classes, results[seen_classes_key[0]])
    
    aggregated["total_classes_learned"] = total_classes
    aggregated["total_stages"] = len(stage_results)
    
    if logger:
        logger.info("="*50)
        logger.info("INCREMENTAL LEARNING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total stages: {len(stage_results)}")
        logger.info(f"Total classes learned: {total_classes}")
        
        for key, value in aggregated.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
        logger.info("="*50)
    
    return aggregated


def print_incremental_progress(stage_results: List[Dict[str, float]], 
                              class_names: List[str]):
    """Print a progress table showing performance across stages."""
    if not stage_results:
        return
    
    print("\nIncremental Learning Progress:")
    print("-" * 80)
    print(f"{'Stage':<8} {'Classes':<10} {'mAP_0.25':<10} {'mAP_0.5':<10} {'AR':<10}")
    print("-" * 80)
    
    for stage_idx, results in enumerate(stage_results):
        seen_classes = results.get(f"stage_{stage_idx}_seen_classes", 0)
        map_25 = results.get(f"stage_{stage_idx}_mAP_0.25", 0.0)
        map_50 = results.get(f"stage_{stage_idx}_mAP_0.5", 0.0)
        ar = results.get(f"stage_{stage_idx}_AR@100", 0.0)
        
        print(f"{stage_idx + 1:<8} {seen_classes:<10} {map_25:<10.4f} {map_50:<10.4f} {ar:<10.4f}")
    
    print("-" * 80)
    
    # Show final class distribution if available
    if stage_results:
        final_results = stage_results[-1]
        class_list_key = [k for k in final_results.keys() if k.endswith('_class_list')]
        if class_list_key and class_names:
            final_classes = final_results[class_list_key[0]]
            print(f"\nFinal learned classes ({len(final_classes)}):")
            learned_names = [class_names[i] for i in final_classes if i < len(class_names)]
            print(", ".join(learned_names))
            print()
