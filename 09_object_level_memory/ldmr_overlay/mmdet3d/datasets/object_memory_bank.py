"""Self-contained object-level replay memory for incremental 3D detection.

Unlike :mod:`scene_memory_bank`, this bank stores individual object crops. A
stored exemplar owns its point cloud in a local, gravity-centred coordinate
frame and remains usable without retaining or reopening its source scene.
"""

from __future__ import annotations

import json
import pickle
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class ObjectMemoryBank:
    """Balanced bank of self-contained object exemplars.

    ``cache_extracted_objects`` and ``max_cache_size`` remain accepted for old
    configs. Object crops are now authoritative memory, not an evictable cache.
    """

    def __init__(self,
                 exemplars_per_class: int = 20,
                 selection_strategy: str = 'random',
                 max_total_exemplars: int = 1000,
                 min_points: int = 5,
                 crop_margin: float = 0.0,
                 floor_percentile: float = 1.0,
                 random_seed: int = 0,
                 learning_dynamics_update: Optional[Dict[str, Any]] = None,
                 learning_dynamics_design2: Optional[Dict[str, Any]] = None,
                 work_dir: Optional[str] = None,
                 cache_extracted_objects: bool = True,
                 max_cache_size: int = 200,
                 **kwargs):
        if kwargs:
            raise TypeError(
                'Unknown ObjectMemoryBank arguments: ' + ', '.join(sorted(kwargs)))
        if int(exemplars_per_class) <= 0 or int(max_total_exemplars) <= 0:
            raise ValueError('Object-memory budgets must be positive.')
        if selection_strategy not in (
                'random', 'largest_point_count', 'learning_dynamics_design2'):
            raise ValueError(
                "selection_strategy must be 'random', 'largest_point_count', "
                "or 'learning_dynamics_design2', "
                f'got {selection_strategy!r}.')
        if float(crop_margin) < 0:
            raise ValueError('crop_margin must be non-negative.')
        if not 0.0 <= float(floor_percentile) <= 100.0:
            raise ValueError('floor_percentile must be in [0, 100].')

        self.exemplars_per_class = int(exemplars_per_class)
        self.selection_strategy_name = str(selection_strategy)
        self.max_total_exemplars = int(max_total_exemplars)
        self.min_points = int(min_points)
        self.crop_margin = float(crop_margin)
        self.floor_percentile = float(floor_percentile)
        self.random_seed = int(random_seed)
        # Keep the same attribute names as SceneMemoryBank so the trainer's
        # tested learning-dynamics evaluator can serve either memory unit.
        self.selection_strategy = self.selection_strategy_name
        self.learning_dynamics_update = dict(
            learning_dynamics_update or {})
        self.learning_dynamics_design2 = dict(
            learning_dynamics_design2 or {})
        self.learning_dynamics_design1_q_metric = str(
            self.learning_dynamics_design2.get('q_metric', 'recall')
        ).strip().lower()
        if self.learning_dynamics_design1_q_metric not in ('f1', 'recall'):
            raise ValueError(
                'learning_dynamics_design2.q_metric must be one of '
                "['f1', 'recall'].")
        self.learning_dynamics_design_version = (
            2 if self.selection_strategy_name == 'learning_dynamics_design2'
            else 1)
        self.learning_dynamics_design1_supply_scaling_mode = str(
            self.learning_dynamics_design2.get('supply_scaling_mode', 'raw'))
        self.learning_dynamics_design2_w_max = float(
            self.learning_dynamics_design2.get('w_max', 10.0))
        self.design2_redundancy_lambda = float(
            self.learning_dynamics_design2.get('redundancy_lambda', 0.5))
        self.design2_redundancy_topk = int(
            self.learning_dynamics_design2.get('redundancy_topk', 5))
        if not 0.0 <= self.design2_redundancy_lambda <= 1.0:
            raise ValueError(
                'learning_dynamics_design2.redundancy_lambda must be in [0, 1].')
        if self.design2_redundancy_topk <= 0:
            raise ValueError(
                'learning_dynamics_design2.redundancy_topk must be positive.')
        self.work_dir = work_dir
        self.exemplars: Dict[int, List[Dict[str, Any]]] = {}
        self.exemplar_count = 0
        self.previous_classes: List[int] = []
        self.dataset_ref = None
        self.loaded_stage_id: Optional[int] = None

        # Compatibility attributes used by legacy diagnostics.
        self.cache_extracted_objects = bool(cache_extracted_objects)
        self.max_cache_size = int(max_cache_size) * 1024 * 1024
        self.point_cloud_cache: Dict[Any, np.ndarray] = {}
        self.cache_size_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.reduction_history: List[Dict[str, Any]] = []
        self.removed_exemplars: Dict[int, List[Dict[str, Any]]] = {}

    @staticmethod
    def extract_object_points(scene_points: np.ndarray,
                              bbox: np.ndarray,
                              debug_info: str = 'unknown',
                              crop_margin: float = 0.0) -> np.ndarray:
        """Crop an oriented box and return points in its local box frame.

        Boxes are ``[cx, cy, cz, dx, dy, dz, yaw]`` with a gravity centre.
        Returned XYZ values are centred on the box and aligned with its axes.
        """
        del debug_info
        points = np.asarray(scene_points)
        box = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if points.ndim != 2:
            raise ValueError(f'scene_points must be 2D, got {points.shape}.')
        if box.size < 6:
            raise ValueError(f'bbox must have at least 6 values, got {box.size}.')
        if points.shape[0] == 0:
            return np.empty((0, points.shape[1]), dtype=points.dtype)
        if not np.all(np.isfinite(box[:6])) or np.any(box[3:6] <= 0):
            return np.empty((0, points.shape[1]), dtype=points.dtype)

        local_xyz = points[:, :3].astype(np.float32) - box[:3]
        yaw = float(box[6]) if box.size >= 7 else 0.0
        if abs(yaw) > 1e-7:
            c, s = np.cos(yaw), np.sin(yaw)
            # Row-vector rotation by -yaw: world coordinates -> box axes.
            local_xyz = local_xyz @ np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32)
        half = box[3:6] * (0.5 + float(crop_margin))
        mask = np.all(np.abs(local_xyz) <= half + 1e-6, axis=1)
        cropped = points[mask].copy()
        cropped[:, :3] = local_xyz[mask]
        return cropped

    @staticmethod
    def _nested_get(mapping: Dict[str, Any], *keys: int) -> Any:
        value: Any = mapping
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key, value.get(str(key)))
        return value

    @staticmethod
    def _percentile_rank(values: np.ndarray) -> np.ndarray:
        """Return deterministic ranks in [0, 1], averaging tied ranks."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) <= 1:
            return np.ones_like(values)
        order = np.argsort(values, kind='stable')
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while (end < len(values)
                   and values[order[end]] == values[order[start]]):
                end += 1
            mean_rank = 0.5 * (start + end - 1)
            ranks[order[start:end]] = mean_rank / float(len(values) - 1)
            start = end
        return ranks

    def rank_learning_dynamics_design2(
            self,
            candidates: Sequence[Dict[str, Any]],
            class_id: int,
            stage_id: int,
            payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank object candidates using LDMR Design-2 scene/class terms.

        LDMR measures detector learning dynamics on complete source scenes. An
        object inherits the term for its source ``(scene, stage, class)`` seat;
        this avoids evaluating an isolated crop as if it were a valid detector
        input. Within the fixed class quota, candidates are greedily ranked by
        learnability with the same positive-cosine redundancy penalty used by
        Design-2 scene selection.
        """
        if not isinstance(payload, dict):
            raise ValueError('Design-2 object selection requires a score payload.')
        seat_terms = payload.get('seat_class_terms')
        if not isinstance(seat_terms, dict) or not seat_terms:
            raise ValueError(
                "Design-2 object selection requires non-empty 'seat_class_terms'.")

        class_id = int(class_id)
        stage_id = int(stage_id)
        class_need = self._nested_get(
            payload.get('class_need', {}), class_id)
        class_need = 0.0 if class_need is None else float(class_need)
        rows = []
        missing = []
        for index, candidate in enumerate(candidates):
            scene_id = str(candidate.get('scene_id', ''))
            by_scene = seat_terms.get(scene_id)
            term = self._nested_get(by_scene or {}, stage_id, class_id)
            if not isinstance(term, dict):
                missing.append((scene_id, stage_id, class_id))
                continue
            g = float(term.get('g', 0.0))
            r_best = float(term.get('r_best', 0.0))
            d = float(term.get('d', 0.0))
            u = float(term.get('u', g * r_best + d))
            vals = np.asarray([g, r_best, d, u], dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                raise ValueError(
                    'Non-finite Design-2 term for object candidate '
                    f'{scene_id}@{stage_id}/class={class_id}: {term}')
            rows.append(dict(
                index=int(index), scene_id=scene_id,
                x1=float(g * r_best), x2=max(0.0, d),
                unary=max(0.0, class_need * u), term=dict(term)))
        if missing:
            raise ValueError(
                'Design-2 payload is missing source scene/class terms for '
                f'{len(missing)} object candidates (example={missing[0]}).')
        if not rows:
            return []

        x1 = np.asarray([row['x1'] for row in rows], dtype=np.float64)
        x2 = np.asarray([row['x2'] for row in rows], dtype=np.float64)
        e1 = (x1 - x1.mean()) / (x1.std() if x1.std() > 1e-12 else 1.0)
        e2 = (x2 - x2.mean()) / (x2.std() if x2.std() > 1e-12 else 1.0)
        embeddings = np.stack([e1, e2], axis=1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(
            embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)
        unary_rank = self._percentile_rank(
            np.asarray([row['unary'] for row in rows], dtype=np.float64))

        rng = np.random.RandomState(
            self.random_seed + 1009 * class_id + 1000003 * stage_id)
        tie_break = rng.random_sample(len(rows))
        remaining = set(range(len(rows)))
        selected = []
        selected_scene_ids = set()
        lam = self.design2_redundancy_lambda
        topk = self.design2_redundancy_topk
        while remaining:
            # Source-level terms are identical for multiple same-class objects
            # from one scene. Spend seats on distinct sources first; duplicates
            # remain available only when the class has too few source scenes.
            distinct_source_remaining = {
                i for i in remaining
                if rows[i]['scene_id'] not in selected_scene_ids
            }
            eligible = distinct_source_remaining or remaining
            scored = []
            for i in eligible:
                redundancy = 0.0
                if selected:
                    similarities = np.maximum(
                        0.0, embeddings[selected] @ embeddings[i])
                    similarities = np.sort(similarities)[::-1][:topk]
                    redundancy = float(similarities.mean())
                score = (1.0 - lam) * float(unary_rank[i]) - lam * redundancy
                scored.append((score, -float(tie_break[i]), -i, i, redundancy))
            _, _, _, best, redundancy = max(scored)
            selected.append(int(best))
            remaining.remove(int(best))
            selected_scene_ids.add(rows[best]['scene_id'])
            rows[best]['redundancy'] = float(redundancy)
            rows[best]['selection_score'] = float(
                (1.0 - lam) * unary_rank[best] - lam * redundancy)

        ranked = []
        for rank, row_index in enumerate(selected):
            row = rows[row_index]
            candidate = dict(candidates[row['index']])
            candidate['learning_dynamics_design2'] = {
                'rank': int(rank),
                'class_need': float(class_need),
                'unary': float(row['unary']),
                'unary_percentile': float(unary_rank[row_index]),
                'embedding': embeddings[row_index].astype(np.float32),
                'redundancy': float(row['redundancy']),
                'selection_score': float(row['selection_score']),
                'source_term': row['term'],
            }
            ranked.append(candidate)
        return ranked

    def _select(self, candidates: List[Dict[str, Any]], count: int,
                class_id: int, stage_id: Optional[int] = None,
                learning_dynamics_design2_payload: Optional[Dict[str, Any]] = None
                ) -> List[Dict[str, Any]]:
        if self.selection_strategy_name == 'learning_dynamics_design2':
            if stage_id is None:
                raise ValueError('Design-2 object selection requires stage_id.')
            ranked = self.rank_learning_dynamics_design2(
                candidates, class_id, int(stage_id),
                learning_dynamics_design2_payload)
            return ranked[:count]
        if count >= len(candidates):
            indices = np.arange(len(candidates))
        elif self.selection_strategy_name == 'largest_point_count':
            indices = np.argsort(
                [-int(x['point_count']) for x in candidates], kind='stable')[:count]
        elif self.selection_strategy_name == 'random':
            # Per-class seed makes selection independent of class iteration order.
            rng = np.random.RandomState(self.random_seed + 1009 * int(class_id))
            indices = np.sort(rng.choice(len(candidates), size=count, replace=False))
        return [candidates[int(i)] for i in indices]

    def add_exemplars(self,
                      class_id: int,
                      objects: Sequence[Dict[str, Any]],
                      confidences: Optional[Sequence[float]] = None,
                      scene_points_dict: Optional[Dict[str, np.ndarray]] = None,
                      debug_save_dir: Optional[str] = None,
                      stage_id: Optional[int] = None,
                      learning_dynamics_design2_payload: Optional[
                          Dict[str, Any]] = None) -> int:
        """Select and store complete crops for one class.

        Each input may provide already-local ``points``, or a source ``scene_id``
        and ``bbox`` while ``scene_points_dict`` supplies the full scene.
        """
        class_id = int(class_id)
        candidates: List[Dict[str, Any]] = []
        for i, obj in enumerate(objects):
            box = np.asarray(obj['bbox'], dtype=np.float32).reshape(-1)
            scene_id = str(obj.get('scene_id', ''))
            points = obj.get('points')
            if points is None and scene_points_dict is not None:
                scene_points = scene_points_dict.get(scene_id)
                if scene_points is not None:
                    points = self.extract_object_points(
                        scene_points, box, crop_margin=self.crop_margin)
                    obj = dict(obj)
                    source_floor = float(np.percentile(
                        np.asarray(scene_points)[:, 2], self.floor_percentile))
                    obj['source_floor_offset'] = float(
                        box[2] - box[5] * 0.5 - source_floor)
            if points is None:
                continue
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 3 or len(points) < self.min_points:
                continue
            confidence = obj.get('confidence', None)
            if confidence is None:
                confidence = confidences[i] if confidences is not None else 1.0
            candidates.append({
                'scene_id': scene_id,
                'object_idx': int(obj.get('object_idx', i)),
                'class_id': class_id,
                'bbox': box.copy(),
                'box_size': box[3:6].copy(),
                'yaw': float(box[6]) if box.size >= 7 else 0.0,
                'confidence': float(confidence),
                'nyu40_id': obj.get('nyu40_id'),
                'stage_id': int(stage_id) if stage_id is not None else None,
                'point_count': int(len(points)),
                'points': points.copy(),
                'coordinate_frame': 'object_local_gravity_center',
                # Bottom-of-box height relative to the robust source-scene
                # floor. This compact context is needed for wall-mounted and
                # elevated objects; forcing every crop onto the floor changes
                # the task semantics for classes such as pictures and lamps.
                'source_floor_offset': float(
                    obj.get('source_floor_offset', 0.0)),
            })

        selected = self._select(
            candidates, min(len(candidates), self.exemplars_per_class), class_id,
            stage_id=stage_id,
            learning_dynamics_design2_payload=learning_dynamics_design2_payload)
        self.exemplars[class_id] = selected
        self._update_exemplar_count()
        self._reduce_exemplars()
        self._rebuild_cache()
        if debug_save_dir is not None:
            self._save_debug_class(Path(debug_save_dir), class_id, selected, stage_id)
        return len(selected)

    def get_exemplars(self, class_ids: Iterable[int]) -> List[Dict[str, Any]]:
        return [x for cid in class_ids for x in self.exemplars.get(int(cid), [])]

    def get_all_exemplars(self) -> List[Dict[str, Any]]:
        return self.get_exemplars(sorted(self.exemplars))

    def get_exemplar_points(self, exemplar: Dict[str, Any], **kwargs) -> Optional[np.ndarray]:
        del kwargs
        points = exemplar.get('points')
        return None if points is None else np.asarray(points).copy()

    def get_class_exemplar_count(self, class_id: int) -> int:
        return len(self.exemplars.get(int(class_id), []))

    def get_total_exemplar_count(self) -> int:
        return self.exemplar_count

    def get_stored_classes(self) -> List[int]:
        return sorted(self.exemplars)

    @staticmethod
    def source_seat_id(exemplar: Dict[str, Any]) -> str:
        """Return the LDMR-compatible source-scene seat identifier."""
        stage_id = exemplar.get('stage_id')
        if stage_id is None:
            raise ValueError('Object exemplar has no source stage_id.')
        return f"{str(exemplar.get('scene_id', ''))}_stage{int(stage_id)}"

    def apply_source_seat_replay_weights(
            self, weights_by_source_seat: Dict[str, float],
            *, strict: bool = True) -> Dict[str, int]:
        """Transfer LDMR reviewing weights from source scenes to objects.

        Until carrier-scene evaluation is implemented, all objects originating
        from a scene seat inherit that seat's tested ``ld_drop`` weight.
        """
        if not isinstance(weights_by_source_seat, dict):
            raise TypeError('weights_by_source_seat must be a dict.')
        applied = 0
        missing = []
        for exemplar in self.get_all_exemplars():
            uid = self.source_seat_id(exemplar)
            if uid not in weights_by_source_seat:
                missing.append(uid)
                continue
            weight = float(weights_by_source_seat[uid])
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f'Invalid replay weight for source seat {uid}: {weight!r}.')
            exemplar['replay_weight'] = weight
            applied += 1
        if strict and missing:
            raise ValueError(
                'Missing source-seat replay weights for '
                f'{len(missing)} object exemplars (example={missing[0]}).')
        self._rebuild_cache()
        return {'applied': int(applied), 'missing': int(len(missing))}

    def list_source_scene_entries(
            self, max_save_stage: Optional[int] = None) -> List[Dict[str, Any]]:
        """Materialize unique source-scene seats for optional reviewing.

        Object replay remains self-contained. Reviewing is an analysis/training
        extension that evaluates the original carrier scenes so each object can
        inherit a detector-level LDMR drop weight from its source seat.
        """
        dataset = self.dataset_ref
        source_infos = getattr(dataset, '_object_scene_info_by_id', None)
        if not isinstance(source_infos, dict):
            raise RuntimeError(
                'Object source-scene reviewing requires a dataset reference '
                'with _object_scene_info_by_id.')

        entries = []
        seen = set()
        missing = []
        for exemplar in self.get_all_exemplars():
            stage_id = exemplar.get('stage_id')
            if stage_id is None:
                raise ValueError('Object exemplar has no source stage_id.')
            stage_id = int(stage_id)
            if max_save_stage is not None and stage_id > int(max_save_stage):
                continue
            scene_id = str(exemplar.get('scene_id', ''))
            key = (scene_id, stage_id)
            if key in seen:
                continue
            seen.add(key)
            info = source_infos.get(scene_id)
            if not isinstance(info, dict):
                missing.append(key)
                continue
            entries.append({
                'scene_id': scene_id,
                'save_stage': stage_id,
                'snapshot': {'data_info': copy.deepcopy(info)},
            })
        if missing:
            raise RuntimeError(
                'Missing original source-scene metadata for '
                f'{len(missing)} object-memory seats (example={missing[0]}).')
        entries.sort(key=lambda row: (
            int(row['save_stage']), str(row['scene_id'])))
        return entries

    def clear_class_exemplars(self, class_id: int) -> None:
        self.exemplars.pop(int(class_id), None)
        self._update_exemplar_count()
        self._rebuild_cache()

    def clear_all_exemplars(self) -> None:
        self.exemplars.clear()
        self._update_exemplar_count()
        self._rebuild_cache()

    def _update_exemplar_count(self) -> None:
        self.exemplar_count = sum(len(v) for v in self.exemplars.values())

    def _reduce_exemplars(self) -> None:
        """Enforce the global budget with balanced round-robin allocation."""
        before = self.exemplar_count
        if before <= self.max_total_exemplars:
            return
        classes = sorted(self.exemplars)
        kept = {cid: [] for cid in classes}
        kept_count = 0
        max_depth = max(map(len, self.exemplars.values()), default=0)
        for depth in range(max_depth):
            for cid in classes:
                if depth >= len(self.exemplars[cid]):
                    continue
                if kept_count >= self.max_total_exemplars:
                    break
                kept[cid].append(self.exemplars[cid][depth])
                kept_count += 1
        for cid in classes:
            removed = self.exemplars[cid][len(kept[cid]):]
            if removed:
                self.removed_exemplars.setdefault(cid, []).extend(removed)
        self.exemplars = {cid: xs for cid, xs in kept.items() if xs}
        self._update_exemplar_count()
        self.reduction_history.append({
            'before_count': before,
            'after_count': self.exemplar_count,
            'removed_count': before - self.exemplar_count,
        })

    def _rebuild_cache(self) -> None:
        self.point_cloud_cache = {
            (x['scene_id'], x['object_idx']): x['points']
            for x in self.get_all_exemplars()
        }
        self.cache_size_bytes = sum(x.nbytes for x in self.point_cloud_cache.values())

    def get_statistics(self) -> Dict[str, Any]:
        counts = {str(k): len(v) for k, v in sorted(self.exemplars.items())}
        return {
            'memory_level': 'object',
            'total_exemplars': self.exemplar_count,
            'stored_classes': len(self.exemplars),
            'exemplars_per_class': counts,
            'average_exemplars_per_class': (
                self.exemplar_count / max(1, len(self.exemplars))),
            'max_total_exemplars': self.max_total_exemplars,
            'memory_utilization': (
                100.0 * self.exemplar_count / self.max_total_exemplars),
            'point_count': sum(x['point_count'] for x in self.get_all_exemplars()),
            'storage_mb': self.cache_size_bytes / (1024 * 1024),
            'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
            'cache_entries': len(self.point_cloud_cache),
            'cache_hit_rate': 100.0,
            'selection_strategy': self.selection_strategy_name,
            'edge_cases': {
                'empty_classes': [],
                'insufficient_classes': [
                    int(cid) for cid, count in counts.items()
                    if 0 < count < self.exemplars_per_class],
                'classes_at_limit': [
                    int(cid) for cid, count in counts.items()
                    if count == self.exemplars_per_class],
                'is_at_max_capacity': (
                    self.exemplar_count >= self.max_total_exemplars),
                'overflow_risk': (
                    self.exemplar_count > 0.9 * self.max_total_exemplars),
            },
        }

    def save_state(self, filepath: str, stage_id: Optional[int] = None) -> None:
        """Persist the complete crops and a human-readable sidecar manifest."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': 3,
            'memory_level': 'object',
            'stage_id': stage_id,
            'config': {
                'exemplars_per_class': self.exemplars_per_class,
                'selection_strategy': self.selection_strategy_name,
                'max_total_exemplars': self.max_total_exemplars,
                'min_points': self.min_points,
                'crop_margin': self.crop_margin,
                'floor_percentile': self.floor_percentile,
                'random_seed': self.random_seed,
                'learning_dynamics_update': self.learning_dynamics_update,
                'learning_dynamics_design2': self.learning_dynamics_design2,
            },
            'exemplars': self.exemplars,
        }
        with path.open('wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = dict(payload)
        meta['exemplars'] = {
            str(cid): [{k: v for k, v in x.items() if k != 'points'} for x in xs]
            for cid, xs in self.exemplars.items()
        }
        path.with_suffix('.json').write_text(json.dumps(_jsonable(meta), indent=2))

    @classmethod
    def load_state(cls, filepath: str) -> 'ObjectMemoryBank':
        with Path(filepath).open('rb') as f:
            payload = pickle.load(f)
        if payload.get('memory_level') != 'object':
            raise ValueError(f'Not an object-memory state: {filepath}')
        bank = cls(**payload['config'])
        bank.exemplars = {
            int(cid): xs for cid, xs in payload.get('exemplars', {}).items()}
        bank.loaded_stage_id = (
            int(payload['stage_id']) if payload.get('stage_id') is not None else None)
        bank._update_exemplar_count()
        bank._rebuild_cache()
        return bank

    def save_active_manifest(self, filepath: str,
                             stage_id: Optional[int] = None) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable({
            'memory_level': 'object',
            'stage_id': stage_id,
            'statistics': self.get_statistics(),
            'active_exemplars': {
                str(cid): [{k: v for k, v in x.items() if k != 'points'} for x in xs]
                for cid, xs in self.exemplars.items()
            },
        }), indent=2))

    def _save_debug_class(self, directory: Path, class_id: int,
                          exemplars: List[Dict[str, Any]],
                          stage_id: Optional[int]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for i, exemplar in enumerate(exemplars):
            np.save(directory / f'class_{class_id}_exemplar_{i}.npy', exemplar['points'])
        self.save_active_manifest(
            str(directory / f'object_memory_stage_{stage_id or 0}.json'), stage_id)

    def print_statistics(self) -> None:
        stats = self.get_statistics()
        print(
            'Object Memory Bank: '
            f"{stats['total_exemplars']}/{stats['max_total_exemplars']} objects, "
            f"{stats['stored_classes']} classes, {stats['point_count']} points, "
            f"{stats['storage_mb']:.1f} MB")


# Backward-compatible name used by the existing ScanNet wrapper.
MemoryBank = ObjectMemoryBank
