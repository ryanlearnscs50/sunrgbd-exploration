import numpy as np

from mmdet3d.datasets.pipelines.exemplar_insertion import InsertExemplarObjects


def test_local_object_is_placed_on_floor_with_bottom_center_box():
    transform = InsertExemplarObjects(
        collision_threshold=0.0, placement_jitter=0.0,
        max_placement_attempts=1, floor_offset=0.0, floor_percentile=0.0)
    local_points = np.array([
        [0.0, 0.0, -0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
    ], dtype=np.float32)
    source_box = np.array([10, 20, 4, 1, 1, 1, 0], dtype=np.float32)
    scene_points = np.array([
        [-2, -2, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0],
    ], dtype=np.float32)

    placed_points, placed_box = transform._find_valid_placement(
        local_points, source_box, np.empty((0, 7), dtype=np.float32), scene_points)

    assert placed_points is not None
    assert placed_box[2] == 0.0  # DepthInstance3DBoxes bottom-centre convention
    np.testing.assert_allclose(placed_points[:, 2], [0.0, 1.0], atol=1e-6)
    # The source centre must not be subtracted a second time.
    assert np.max(np.abs(placed_points[:, :2])) <= 2.0


def test_collision_rejects_any_overlap_when_threshold_is_zero():
    transform = InsertExemplarObjects(collision_threshold=0.0)
    existing = np.array([[0, 0, 0.5, 1, 1, 1, 0]], dtype=np.float32)
    overlapping = np.array([0.25, 0, 0.5, 1, 1, 1, 0], dtype=np.float32)
    separate = np.array([2, 0, 0.5, 1, 1, 1, 0], dtype=np.float32)

    assert transform._check_collision(overlapping, existing)
    assert not transform._check_collision(separate, existing)


def test_collision_envelope_accounts_for_yaw():
    transform = InsertExemplarObjects(collision_threshold=0.0)
    # Rotating this 4x1 box by 90 degrees makes its enclosing Y extent 4.
    existing = np.array([
        [0, 0, 0.5, 4, 1, 1, np.pi / 2]], dtype=np.float32)
    overlaps_rotated_envelope = np.array(
        [0, 1.5, 0.5, 1, 1, 1, 0], dtype=np.float32)

    assert transform._check_collision(overlaps_rotated_envelope, existing)


def test_source_floor_offset_preserves_elevated_object_height():
    transform = InsertExemplarObjects(
        collision_threshold=0.0, placement_jitter=0.0,
        max_placement_attempts=1, floor_percentile=0.0,
        preserve_source_height=True)
    local_points = np.array([
        [0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], dtype=np.float32)
    source_box = np.array([0, 0, 4, 1, 1, 1, 0], dtype=np.float32)
    scene_points = np.array([
        [-2, -2, -1], [2, 2, 2]], dtype=np.float32)

    placed_points, placed_box = transform._find_valid_placement(
        local_points, source_box, np.empty((0, 7), dtype=np.float32),
        scene_points, source_floor_offset=1.5)

    assert placed_box[2] == 0.5  # target floor -1 + saved clearance 1.5
    np.testing.assert_allclose(placed_points[:, 2], [0.5, 1.5], atol=1e-6)


def test_review_weights_drive_object_sampling():
    class _Bank:
        previous_classes = [0]

        @staticmethod
        def get_exemplars(_):
            return [
                {'scene_id': 'never', 'replay_weight': 0.0},
                {'scene_id': 'chosen', 'replay_weight': 2.0},
            ]

    transform = InsertExemplarObjects(
        memory_bank=_Bank(), max_exemplars_per_scene=1,
        insertion_probability=1.0)

    for _ in range(20):
        selected = transform._sample_exemplars_for_insertion([0])
        assert selected[0]['scene_id'] == 'chosen'
