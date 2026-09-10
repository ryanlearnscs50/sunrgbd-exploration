import numpy as np

from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank


def test_oriented_crop_is_stored_in_local_coordinates():
    # A 90-degree box: its long local X axis lies along world Y.
    box = np.array([10.0, 20.0, 2.0, 4.0, 2.0, 2.0, np.pi / 2], dtype=np.float32)
    local = np.array([
        [1.5, 0.0, 0.0],
        [-1.5, 0.5, 0.5],
        [3.0, 0.0, 0.0],  # outside local X extent
    ], dtype=np.float32)
    c, s = np.cos(box[6]), np.sin(box[6])
    world_xyz = local @ np.array(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points = np.concatenate(
        [world_xyz + box[:3], np.ones((3, 3), dtype=np.float32)], axis=1)

    cropped = ObjectMemoryBank.extract_object_points(points, box)

    assert cropped.shape == (2, 6)
    np.testing.assert_allclose(cropped[:, :3], local[:2], atol=1e-5)


def test_bank_owns_points_and_round_trips_state(tmp_path):
    bank = ObjectMemoryBank(
        exemplars_per_class=2, max_total_exemplars=4, min_points=1,
        selection_strategy='largest_point_count')
    source = np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    objects = [
        dict(scene_id='a', object_idx=0, class_id=2,
             bbox=np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float32),
             points=source),
        dict(scene_id='b', object_idx=1, class_id=2,
             bbox=np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float32),
             points=np.repeat(source, 3, axis=0)),
        dict(scene_id='c', object_idx=2, class_id=2,
             bbox=np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float32),
             points=np.repeat(source, 2, axis=0)),
    ]

    assert bank.add_exemplars(2, objects, stage_id=1) == 2
    source[:] = 99  # the bank must not alias caller-owned point storage
    assert sorted(x['point_count'] for x in bank.get_exemplars([2])) == [2, 3]
    assert np.max(bank.get_exemplar_points(bank.get_exemplars([2])[0])[:, :3]) < 10

    state = tmp_path / 'bank.pkl'
    bank.save_state(str(state), stage_id=1)
    restored = ObjectMemoryBank.load_state(str(state))

    assert restored.get_statistics()['memory_level'] == 'object'
    assert restored.loaded_stage_id == 1
    assert restored.get_total_exemplar_count() == 2
    np.testing.assert_array_equal(
        restored.get_exemplar_points(restored.get_exemplars([2])[0]),
        bank.get_exemplar_points(bank.get_exemplars([2])[0]))
    assert state.with_suffix('.json').exists()


def test_global_budget_is_balanced_across_classes():
    bank = ObjectMemoryBank(
        exemplars_per_class=3, max_total_exemplars=4, min_points=1)
    point = np.zeros((1, 3), dtype=np.float32)
    box = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.float32)
    for class_id in (0, 1):
        bank.add_exemplars(class_id, [
            dict(scene_id=f'{class_id}-{i}', object_idx=i, bbox=box, points=point)
            for i in range(3)
        ])

    assert bank.get_total_exemplar_count() == 4
    assert bank.get_class_exemplar_count(0) == 2
    assert bank.get_class_exemplar_count(1) == 2


def test_scene_population_records_floor_relative_box_height():
    bank = ObjectMemoryBank(
        exemplars_per_class=1, max_total_exemplars=1, min_points=1,
        floor_percentile=0.0)
    scene = np.array([
        [-1, -1, -2], [0, 0, 1], [1, 1, 2]], dtype=np.float32)
    box = np.array([0, 0, 1, 2, 2, 2, 0], dtype=np.float32)

    bank.add_exemplars(
        0, [dict(scene_id='s', object_idx=0, bbox=box)],
        scene_points_dict={'s': scene})

    exemplar = bank.get_exemplars([0])[0]
    # Box bottom is 0 and robust source floor is -2.
    assert exemplar['source_floor_offset'] == 2.0


def test_design2_object_selection_prefers_learning_and_nonredundancy():
    bank = ObjectMemoryBank(
        exemplars_per_class=2, max_total_exemplars=2, min_points=1,
        random_seed=7, selection_strategy='learning_dynamics_design2',
        learning_dynamics_design2=dict(
            redundancy_lambda=0.5, redundancy_topk=1))
    point = np.zeros((1, 3), dtype=np.float32)
    box = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.float32)
    objects = [
        dict(scene_id=sid, object_idx=i, bbox=box, points=point)
        for i, sid in enumerate(('a', 'b', 'c', 'd'))
    ]

    # a and b have similar gain-dominated dynamics; c is slightly weaker but
    # drop-dominated, so Design-2 should take a then c rather than redundant b.
    terms = {
        'a': {'1': {'0': dict(g=1.0, r_best=1.0, d=0.0, u=1.0)}},
        'b': {'1': {'0': dict(g=0.9, r_best=1.0, d=0.0, u=0.9)}},
        'c': {'1': {'0': dict(g=0.0, r_best=0.8, d=0.8, u=0.8)}},
        'd': {'1': {'0': dict(g=0.1, r_best=0.5, d=0.0, u=0.1)}},
    }
    payload = dict(class_need={'0': 1.0}, seat_class_terms=terms)

    assert bank.add_exemplars(
        0, objects, stage_id=1,
        learning_dynamics_design2_payload=payload) == 2
    selected = bank.get_exemplars([0])
    assert [x['scene_id'] for x in selected] == ['a', 'c']
    assert selected[0]['learning_dynamics_design2']['rank'] == 0
    assert selected[1]['learning_dynamics_design2']['redundancy'] == 0.0


def test_design2_object_selection_requires_complete_source_terms():
    bank = ObjectMemoryBank(
        exemplars_per_class=1, max_total_exemplars=1, min_points=1,
        selection_strategy='learning_dynamics_design2')
    point = np.zeros((1, 3), dtype=np.float32)
    box = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.float32)

    try:
        bank.add_exemplars(
            0, [dict(scene_id='missing', object_idx=0, bbox=box, points=point)],
            stage_id=1,
            learning_dynamics_design2_payload=dict(
                class_need={'0': 1.0},
                seat_class_terms={'other': {'1': {'0': {'u': 1.0}}}}))
    except ValueError as exc:
        assert 'missing source scene/class terms' in str(exc)
    else:
        raise AssertionError('Expected missing Design-2 object terms to fail.')


def test_design2_uses_distinct_source_scenes_before_duplicates():
    bank = ObjectMemoryBank(
        exemplars_per_class=2, max_total_exemplars=2, min_points=1,
        random_seed=3, selection_strategy='learning_dynamics_design2')
    point = np.zeros((1, 3), dtype=np.float32)
    box = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.float32)
    objects = [
        dict(scene_id='strong', object_idx=0, bbox=box, points=point),
        dict(scene_id='strong', object_idx=1, bbox=box, points=point),
        dict(scene_id='weak', object_idx=0, bbox=box, points=point),
    ]
    payload = dict(
        class_need={'0': 1.0},
        seat_class_terms={
            'strong': {'1': {'0': dict(g=1.0, r_best=1.0, d=0.0, u=1.0)}},
            'weak': {'1': {'0': dict(g=0.1, r_best=1.0, d=0.0, u=0.1)}},
        })

    bank.add_exemplars(
        0, objects, stage_id=1,
        learning_dynamics_design2_payload=payload)

    assert {x['scene_id'] for x in bank.get_exemplars([0])} == {
        'strong', 'weak'}


def test_source_scene_review_weights_transfer_to_object_seats():
    bank = ObjectMemoryBank(
        exemplars_per_class=2, max_total_exemplars=2, min_points=1)
    point = np.zeros((1, 3), dtype=np.float32)
    box = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.float32)
    bank.add_exemplars(0, [
        dict(scene_id='a', object_idx=0, bbox=box, points=point),
        dict(scene_id='b', object_idx=0, bbox=box, points=point),
    ], stage_id=2)

    report = bank.apply_source_seat_replay_weights(
        {'a_stage2': 4.0, 'b_stage2': 1.0})

    assert report == {'applied': 2, 'missing': 0}
    weights = {x['scene_id']: x['replay_weight']
               for x in bank.get_exemplars([0])}
    assert weights == {'a': 4.0, 'b': 1.0}


def test_source_scene_entries_are_deduplicated_and_stage_bounded():
    bank = ObjectMemoryBank(
        exemplars_per_class=3, max_total_exemplars=3, min_points=1)
    point = np.zeros((1, 3), dtype=np.float32)
    box = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.float32)
    bank.add_exemplars(0, [
        dict(scene_id='a', object_idx=0, bbox=box, points=point),
        dict(scene_id='a', object_idx=1, bbox=box, points=point),
    ], stage_id=1)
    bank.add_exemplars(1, [
        dict(scene_id='b', object_idx=0, bbox=box, points=point),
    ], stage_id=2)

    class Dataset:
        _object_scene_info_by_id = {
            'a': {'scene_id': 'a', 'payload': [1]},
            'b': {'scene_id': 'b', 'payload': [2]},
        }

    bank.dataset_ref = Dataset()
    entries = bank.list_source_scene_entries(max_save_stage=1)

    assert [(row['scene_id'], row['save_stage']) for row in entries] == [('a', 1)]
    entries[0]['snapshot']['data_info']['payload'].append(99)
    assert Dataset._object_scene_info_by_id['a']['payload'] == [1]
