import pandas as pd
import torch

from lake_pinn import state_multilake as sm
from lake_pinn.state_reconstruction import _profile_lookup


def _toy_target_lakes():
    df = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=4, freq='D')})
    depths = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float32)
    date_to_index = sm._date_index_map(df)
    train_lookup = {
        pd.Timestamp('2020-01-02'): [1.0, 2.0, 3.0],
    }
    val_lookup = {
        pd.Timestamp('2020-01-03'): [4.0, 5.0, 6.0],
    }
    all_lookup = dict(train_lookup)
    all_lookup.update(val_lookup)
    train_masks = {
        pd.Timestamp('2020-01-02'): [True, False, True],
    }
    val_masks = {
        pd.Timestamp('2020-01-03'): [False, True, True],
    }
    all_masks = dict(train_masks)
    all_masks.update(val_masks)
    base_lake = {
        'df': df,
        'depths': depths,
        'date_to_index': date_to_index,
        'lookups': {'train': train_lookup, 'val': val_lookup, 'all': all_lookup},
        'lookup_tensors': {
            'train': sm._tensorize_profile_lookup(train_lookup, device='cpu'),
            'val': sm._tensorize_profile_lookup(val_lookup, device='cpu'),
            'all': sm._tensorize_profile_lookup(all_lookup, device='cpu'),
        },
        'lookup_masks': {'train': train_masks, 'val': val_masks, 'all': all_masks},
        'lookup_mask_tensors': {
            'train': sm._tensorize_mask_lookup(train_masks, device='cpu'),
            'val': sm._tensorize_mask_lookup(val_masks, device='cpu'),
            'all': sm._tensorize_mask_lookup(all_masks, device='cpu'),
        },
    }
    matrix_lake = dict(base_lake)
    matrix_lake['target_tensors_by_day'] = {
        'train': sm._build_target_tensor_matrix(
            df,
            train_lookup,
            train_masks,
            depths.numpy(),
            device='cpu',
            date_to_index=date_to_index,
        ),
        'val': sm._build_target_tensor_matrix(
            df,
            val_lookup,
            val_masks,
            depths.numpy(),
            device='cpu',
            date_to_index=date_to_index,
        ),
        'all': sm._build_target_tensor_matrix(
            df,
            all_lookup,
            all_masks,
            depths.numpy(),
            device='cpu',
            date_to_index=date_to_index,
        ),
    }
    return base_lake, matrix_lake


def test_target_matrix_single_lookup_matches_lookup_tensor_path():
    base_lake, matrix_lake = _toy_target_lakes()

    for date in (pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')):
        old_profile, old_mask = sm._target_tensor_and_mask(base_lake, 'train', date)
        new_profile, new_mask = sm._target_tensor_and_mask(matrix_lake, 'train', date)

        assert torch.equal(new_profile, old_profile)
        assert torch.equal(new_mask, old_mask)


def test_target_matrix_batch_lookup_matches_lookup_tensor_path_with_fallback():
    base_lake, matrix_lake = _toy_target_lakes()
    dates = iter([pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')])

    old_profiles, old_masks = sm._target_tensor_and_mask_batch(
        base_lake,
        'train',
        [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')],
    )
    new_profiles, new_masks = sm._target_tensor_and_mask_batch(matrix_lake, 'train', dates)

    assert torch.equal(new_profiles, old_profiles)
    assert torch.equal(new_masks, old_masks)


def test_segment_rollout_target_plan_prefetches_fixed_time_axis():
    _base_lake, matrix_lake = _toy_target_lakes()
    active_targets = [
        {
            1: pd.Timestamp('2020-01-02'),
            2: pd.Timestamp('2020-01-03'),
        },
        {
            1: pd.Timestamp('2020-01-02'),
        },
    ]

    plan = sm._segment_rollout_target_plan_for_lake(
        matrix_lake,
        'train',
        active_targets,
        [0, 0],
        2,
    )

    assert len(plan) == 2
    assert plan[0]['active_indices'] == [0, 1]
    assert plan[0]['target_gap_days'] == [1, 1]
    assert torch.equal(
        plan[0]['target'],
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32),
    )
    assert torch.equal(
        plan[0]['target_mask'],
        torch.tensor([[True, False, True], [True, False, True]]),
    )
    assert plan[1]['active_indices'] == [0]
    assert plan[1]['target_gap_days'] == [2]
    assert torch.equal(
        plan[1]['target'],
        torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32),
    )
    assert torch.equal(
        plan[1]['target_mask'],
        torch.tensor([[False, True, True]]),
    )


def test_packed_lake_tensor_store_serves_targets_and_forcing_batches():
    _base_lake, matrix_lake = _toy_target_lakes()
    forcing_tensors = {
        'air_temperature': torch.arange(4.0, dtype=torch.float32),
        'history_features': torch.arange(8.0, dtype=torch.float32).reshape(4, 2),
    }
    matrix_lake['forcing_tensors'] = forcing_tensors
    matrix_lake['forcing_rows'] = sm._forcing_rows_from_matrix(forcing_tensors, 4)
    matrix_lake['area'] = torch.ones_like(matrix_lake['depths'])
    matrix_lake['static_features'] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    matrix_lake['static_features_2d'] = matrix_lake['static_features'].reshape(1, -1).contiguous()
    matrix_lake['heat_content_layer_weights'] = sm._heat_content_layer_weights(
        matrix_lake['depths'],
        matrix_lake['area'],
        device=matrix_lake['depths'].device,
        dtype=matrix_lake['depths'].dtype,
    )
    matrix_lake['packed_tensor_store'] = sm.PackedLakeTensorStore(
        forcing_tensors=forcing_tensors,
        target_tensors_by_day=matrix_lake['target_tensors_by_day'],
        depths=matrix_lake['depths'],
        area=matrix_lake['area'],
        heat_content_layer_weights=matrix_lake['heat_content_layer_weights'],
        static_features=matrix_lake['static_features'],
        static_features_2d=matrix_lake['static_features_2d'],
        date_to_index=matrix_lake['date_to_index'],
    )

    profile, mask = sm._target_tensor_and_mask(
        matrix_lake,
        'train',
        pd.Timestamp('2020-01-02'),
    )
    forcing_batch = sm._forcing_row_batch(matrix_lake, torch.tensor([1, 3]))
    static_batch = sm._stack_static_features_for_items([(0, matrix_lake), (1, matrix_lake)])

    assert torch.equal(profile, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
    assert torch.equal(mask, torch.tensor([[True, False, True]]))
    assert torch.equal(forcing_batch['air_temperature'], torch.tensor([1.0, 3.0]))
    assert torch.equal(
        forcing_batch['history_features'],
        torch.tensor([[2.0, 3.0], [6.0, 7.0]], dtype=torch.float32),
    )
    assert torch.equal(static_batch, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))


def test_profile_lookup_observed_point_strict_masks_only_observed_depth_cells():
    profile_obs = pd.DataFrame({
        'Date': [pd.Timestamp('2020-07-01'), pd.Timestamp('2020-07-01')],
        'Depth_m': [0.0, 5.0],
        'Temperature_C': [20.0, 8.0],
    })
    depths = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).numpy()

    grid_lookup, grid_masks = _profile_lookup(
        profile_obs,
        depths,
        return_masks=True,
        target_mode='grid_masked',
    )
    strict_lookup, strict_masks = _profile_lookup(
        profile_obs,
        depths,
        return_masks=True,
        target_mode='observed_point_strict',
    )

    date = pd.Timestamp('2020-07-01')
    assert grid_masks[date].tolist() == [True, True, True, True, True, True]
    assert strict_masks[date].tolist() == [True, False, False, False, False, True]
    assert strict_lookup[date][0] == 20.0
    assert strict_lookup[date][-1] == 8.0
    assert pd.isna(strict_lookup[date][1])


def test_season_balanced_profile_sampling_covers_available_seasons():
    dates = [
        pd.Timestamp('2020-01-10'),
        pd.Timestamp('2020-03-10'),
        pd.Timestamp('2020-07-10'),
        pd.Timestamp('2020-07-20'),
        pd.Timestamp('2020-10-10'),
    ]
    df = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=366, freq='D')})
    lake = {
        'lookups': {'train': {date: [1.0, 2.0] for date in dates}},
        'date_to_index': sm._date_index_map(df),
    }

    selected = sm._select_zero_profile_init_net_dates(
        lake,
        'train',
        samples_per_lake=4,
        epoch=0,
        sampling_mode='season_balanced',
    )
    seasons = {sm._season_name(date.month) for date in selected}

    assert len(selected) == 4
    assert seasons == {'winter', 'spring', 'summer', 'fall'}
