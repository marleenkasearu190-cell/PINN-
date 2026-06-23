from lake_pinn import state_multilake as sm


def test_gpu_batch_autotune_enables_segment_rollout_batch_when_unset():
    resolved = sm._resolve_gpu_batch_autotune(
        gpu_batch_autotune='on',
        gpu_batch_autotune_target_batch_size=128,
        transition_batch_size=0,
        segment_rollout_batch_size=0,
        rolling_horizon_batch_size=0,
        unlabeled_heat_closure_batch_size=0,
        cross_lake_batch_mode='off',
        cross_lake_batch_size=0,
    )

    assert resolved['gpu_batch_autotune_applied'] is True
    assert resolved['cross_lake_batch_mode'] == 'on'
    assert resolved['cross_lake_batch_size'] == 128
    assert resolved['segment_rollout_batch_size'] == 64


def test_gpu_batch_autotune_keeps_segment_rollout_batch_off_when_autotune_off():
    resolved = sm._resolve_gpu_batch_autotune(
        gpu_batch_autotune='off',
        gpu_batch_autotune_target_batch_size=128,
        transition_batch_size=0,
        segment_rollout_batch_size=0,
        rolling_horizon_batch_size=0,
        unlabeled_heat_closure_batch_size=0,
        cross_lake_batch_mode='off',
        cross_lake_batch_size=0,
    )

    assert resolved['gpu_batch_autotune_applied'] is False
    assert resolved['cross_lake_batch_mode'] == 'off'
    assert resolved['segment_rollout_batch_size'] == 0
