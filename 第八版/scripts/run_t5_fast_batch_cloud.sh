#!/usr/bin/env bash
set -euo pipefail

ROOT="${LAKEPINN_ROOT:-/root/LakePINN}"
MANIFEST="${MANIFEST:-$ROOT/experiments/manifests_20260522/T1_mendota_reconstruction_night_cloud.json}"
RUN_KIND="${RUN_KIND:-B0}"

cd "$ROOT"

python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

python -m compileall -q lake_pinn tests
python -m pyflakes lake_pinn tests
python -m pytest \
  tests/test_state_multilake.py::test_batched_transition_loss_matches_scalar_loop \
  tests/test_state_multilake.py::test_batched_segment_rollout_loss_matches_scalar_loop \
  -q

if [[ "$RUN_KIND" == "B0" ]]; then
  OUT="${OUT:-$ROOT/experiments/B0_T5fast_smoke_batchOn_kz15_heat005_5ep_20260525}"
  EPOCHS="${EPOCHS:-5}"
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
  EVAL_EVERY="${EVAL_EVERY:-5}"
elif [[ "$RUN_KIND" == "B2" ]]; then
  OUT="${OUT:-$ROOT/experiments/B2_T5fast_batchOn_kz15_heat005_200ep_20260525}"
  EPOCHS="${EPOCHS:-200}"
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
  EVAL_EVERY="${EVAL_EVERY:-20}"
else
  echo "RUN_KIND must be B0 or B2, got: $RUN_KIND" >&2
  exit 2
fi

mkdir -p "$OUT"

nohup python -u -m lake_pinn.state_multilake \
  --manifest "$MANIFEST" \
  --output-dir "$OUT" \
  --epochs "$EPOCHS" \
  --task-mode analysis \
  --data-fill-mode reconstruction \
  --test-lake-id mendota_2020 \
  --residual-limit-c 0.25 \
  --wind-kz-scale 1.5 \
  --autumn-convective-boost 2.0 \
  --transition-loss-weight 0.5 \
  --segment-rollout-loss-weight 0.15 \
  --segment-rollout-max-days 30 \
  --long-free-roll-start-epoch 0 \
  --long-free-roll-ramp-epochs 60 \
  --long-free-roll-samples-per-lake 4 \
  --rolling-horizon-eval-max-starts 10 \
  --heat-content-transition-weight 0.05 \
  --heat-content-transition-season-mode auto \
  --heat-content-transition-depth-factor on \
  --heat-content-transition-effective-max 0.10 \
  --heat-content-full-column-min-coverage 0.75 \
  --turbulent-flux-mode bulk \
  --transition-batch-mode on \
  --segment-rollout-batch-mode on \
  --transition-batch-size 0 \
  --segment-rollout-batch-size 0 \
  --checkpoint-every-epochs "$CHECKPOINT_EVERY" \
  --eval-every-epochs "$EVAL_EVERY" \
  --profile-runtime \
  --device cuda \
  > "$OUT/run.log" 2>&1 &

echo $! > "$OUT/pid.txt"
echo "Started LakePINN $RUN_KIND fast batch run"
echo "pid: $(cat "$OUT/pid.txt")"
echo "log: $OUT/run.log"
echo "watch: tail -f '$OUT/run.log'"
