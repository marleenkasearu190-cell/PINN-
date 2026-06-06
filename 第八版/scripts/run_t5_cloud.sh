#!/usr/bin/env bash
set -euo pipefail

ROOT="${LAKEPINN_ROOT:-/root/LakePINN}"
MANIFEST="${MANIFEST:-$ROOT/experiments/manifests_20260522/T1_mendota_reconstruction_night_cloud.json}"
OUT="${OUT:-$ROOT/experiments/T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud}"
EPOCHS="${EPOCHS:-200}"

mkdir -p "$OUT"
cd "$ROOT"

python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

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
  --device cuda \
  > "$OUT/run.log" 2>&1 &

echo $! > "$OUT/pid.txt"
echo "Started LakePINN T5 cloud run"
echo "pid: $(cat "$OUT/pid.txt")"
echo "log: $OUT/run.log"
echo "watch: tail -f '$OUT/run.log'"
