# LakePINN Cloud GPU Runbook

This bundle is prepared for running the Mendota T5 experiment on a Linux GPU
instance. The expected cloud root is:

```bash
/root/LakePINN
```

## 1. Upload

From Windows, create the bundle:

```powershell
Set-Location ".\归档\第七版"
powershell -ExecutionPolicy Bypass -File scripts\package_mendota_cloud.ps1
```

Upload `lakepinn_mendota_cloud_bundle.zip` to the cloud GPU instance, then:

```bash
cd /root
unzip lakepinn_mendota_cloud_bundle.zip
cd /root/LakePINN
```

## 2. Environment

Use a PyTorch CUDA image when creating the GPU instance. Then run:

```bash
cd /root/LakePINN
pip install -r requirements.txt

python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Optional checks:

```bash
python -m compileall -q lake_pinn tests
python -m pytest tests/test_state_multilake.py tests/test_state_forecaster_solver.py -q
```

## 3. Run T5

```bash
cd /root/LakePINN
bash scripts/run_t5_cloud.sh
tail -f /root/LakePINN/experiments/T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud/run.log
```

Check GPU use:

```bash
nvidia-smi
```

The script starts training with `nohup`, writes the process id to `pid.txt`,
and keeps running after the SSH session closes as long as the GPU instance stays
alive.

## 4. Download Results

After training finishes, download:

```text
global_state_forecaster_training_history.csv
global_state_forecaster_checkpoint.pt
mendota_2020_heldout_state_reconstruction_year_heatmap.png
mendota_2020_heldout_state_reconstruction_bias_contour_heatmap.png
mendota_2020_heldout_state_reconstruction_heat_closure_annual_summary.csv
mendota_2020_heldout_state_reconstruction_heat_closure_monthly_summary.csv
mendota_2020_heldout_state_reconstruction_density_stability_summary.csv
mendota_2020_heldout_state_reconstruction_scorecard_report.png
```

Example:

```bash
scp -r root@SERVER_IP:/root/LakePINN/experiments/T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud .
```
