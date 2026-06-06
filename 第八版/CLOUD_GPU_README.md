# LakePINN v8 Cloud GPU Runbook

本文档用于第八版云端 GPU 训练和导出。默认云端根目录示例为：

```bash
/root/LakePINN_v8
```

## 1. 打包和上传

在 Windows 仓库根目录执行：

```powershell
Set-Location ".\第八版"
powershell -ExecutionPolicy Bypass -File scripts\package_mendota_cloud.ps1
```

如果使用第八版跨湖泛化实验，请优先检查打包脚本中的 manifest 和数据路径，避免把本地绝对路径写入云端配置。上传压缩包后：

```bash
cd /root
unzip lakepinn_mendota_cloud_bundle.zip -d LakePINN_v8
cd /root/LakePINN_v8
```

## 2. 环境

建议使用包含 CUDA 版 PyTorch 的镜像。

```bash
pip install -r requirements.txt

python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

基础检查：

```bash
python -m compileall -q lake_pinn tests scripts benchmarks
python -m pytest tests -q
python -m lake_pinn --help
```

## 3. 运行

第八版主入口：

```bash
python -m lake_pinn \
  --manifest path/to/manifest.json \
  --output-dir experiments/v8_run \
  --epochs 120 \
  --device cuda
```

如需后台运行：

```bash
nohup python -m lake_pinn \
  --manifest path/to/manifest.json \
  --output-dir experiments/v8_run \
  --epochs 120 \
  --device cuda > experiments/v8_run/run.log 2>&1 &
echo $! > experiments/v8_run/pid.txt
```

监控：

```bash
nvidia-smi
tail -f experiments/v8_run/run.log
```

## 4. 下载结果

优先下载：

```text
global_state_forecaster_training_history.csv
global_state_forecaster_checkpoint.pt
*_heldout_state_reconstruction_year_heatmap.png
*_heldout_state_reconstruction_bias_contour_heatmap.png
*_heldout_state_reconstruction_scorecard_report.png
*_heldout_state_reconstruction_temperature_depth_predictions.csv
```

公开提交时只挑选代表 PNG 放入仓库根目录 `docs/figures/`，不要提交 checkpoint、CSV、完整 `experiments/` 或日志。
