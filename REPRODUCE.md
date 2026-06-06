# 复现说明

本文档给出公开仓库层面的最小复现方式。由于原始数据、checkpoint 和完整实验输出没有提交，完整数值复现需要本地准备 manifest 指向的数据文件。

## 环境

建议使用 Python 3.10 或更新版本。

```powershell
pip install -r requirements.txt
```

验证当前主线入口：

```powershell
Push-Location ".\第八版"
python -m compileall -q lake_pinn tests scripts benchmarks
python -m pytest tests -q
python -m lake_pinn --help
Pop-Location
```

当前第八版本地测试记录为 `149 passed`。

## 训练示例

```powershell
Push-Location ".\第八版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --output-dir "..\outputs\v8_run" `
  --epochs 120 `
  --device cpu
Pop-Location
```

GPU 训练可参考 [`第八版/CLOUD_GPU_README.md`](./第八版/CLOUD_GPU_README.md)。

## 导出示例

```powershell
Push-Location ".\第八版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --checkpoint-path "..\outputs\v8_run\global_state_forecaster_checkpoint.pt" `
  --output-dir "..\outputs\v8_export" `
  --export-only `
  --device cpu
Pop-Location
```

## PGDL-WRR benchmark

第八版包含 `benchmarks/pgdl_wrr_compare.py`，用于构建 Mendota PGDL-WRR 2019 对照。该脚本会下载外部数据到未提交的 `external/`，并把结果写到未提交的 `experiments/`。

```powershell
Push-Location ".\第八版"
python benchmarks\pgdl_wrr_compare.py --skip-download
Pop-Location
```

首次运行如果需要下载官方数据，可去掉 `--skip-download`。

## 验收检查

提交前应执行：

```powershell
python -m compileall -q .\第八版\lake_pinn .\第八版\tests .\第八版\scripts .\第八版\benchmarks
Push-Location .\第八版; python -m pytest tests -q; python -m lake_pinn --help > $null; Pop-Location
git diff --check
rg -n "<local absolute path patterns>" .
```

并确认没有 checkpoint、CSV、完整实验目录、外部数据、日志或缓存文件进入提交。
