# 复现说明

本文档给出公开仓库层面的最小复现方式。完整数值复现需要本地准备 manifest 指向的标准输入数据，并另行取得对应 checkpoint 或重新训练。

## 环境

建议使用 Python 3.10 或更新版本。

```powershell
pip install -r requirements.txt
```

验证当前主线入口：

```powershell
Push-Location ".\第十版"
python -m compileall -q lake_pinn tests scripts
python -m pytest tests -q
python -m lake_pinn --help
Pop-Location
```

## 训练示例

```powershell
Push-Location ".\第十版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --output-dir "..\outputs\v10_run" `
  --epochs 20 `
  --device cpu
Pop-Location
```

## 导出示例

```powershell
Push-Location ".\第十版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --checkpoint-path "..\outputs\v10_run\global_state_forecaster_checkpoint.pt" `
  --output-dir "..\outputs\v10_export" `
  --export-only `
  --device cpu
Pop-Location
```

## 第十版诊断脚本

第十版 `scripts/` 提供两个公开维护脚本：

- `export_source_package.py`：按 allow-list 导出干净源码包。
- `initial_column_eval.py`：按湖泊和深度带评估 zero-profile initial-column 误差。

## 验收检查

提交前应执行：

```powershell
python -m compileall -q .\第十版\lake_pinn .\第十版\tests .\第十版\scripts
Push-Location .\第十版; python -m pytest tests -q; python -m lake_pinn --help > $null; Pop-Location
git diff --check
rg -n "<local absolute path patterns>" README.md EXPERIMENTS.md MODEL.md DATA.md REPRODUCE.md .\第十版
```

并确认 checkpoint、CSV、完整实验目录、外部数据、日志和缓存文件没有进入提交。
