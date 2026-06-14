# 复现说明

本文档给出公开仓库层面的最小复现方式。由于原始数据、checkpoint 和完整实验输出没有提交，完整数值复现需要本地准备 manifest 指向的数据文件。

## 环境

建议使用 Python 3.10 或更新版本。

```powershell
pip install -r requirements.txt
```

验证当前主线入口：

```powershell
Push-Location ".\第九版"
python -m compileall -q lake_pinn tests scripts
python -m pytest tests -q
python -m lake_pinn --help
Pop-Location
```

## 训练示例

```powershell
Push-Location ".\第九版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --output-dir "..\outputs\v9_run" `
  --epochs 120 `
  --device cpu
Pop-Location
```

## 导出示例

```powershell
Push-Location ".\第九版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --checkpoint-path "..\outputs\v9_run\best_by_val_rolling.pt" `
  --output-dir "..\outputs\v9_export" `
  --export-only `
  --device cpu
Pop-Location
```

## 第九版诊断脚本

第九版 `scripts/` 提供 reconstruction 诊断工具，例如：

- `pipeline_controller.py`：实验 registry 和任务状态管理。
- `prepare_recon_tiered_smokes.py`：生成分层 smoke/diagnostic manifests。
- `diagnose_1d_loop_consistency.py`：检查 1d rolling-start 与 transition-pair 评估一致性。
- `diagnose_support_update_effect.py`：检查 support update 是否改善 query-start profile。
- `r19_observer_update_autopsy.py`、`r22_conservative_surface_generalization_autopsy.py`、`r23a_kd_source_separation_preflight.py`：定位 LSWT observer、Kd/source separation 和 lake-type 偏差来源。

## 验收检查

提交前应执行：

```powershell
python -m compileall -q .\第九版\lake_pinn .\第九版\tests .\第九版\scripts
Push-Location .\第九版; python -m pytest tests -q; python -m lake_pinn --help > $null; Pop-Location
git diff --check
rg -n "<local absolute path patterns>" .
```

并确认没有 checkpoint、CSV、完整实验目录、外部数据、日志或缓存文件进入提交。
