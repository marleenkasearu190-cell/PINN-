# 第四版：模块化 LakePINN

第四版把 `11维测试/九` 之后的主线代码整理为 `lake_pinn` Python 包，保留 PINN、PPO 动态调度、Kalman 同化、滚动预测和分层评分流程。它现在已归档为第五版前的模块化对照，不包含训练 checkpoint、预测 CSV、热图或完整实验输出。

## 目录结构

```text
归档/第四版/
|-- README.md
|-- 更新说明.md
`-- lake_pinn/
    |-- __main__.py
    |-- pipeline.py
    |-- standard_inputs.py
    |-- lake_profile_scorecard.py
    |-- train.py
    |-- predict.py
    |-- kalman.py
    |-- ppo.py
    `-- ...
```

## 主要入口

- `python -m lake_pinn`：训练或预测主入口，对应 `lake_pinn/pipeline.py`。
- `python -m lake_pinn.standard_inputs`：把 ERA5、LST 和剖面观测整理为标准宽表输入。
- `python lake_pinn/lake_profile_scorecard.py`：对预测剖面做 RMSE、MAE、bias 和物理过程评分。

运行时建议先进入归档第四版目录：

```powershell
cd <repo>\归档\第四版
python -m lake_pinn --help
```

## 训练示例

```powershell
python -m lake_pinn `
  --mode train `
  --era5 "path\to\ERA5_daily.csv" `
  --lst "path\to\LST_2017.csv" `
  --profile-obs "path\to\profile_observations.csv" `
  --profile-split-mode time_blocked `
  --model-input-dim 11 `
  --epochs 600 `
  --save-model-checkpoint "outputs\run4\mohonk_lake_2017_pinn_model_checkpoint.pt" `
  --output-dir "outputs\run4"
```

说明：第四版默认面向扩展 forcing，`--model-input-dim` 默认是 17。若要兼容 `11维测试/九` 和 run9 系列 checkpoint/retraining，请显式使用 `--model-input-dim 11`。

## 预测示例

```powershell
python -m lake_pinn `
  --mode predict `
  --era5 "path\to\ERA5_daily.csv" `
  --lst "path\to\LST_2017.csv" `
  --model-checkpoint-path "outputs\run4\mohonk_lake_2017_pinn_model_checkpoint.pt" `
  --model-input-dim 11 `
  --use-kalman `
  --output-dir "outputs\predict_run4"
```

如果预测阶段启用 Kalman，同化后的 `kalman_assimilated` 输出优先用于展示和评分；`pinn_rolling` 输出保留为 PINN 原始滚动预测诊断。

## 结果管理

第四版源码可以提交到 GitHub，但以下内容继续保留在本地或单独发布：

- `*.pt`、`*.pth`、`*.ckpt` 等模型 checkpoint。
- 预测 CSV、Kalman diagnostics、PPO history 和 scorecard CSV。
- 大批量热图、完整训练输出目录和临时缓存。

仓库只保留源码、说明文档和经过筛选的代表性图表。
