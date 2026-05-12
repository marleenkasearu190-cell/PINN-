# 第五版：Raw PINN 主线

第五版是在第四版模块化代码基础上的下一阶段主线。它继续使用 `lake_pinn` 包结构，但把预测侧重点调整为 raw PINN：默认不再依赖 rolling 后处理、Kalman 同化或 PPO 预测调度，相关能力保留为可选诊断和对照。

本目录只保存源码和说明，不包含 checkpoint、预测 CSV 或完整实验输出；代表热图和 scorecard 摘要整理在根目录 `docs/figures/`。

## 目录结构

```text
第五版/
|-- README.md
|-- 更新说明.md
`-- lake_pinn/
    |-- __main__.py
    |-- pipeline.py
    |-- train.py
    |-- predict.py
    |-- losses.py
    |-- lake_profile_scorecard.py
    `-- ...
```

## 主要变化

- 默认输入维度提升为 27 维，用于扩展 forcing、过去天气记忆和因果 previous-state memory。
- raw PINN 输出成为主线；rolling prediction、Kalman 和 PPO 训练/预测调度均可显式开启或关闭。
- 新增 `seasonal_blocked` 剖面切分，以及 `--profile-train-date-fraction` 低数据实验抽样。
- 新增 whole-profile physics 训练约束，包括 profile-grid physics、density regularization 和 bottom slow-change。
- 评分工具升级为 scorecard v2，支持季节覆盖、失败项统计、图像评分和候选输出排序。

## 当前代表实验

本地代表实验为 `T34_rawPINN_noRolling_noKalman_noPPO_20260511`，使用 raw PINN 默认预测输出与 Mohonk 2017 的 0-13 m 观测剖面对齐评分：

| RMSE | MAE | bias | scorecard v2 | 当前判断 |
|---:|---:|---:|---:|---|
| 1.250 | 0.830 | -0.069 | 80.71 | 数值精度已明显好于第三版和第四版对照，但 density stability 仍未通过 |

## 运行入口

```powershell
cd E:\pycharm\PINN\github仓库\PINN-\第五版
python -m lake_pinn --help
```

## 训练示例

```powershell
python -m lake_pinn `
  --mode train `
  --era5 "path\to\ERA5_daily.csv" `
  --lst "path\to\LST_2017.csv" `
  --profile-obs "path\to\profile_observations.csv" `
  --profile-split-mode seasonal_blocked `
  --epochs 600 `
  --save-model-checkpoint "outputs\run5\mohonk_lake_2017_pinn_model_checkpoint.pt" `
  --output-dir "outputs\run5"
```

## 预测示例

```powershell
python -m lake_pinn `
  --mode predict `
  --era5 "path\to\ERA5_daily.csv" `
  --lst "path\to\LST_2017.csv" `
  --model-checkpoint-path "outputs\run5\mohonk_lake_2017_pinn_model_checkpoint.pt" `
  --output-dir "outputs\predict_run5"
```

默认预测输出优先代表 raw PINN。需要对照时可以显式启用 `--use-kalman` 或 legacy rolling 诊断参数。

## 结果管理

以下内容不进入 GitHub：

- `*.pt`、`*.pth`、`*.ckpt` 模型 checkpoint。
- 预测 CSV、scorecard CSV 和 diagnostics CSV。
- 批量热图 PNG 和完整实验输出目录。
- `__pycache__/` 与 `.pyc` 缓存文件。

本地 `T34_rawPINN_noRolling_noKalman_noPPO_20260511/` 属于第五版实验输出，只在文档中记录结论，不整目录提交。
