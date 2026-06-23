# LakePINN：物理约束湖泊温度剖面预测

LakePINN 是一个面向湖泊水温剖面预测的研究型实验仓库。项目以 ERA5 气象强迫、MODIS/LST 表面温度、湖泊 metadata 和剖面观测为输入，预测逐日温度深度剖面 `T(z,t)`，并围绕物理一致性、长时段滚动稳定性、跨湖泛化和观测稀疏条件下的 reconstruction 展开实验。

当前推荐继续开发的主线是 [`第十版/lake_pinn`](./第十版/lake_pinn)。第十版定位为 zero-profile reconstruction-state 诊断主线，重点验证 EOF/PCA init-net、daily-memory 分支、init-physics-rollout、unlabeled heat-closure windows、GPU batch autotune、source package 导出和 initial-column evaluation。

## Highlights

- 第十版是当前主线，主记录为 `RECON_R42_GROUPKFOLD_V1_fold01-05_6H_SUBMIT`。
- 第九版已归档，保留为 reconstruction diagnostic / support transfer 前一代基线。
- 第八版仍是历史跨湖泛化 R9 代表结果；第十版 R42 是 diagnostic-only，不直接替代长时段泛化结论。
- 仓库保留源码、测试、脚本、说明文档和精选代表图；完整数据、checkpoint、CSV、日志和实验目录保留在本地或外部存储。

## 项目结构

```text
PINN-
|-- 第十版/              # 当前主线
|   |-- lake_pinn/
|   |-- tests/
|   `-- scripts/
|-- 归档/                # 历史版本
|-- docs/figures/        # 精选公开图像
|-- DATA.md
|-- MODEL.md
|-- EXPERIMENTS.md
`-- REPRODUCE.md
```

## 第十版结果快照

第十版公开主记录采用 `RECON_R42_GROUPKFOLD_V1_fold01-05_6H_SUBMIT`。该记录是 5-fold group-kfold smoke/diagnostic，5/5 folds 完成到 epoch 2；heldout 指标只用于诊断，不作为最终 formal transfer claim。

| 指标 | 数值 |
|---|---:|
| train mean RMSE mean | 0.564 |
| validation mean RMSE mean | 0.288 |
| heldout transition RMSE mean | 0.525 |
| heldout transition RMSE median | 0.448 |
| heldout transition RMSE min / max | 0.340 / 0.884 |
| heldout persistence RMSE mean | 0.381 |
| Kd multiplier mean / p95 mean | 1.009 / 1.017 |
| Kd saturation fraction | 0.000 |
| unlabeled heat-closure loss | 0.000 |

![LakePINN v10 R42 fold RMSE heatmap](./docs/figures/lakepinn_v10_r42_groupkfold_fold_rmse_heatmap.png)

![LakePINN v10 R42 heldout RMSE](./docs/figures/lakepinn_v10_r42_groupkfold_heldout_rmse_bar.png)

![LakePINN v10 R42 heat-closure diagnostic](./docs/figures/lakepinn_v10_r42_heatclosure_diagnostic.png)

核心判断：

- R42 说明第十版主线可以在 group-kfold diagnostic 设置下稳定跑通。
- fold02 是主要弱点，heldout groups 为 `crystal_bog / erken / namco / toolik`，heldout RMSE `0.884`。
- no-profile heat-closure loss 为 `0.000`，说明当前阈值和 gating 下约束信号过弱，下一版需要调整。

更多指标见 [`第十版/实验总结.md`](./第十版/实验总结.md) 和 [`EXPERIMENTS.md`](./EXPERIMENTS.md)。

## 版本演进与更新说明

| 版本 | 位置 | 定位 | 最好/代表记录 |
|---|---|---|---|
| 第十版 | [`第十版/`](./第十版) | 当前 zero-profile reconstruction-state 诊断主线 | R42 group-kfold diagnostic heldout RMSE mean 0.525 |
| 第九版 | [`归档/第九版/`](./归档/第九版) | reconstruction diagnostic / support transfer 主线 | L3 overnight few-shot 30d RMSE 2.486，60d RMSE 2.396 |
| 第八版 | [`归档/第八版/`](./归档/第八版) | 跨湖泛化主线，强化 metadata、LST dropout、warm-column 和 roll60 | R9 epoch0099，三湖平均 RMSE 3.856 |
| 第七版 | [`归档/第七版/`](./归档/第七版) | reconstruction-state / state-space forecaster 前一代主线 | Mendota T5 full free-roll RMSE 1.190 |
| 第六版 | [`归档/第六版/`](./归档/第六版) | multi-lake / few-shot 迁移基线 | Kinneret T54 RMSE 0.670，Sparkling few-shot RMSE 1.402 |
| 第五版 | [`归档/第五版/`](./归档/第五版) | Mohonk raw PINN 单湖基线 | T34 raw PINN RMSE 1.250 |
| 第四版 | [`归档/第四版/`](./归档/第四版) | 模块化 LakePINN 对照 | run9 resume RMSE 1.478 |
| 第三版 | [`归档/第三版/`](./归档/第三版) | 11 维 PINN + PPO/Kalman 单文件主线 | 11维测试/九 RMSE 1.498 |
| 第二版 | [`归档/第二版/`](./归档/第二版) | 旧输入结构 PPO 数值对照 | 策略测试/七 RMSE 1.051 |
| 第一版 | [`归档/第一版/`](./归档/第一版) | 早期下载、处理和建模流程整理 | 历史留档 |
| 第零版 | [`归档/第零版/`](./归档/第零版) | 最早期集中式脚本和流程验证 | 历史留档 |

### 第十版：zero-profile reconstruction-state 诊断

第十版将第九版的 reconstruction 诊断继续收敛到可验证的工程主线：zero-profile EOF/PCA init-net、daily-memory 分支、init-physics-rollout 主线、unlabeled heat-closure windows、GPU batch autotune 和 source package hygiene。主记录 R42 完成 5-fold group-kfold epoch-2 diagnostic，heldout transition RMSE mean 为 `0.525`，但 R42 只作为早期诊断，不替代长时段泛化结论。

### 第九版：reconstruction diagnostic / support transfer

第九版围绕 zero-profile reconstruction、support profile 校正和 few-shot 迁移失败来源展开。主记录 `RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1` 的 few-shot 30d / 60d RMSE 分别为 `2.486` / `2.396`，用于确认 support profile 校正方向和后续 zero-profile 改造优先级。

### 第八版：跨湖泛化主线

第八版把第七版的 state-space forecaster 推向更严格的跨湖泛化评估。主结果 `R9_WARMCOL_LST_SURFICE_ALL38_holdout3_120ep_roll60_export25_20260605` 使用 38 个 lake-year，其中 3 个整湖 heldout；`epoch0099` 三湖平均 RMSE 为 `3.856`，平均绝对 bias 为 `2.161`。`carvins_cove_2022` 的冷偏仍是关键问题。

### 第七版：状态空间预测器

第七版从第六版 multi-lake / few-shot 路线切换到 reconstruction-state / state-space forecaster。代表实验 `T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud` 在 Mendota 2020 full free-roll 上达到 RMSE `1.190`、MAE `0.837`、bias `-0.466`。

### 第六版：多湖迁移与 few-shot

第六版引入 global backbone + lake adapter/residual，用于多湖联合训练、leave-one-lake zero-shot 和目标湖 few-shot 适配。代表结果包括 Kinneret `T54` RMSE `0.670`、MAE `0.443`、bias `-0.159`，以及 Sparkling few-shot RMSE `1.402`。

### 第五版：Mohonk raw PINN 基线

第五版将预测主线收敛到 raw PINN，默认不再依赖 rolling prediction、Kalman 同化或 PPO 预测调度。代表实验 `T34_rawPINN_noRolling_noKalman_noPPO_20260511` 在 Mohonk 2017 上达到 RMSE `1.250`、MAE `0.830`、bias `-0.069`。

### 第四版：模块化 LakePINN 对照

第四版把第三版之后的主线从单文件脚本整理成 `lake_pinn` 包，并新增 `python -m lake_pinn` 命令行入口。代表实验 `run9_resume_train_20260509` 达到 RMSE `1.478`、MAE `0.999`、bias `0.060`。

### 第三版：11 维 PINN + PPO/Kalman

第三版保留 11 维 PINN / PPO 主线、热收支 A 线和 `lake_profile_scorecard.py` 分层评分工具。代表实验 `11维测试/九` 达到 RMSE `1.498`、MAE `1.061`、bias `0.188`。

### 第二版：旧输入结构 PPO 策略控制

第二版承载早期 `PINN + PPO` 联合控制路线，默认脚本为 `PPO策略控制.py`。代表实验 `策略测试/七` RMSE `1.051`、MAE `0.734`、bias `0.024`，是历史数值最优之一，但工程组织已经不适合作为当前主线。

### 第一版与第零版：早期流程验证

第一版将下载、数据处理、可视化和建模流程整理成更清晰的目录入口。第零版是最早期集中式脚本阶段，完成 ERA5、MODIS LST、PINN 建模、温度深度热图和观测模拟对比的基础流程验证。

## 快速开始

```powershell
pip install -r requirements.txt
Push-Location ".\第十版"
python -m lake_pinn --help
Pop-Location
```

训练或导出需要准备 manifest，示例：

```powershell
Push-Location ".\第十版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --output-dir "..\outputs\v10_run" `
  --epochs 20 `
  --device cpu
Pop-Location
```
