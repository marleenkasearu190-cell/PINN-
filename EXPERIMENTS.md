# 实验记录摘要

本文档汇总仓库中适合公开展示的实验结论。完整实验目录、checkpoint、预测 CSV 和日志不直接提交，只保留关键指标和代表图。

## 第九版主记录

主记录：`RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1`，采用 `best_by_val_rolling` checkpoint，选择 epoch `47`。

| 实验 | epoch | selection score | few-shot 30d RMSE | few-shot 60d RMSE | rolling-start 30d RMSE | rolling-start 60d RMSE | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| `RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1` | 47 | 2.441 | 2.486 | 2.396 | 1.995 | 2.233 | 第九版主记录 |
| `RECON_L3_SUPPORT_DELTA_MAGNITUDE_DIAG_v1` | 11 | 3.630 | 3.828 | 3.432 | 2.226 | 2.528 | L3 诊断对照 |
| `RECON_L2_SINGLELAKE_RECON_SANITY_v1` | 17 | 3.007 | - | - | 2.558 | 3.457 | 单湖 reconstruction sanity；无 few-shot 指标 |

诊断结论：

- `RECON_DIAG_1D_LOOP_CONSISTENCY_v1` 通过：checkpoint validation transition 1d RMSE `0.300`，rolling 1d RMSE `0.308`，ratio `1.027`。
- `RECON_DIAG_SUPPORT_UPDATE_EFFECT_v1` 表明 support update 在 query-start 处有效：checkpoint validation base RMSE `6.162`，corrected RMSE `6.056`，delta `-0.106`，direction hit `0.967`，leak count `0`。
- R11 export-only 诊断显示观测锚定能明显降低 export 误差：free RMSE `4.07`，support_train `0.624`，profile_train `0.531`，support_all `0.324`，profile_all `0.0561`。该组仅作为诊断，不作为 formal transfer claim。
- R11 lake-type 诊断显示 Natural 与 Reservoir 偏差方向不同：Natural RMSE `2.74`、bias `+1.88`；Reservoir RMSE `4.35`、bias `-3.01`。

代表图：

- [reconstruction framework](./docs/figures/lakepinn_v9_generated_recon_framework_operator_flow.png)
- [R11 export modes RMSE](./docs/figures/lakepinn_v9_generated_r11_export_modes_rmse.png)
- [R11 lake-type RMSE and bias](./docs/figures/lakepinn_v9_generated_r11_laketype_bias_rmse.png)

## 第八版主结果

主结果：`R9_WARMCOL_LST_SURFICE_ALL38_holdout3_120ep_roll60_export25_20260605`，采用 `epoch0099` 导出结果。

| Lake-year | 角色 | RMSE | MAE | bias | 代表图 |
|---|---|---:|---:|---:|---|
| `lacawac_2016` | heldout | 2.405 | - | -0.244 | [year](./docs/figures/lakepinn_v8_r9_epoch0099_lacawac_year_heatmap.png), [bias](./docs/figures/lakepinn_v8_r9_epoch0099_lacawac_bias_contour_heatmap.png), [scorecard](./docs/figures/lakepinn_v8_r9_epoch0099_lacawac_scorecard_report.png) |
| `carvins_cove_2022` | heldout | 5.456 | - | -4.047 | [year](./docs/figures/lakepinn_v8_r9_epoch0099_carvins_cove_year_heatmap.png), [bias](./docs/figures/lakepinn_v8_r9_epoch0099_carvins_cove_bias_contour_heatmap.png), [scorecard](./docs/figures/lakepinn_v8_r9_epoch0099_carvins_cove_scorecard_report.png) |
| `lake_maggiore_2024` | heldout | 3.708 | - | -2.191 | [year](./docs/figures/lakepinn_v8_r9_epoch0099_lake_maggiore_year_heatmap.png), [bias](./docs/figures/lakepinn_v8_r9_epoch0099_lake_maggiore_bias_contour_heatmap.png), [scorecard](./docs/figures/lakepinn_v8_r9_epoch0099_lake_maggiore_scorecard_report.png) |

汇总判断：

- 三湖平均 RMSE 为 `3.856`，平均绝对 bias 为 `2.161`。
- 相比第七版对照 R7 的三湖平均 RMSE `4.158` 和平均绝对 bias `2.455`，R9 有整体改善。
- `carvins_cove_2022` 仍有约 `-4.047 C` 冷偏，说明入流/热源、湖泊自适应参数或 warm-column 约束仍需继续处理。
- R9 原计划 120 epoch，实际在 epoch100 后停止；公开记录采用最后可用的 `epoch0099` 导出结果。

## 第八版背景实验

早期 M2 消融用于确认 lake-adaptive 方向：

| 实验 | 结构 | heldout 30d RMSE | heldout 60d RMSE | 结论 |
|---|---|---:|---:|---|
| R0 | adaptive off | 4.929 | 5.971 | 基线 |
| R1 | adaptive kd | 4.736 | 5.732 | 有改善 |
| R2 | adaptive exchange | 4.651 | 5.625 | 有改善 |
| R3 | adaptive kz + convective | 4.798 | 5.805 | 不如 R2/R4 |
| R4 | adaptive all | 4.570 | 5.591 | M2 40 epoch 组内最好 |

PGDL-WRR benchmark 用于公平 no-LST 对照：

| 模型 | overall RMSE | MAE | bias | 说明 |
|---|---:|---:|---:|---|
| PGDL official | 1.006 | 0.704 | -0.397 | 官方 PGDL-WRR 2019 对照 |
| LakePINN no-LST | 3.002 | 2.062 | 1.768 | 工具化 benchmark 记录，不作为第八版主结果 |

## 各版本最佳记录

| 版本 | 代表实验 | RMSE | MAE | bias | 说明 |
|---|---|---:|---:|---:|---|
| 第九版 | `RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1` | 2.441 | - | - | selection score；few-shot 30d/60d 为 2.486 / 2.396 |
| 第八版 | `R9_WARMCOL...epoch0099` | 3.856 | - | 2.161 abs | 三湖 heldout 平均，跨湖泛化主结果 |
| 第七版 | `T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud` | 1.190 | 0.837 | -0.466 | Mendota 2020 full free-roll 最稳 |
| 第六版 | `T54_Kinneret修复时间步长后T51微调` | 0.670 | 0.443 | -0.159 | Kinneret 单湖最优 |
| 第六版 | `Sparkling lakeSpecificResidual softBound surfaceUniform 20d` | 1.402 | 1.078 | 0.340 | Sparkling few-shot 最优 |
| 第五版 | `T34_rawPINN_noRolling_noKalman_noPPO_20260511` | 1.250 | 0.830 | -0.069 | Mohonk raw PINN 基线 |
| 第四版 | `run9_resume_train_20260509` | 1.478 | 0.999 | 0.060 | 模块化版本续训练候选 |
| 第三版 | `11维测试/九` | 1.498 | 1.061 | 0.188 | 11 维 PINN + PPO/Kalman 稳定对照 |
| 第二版 | `策略测试/七` | 1.051 | 0.734 | 0.024 | 旧输入结构的历史数值最优 |

## 代表图索引

- 第九版：reconstruction operator framework、R11 export mode RMSE、R11 lake-type RMSE/bias 已整理到 `docs/figures/`。
- 第八版 R9：Lacawac、Carvins Cove、Lake Maggiore 的 year heatmap、bias contour 和 scorecard 已保留。
- 第七版 T5：Mendota year heatmap、scorecard、discrete point evaluation、bias contour 已保留。
- 第六版：Kinneret T54、Sparkling few-shot、Mendota LOO zero-shot 代表图已保留。
- 第五版及以前：Mohonk raw PINN、11D 主线、RMSE/bias 月深度诊断图已保留。
