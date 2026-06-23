# 实验记录摘要

本文档汇总仓库中适合公开展示的实验结论。完整实验目录、checkpoint、预测 CSV、scorecard CSV、日志和远程 raw outputs 不直接提交；公开仓库只保留关键指标、说明文档和少量代表图。

## 第十版主记录

主记录：`RECON_R42_GROUPKFOLD_V1_fold01-05_6H_SUBMIT`。

该记录是第十版 zero-profile reconstruction-state 主线的 5-fold group-kfold smoke/diagnostic。5/5 folds 完成到 epoch 2，heldout 指标为 diagnostic-only，不用于 checkpoint selection，也不作为最终 formal transfer claim。

| 指标 | mean | median | min | max |
|---|---:|---:|---:|---:|
| train mean RMSE | 0.564 | 0.584 | 0.459 | 0.621 |
| validation mean RMSE | 0.288 | 0.296 | 0.253 | 0.307 |
| heldout transition RMSE | 0.525 | 0.448 | 0.340 | 0.884 |
| heldout persistence RMSE | 0.381 | 0.388 | 0.291 | 0.454 |
| Kd multiplier mean | 1.009 | 1.011 | 0.949 | 1.057 |
| Kd multiplier p95 | 1.017 | 1.020 | 0.956 | 1.065 |
| heat content transition loss | 0.598 | 0.597 | 0.553 | 0.653 |
| unlabeled heat-closure loss | 0.000 | 0.000 | 0.000 | 0.000 |

Fold 诊断：

| Fold | Heldout groups | Heldout RMSE | Persistence RMSE | 判断 |
|---|---|---:|---:|---|
| fold01 | falling_creek_reservoir, green_lake_4, mohonk, sunapee | 0.581 | 0.454 | 一般 |
| fold02 | crystal_bog, erken, namco, toolik | 0.884 | 0.388 | 当前主要弱点 |
| fold03 | barco, kinneret, sammamish, sparkling, trout_lake | 0.373 | 0.346 | 稳定 |
| fold04 | beaverdam_reservoir, el_val, lake_washington, suggs | 0.448 | 0.429 | 接近 persistence |
| fold05 | lough_feeagh, mendota, rimov, trout_bog | 0.340 | 0.291 | 稳定 |

代表图：

- [R42 fold RMSE heatmap](./docs/figures/lakepinn_v10_r42_groupkfold_fold_rmse_heatmap.png)
- [R42 heldout RMSE bar](./docs/figures/lakepinn_v10_r42_groupkfold_heldout_rmse_bar.png)
- [R42 heat-closure diagnostic](./docs/figures/lakepinn_v10_r42_heatclosure_diagnostic.png)

主要结论：

- R42 证明第十版主线可以在 group-kfold diagnostic 设置下跑通并输出一致指标。
- fold02 的 heldout RMSE `0.884` 暴露出浅湖、高纬湖和深水 OOD metadata 覆盖问题。
- no-profile heat-closure residual 远低于 `50 W/m2` dead-zone threshold，loss 为 `0.000`，说明该约束当前信号过弱。

## 第九版主记录

主记录：`RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1`，采用 `best_by_val_rolling` checkpoint，选择 epoch `47`。

| 实验 | epoch | selection score | few-shot 30d RMSE | few-shot 60d RMSE | rolling-start 30d RMSE | rolling-start 60d RMSE | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| `RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1` | 47 | 2.441 | 2.486 | 2.396 | 1.995 | 2.233 | 第九版主记录 |
| `RECON_L3_SUPPORT_DELTA_MAGNITUDE_DIAG_v1` | 11 | 3.630 | 3.828 | 3.432 | 2.226 | 2.528 | L3 诊断对照 |
| `RECON_L2_SINGLELAKE_RECON_SANITY_v1` | 17 | 3.007 | - | - | 2.558 | 3.457 | 单湖 reconstruction sanity |

第九版结论是 support update 能改善 query-start profile，但 Natural 与 Reservoir 的误差方向不同，后续需要更稳健的 observer 或湖泊类型条件化策略。

## 第八版主结果

主结果：`R9_WARMCOL_LST_SURFICE_ALL38_holdout3_120ep_roll60_export25_20260605`，采用 `epoch0099` 导出结果。

| Lake-year | 角色 | RMSE | MAE | bias | 代表图 |
|---|---|---:|---:|---:|---|
| `lacawac_2016` | heldout | 2.405 | - | -0.244 | [year](./docs/figures/lakepinn_v8_r9_epoch0099_lacawac_year_heatmap.png), [bias](./docs/figures/lakepinn_v8_r9_epoch0099_lacawac_bias_contour_heatmap.png), [scorecard](./docs/figures/lakepinn_v8_r9_epoch0099_lacawac_scorecard_report.png) |
| `carvins_cove_2022` | heldout | 5.456 | - | -4.047 | [year](./docs/figures/lakepinn_v8_r9_epoch0099_carvins_cove_year_heatmap.png), [bias](./docs/figures/lakepinn_v8_r9_epoch0099_carvins_cove_bias_contour_heatmap.png), [scorecard](./docs/figures/lakepinn_v8_r9_epoch0099_carvins_cove_scorecard_report.png) |
| `lake_maggiore_2024` | heldout | 3.708 | - | -2.191 | [year](./docs/figures/lakepinn_v8_r9_epoch0099_lake_maggiore_year_heatmap.png), [bias](./docs/figures/lakepinn_v8_r9_epoch0099_lake_maggiore_bias_contour_heatmap.png), [scorecard](./docs/figures/lakepinn_v8_r9_epoch0099_lake_maggiore_scorecard_report.png) |

三湖平均 RMSE 为 `3.856`，平均绝对 bias 为 `2.161`。相比第七版对照 R7 的三湖平均 RMSE `4.158` 和平均绝对 bias `2.455`，R9 有整体改善；`carvins_cove_2022` 仍有明显冷偏。

## 各版本最佳记录

| 版本 | 代表实验 | RMSE / score | MAE | bias | 说明 |
|---|---|---:|---:|---:|---|
| 第十版 | `RECON_R42_GROUPKFOLD_V1_fold01-05_6H_SUBMIT` | 0.525 | - | - | heldout transition RMSE mean，diagnostic-only |
| 第九版 | `RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1` | 2.441 | - | - | selection score；few-shot 30d/60d 为 2.486 / 2.396 |
| 第八版 | `R9_WARMCOL...epoch0099` | 3.856 | - | 2.161 abs | 三湖 heldout 平均，跨湖泛化主结果 |
| 第七版 | `T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud` | 1.190 | 0.837 | -0.466 | Mendota 2020 full free-roll |
| 第六版 | `T54_Kinneret修复时间步长后R51微调` | 0.670 | 0.443 | -0.159 | Kinneret 单湖最优 |
| 第六版 | `Sparkling lakeSpecificResidual softBound surfaceUniform 20d` | 1.402 | 1.078 | 0.340 | Sparkling few-shot 最优 |
| 第五版 | `T34_rawPINN_noRolling_noKalman_noPPO_20260511` | 1.250 | 0.830 | -0.069 | Mohonk raw PINN 基线 |
| 第四版 | `run9_resume_train_20260509` | 1.478 | 0.999 | 0.060 | 模块化版本续训练 |
| 第三版 | `11维测试/九` | 1.498 | 1.061 | 0.188 | 11 维 PINN + PPO/Kalman |
| 第二版 | `策略测试/七` | 1.051 | 0.734 | 0.024 | 旧输入结构的历史数值最优 |

## 代表图索引

- 第十版 R42：fold RMSE heatmap、heldout RMSE bar、heat-closure diagnostic 已整理到 `docs/figures/`。
- 第九版：reconstruction operator framework、R11 export mode RMSE、R11 lake-type RMSE/bias 已整理到 `docs/figures/`。
- 第八版 R9：Lacawac、Carvins Cove、Lake Maggiore 的 year heatmap、bias contour 和 scorecard 已保留。
- 第七版 T5：Mendota year heatmap、scorecard、discrete point evaluation、bias contour 已保留。
- 第六版：Kinneret T54、Sparkling few-shot、Mendota LOO zero-shot 代表图已保留。
- 第五版及以前：Mohonk raw PINN、11D 主线、RMSE/bias 月-深度诊断图已保留。
