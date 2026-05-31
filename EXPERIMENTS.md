# 实验结论摘要

本文档记录当前仓库中关键版本的定位和已知结论。完整实验输出、checkpoint 和大批量 CSV 不放入 GitHub；少量代表热图和评分摘要保存在 `docs/figures/`。

## 关键版本

| 版本 | 对应脚本 | 对应本地实验 | 结论 |
|---|---|---|---|
| 第二版旧输入 PPO | `归档/第二版/PPO策略控制.py` | `策略测试/七` | Mohonk 2017 数值 RMSE 最低，但属于旧输入结构 |
| 第三版 11 维主线 | `归档/第三版/PPO策略调控_11维主线_20260426.py` | `11维测试/九` | 已归档的 11 维单文件主线对照 |
| 第三版热收支 A 线 | `归档/第三版/PPO策略调控_热收支A线_20260428.py` | `11维测试/十一` 到 `11维测试/十七` | 已归档的热收支实验线 |
| 第四版模块化 LakePINN | `归档/第四版/lake_pinn/` | `run9_resume_train_20260509`、`official_predict_*` | 已归档的模块化 PINN/PPO/Kalman 对照 |
| 第五版 raw PINN | `归档/第五版/lake_pinn/` | `T34_rawPINN_noRolling_noKalman_noPPO_20260511` | 已归档的 Mohonk raw PINN 基线 |
| 第六版 multi-lake / few-shot | `归档/第六版/lake_pinn/` | `T54_Kinneret...`、`Sparkling few-shot 20d`、`LOO_03 Mendota` | 已归档的多湖迁移基线 |
| 第七版 reconstruction-state | `第七版/lake_pinn/` | `T5_Mendota2020...`、`T4_Mendota2020...`、`Sparkling R2` | 当前候选主线，适合继续开发 |

## 当前参考结果

| 实验 | RMSE | MAE | bias | 当前判断 |
|---|---:|---:|---:|---|
| `T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud` | 1.190 | 0.837 | -0.466 | 第七版主结果，Mendota 2020 full free-roll 最稳；heldout transition RMSE 为 0.331 |
| `T4_Mendota2020_bulkFlux_kz20_heat005_200ep_sample4_eval10_rerun1` | 1.529 | 1.043 | -0.762 | 第七版对照，transition RMSE 0.330，但长时段稳定性不如 T5 |
| `R_free_roll_fix/sparkling_R2_reslim025_20260517` | 3.266 | 2.504 | 0.250 | 第七版 Sparkling 2003 future-year free-roll 对照；transition RMSE 为 0.297 |
| `T54_Kinneret修复时间步长后T51微调` | 0.670 | 0.443 | -0.159 | 归档第六版 Kinneret 单湖最优，物理底线通过 |
| `Sparkling lakeSpecificResidual softBound surfaceUniform 20d` | 1.402 | 1.078 | 0.340 | 第六版 few-shot 最优，20 个剖面日期适配且物理底线通过 |
| `LOO_03 Mendota zero-shot` | 1.079 | - | - | 第六版 leave-one-lake 最优，训练摘要 test RMSE |
| `T34_rawPINN_noRolling_noKalman_noPPO_20260511` | 1.250 | 0.830 | -0.069 | 第五版 Mohonk raw PINN 基线，scorecard v2=80.71；密度稳定性仍需优化 |
| `策略测试/七` | 1.051 | 0.734 | 0.024 | 数值精度最好，已归档为旧输入对照 |
| `11维测试/九` | 1.498 | 1.061 | 0.188 | 已归档 11 维稳定对照 |
| `run9_resume_train_20260509` | 1.478 | 0.999 | 0.060 | 归档第四版续训候选增强，物理底线通过 |
| `official_predict_pinn` | 1.502 | 1.056 | 0.152 | PINN rolling 输出作为诊断，预测侧物理底线仍需谨慎 |
| `official_predict_kalman` | 1.567 | 1.084 | 0.095 | Kalman 同化输出物理底线通过，启用 Kalman 时优先用于展示和评分 |
| `11维测试/十六` | 1.557 | 1.106 | 0.213 | 热收支 A3 预测结果，数值接近主线 |
| `11维测试/十七` | 1.672 | 1.172 | 0.282 | A4 预测侧仍未完全通过物理底线 |

这些指标基于对应湖泊/年份预测 CSV 与剖面观测对齐后的结果。评价时不能只看 RMSE，还需要结合物理底线和季节过程。

## 第七版 reconstruction-state 结论

第七版将研究重点转向 reconstruction-state / state-space forecaster。模型从当前剖面状态、forcing 和湖泊属性推进下一步温度剖面，并用 heat-content transition、bulk turbulent flux、hypsometry-aware diffusion、segment rollout 和 free-roll 评价长时段稳定性。

主要判断：

- `T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud` 是当前公开主结果，Mendota 2020 full free-roll RMSE 为 1.190，MAE 为 0.837，bias 为 -0.466。
- T5 的 heldout transition RMSE 为 0.331，30d free-roll RMSE 为 1.000。
- `T4_Mendota2020_bulkFlux_kz20_heat005_200ep_sample4_eval10_20260524_rerun1` 的 heldout transition RMSE 为 0.330，略低于 T5，但 full free-roll RMSE 为 1.529，长时段稳定性不如 T5。
- `R_free_roll_fix/sparkling_R2_reslim025_20260517` 是 Sparkling 2003 future-year 对照，transition RMSE 为 0.297，free-roll RMSE 为 3.266，说明跨年长滚动仍需继续优化。

代表图像：

- [Mendota T5 年度热图](./docs/figures/lakepinn_v7_mendota_t5_year_heatmap.png) 和 [scorecard](./docs/figures/lakepinn_v7_mendota_t5_scorecard_report.png)
- [Mendota T5 离散点评估](./docs/figures/lakepinn_v7_mendota_t5_discrete_point_evaluation.png) 和 [bias contour](./docs/figures/lakepinn_v7_mendota_t5_bias_contour_heatmap.png)
- [Sparkling R2 年度热图](./docs/figures/lakepinn_v7_sparkling_r2_year_heatmap.png) 和 [scorecard](./docs/figures/lakepinn_v7_sparkling_r2_scorecard_report.png)

更完整的本轮总结见 [`第七版/实验总结.md`](./第七版/实验总结.md)。

## 第六版 multi-lake / few-shot 归档结论

第六版将研究重点从单湖 raw PINN 推进到多湖泛化和 few-shot 迁移。当前源码已移入 `归档/第六版/lake_pinn`，完整实验输出保留在本地第六版实验目录。

主要判断：

- `T54_Kinneret修复时间步长后T51微调` 是本轮 Kinneret 单湖数值最优结果，profile validation RMSE 为 0.670，MAE 为 0.443，bias 为 -0.159，scorecard 物理底线通过。
- `T57_Kinneret一月深层记忆修复` 的 RMSE 为 0.727，MAE 为 0.482，bias 为 -0.090，数值略低于 T54，但一月深层结构更稳，适合作为下一轮 warm/deep lake 物理项基础。
- `FewShot_Sparkling_from_LOO02_lakeSpecificResidual_softBound_surfaceUniform_20d` 是 Sparkling few-shot 最优记录，held-out RMSE 为 1.402，MAE 为 1.078，bias 为 0.340，scorecard 物理底线通过。
- `LOO_03_train_Mohonk_Sparkling_Erken2019_test_Mendota` 是当前 leave-one-lake 最好样例，最佳训练摘要 test RMSE 为 1.079。
- Mohonk 第六版 T55/T56 复评没有超过第五版 T34；Mohonk 继续使用 T34 作为公开对照更稳。

代表图像：

- [Kinneret T54 年度热图](./docs/figures/lakepinn_v6_kinneret_t54_year_heatmap.png) 和 [scorecard](./docs/figures/lakepinn_v6_kinneret_t54_scorecard_report.png)
- [Sparkling few-shot 年度热图](./docs/figures/lakepinn_v6_sparkling_fewshot_surfaceuniform_year_heatmap.png) 和 [scorecard](./docs/figures/lakepinn_v6_sparkling_fewshot_surfaceuniform_scorecard_report.png)
- [Mendota LOO zero-shot 年度热图](./docs/figures/lakepinn_v6_mendota_loo03_zeroshot_year_heatmap.png) 和 [scorecard](./docs/figures/lakepinn_v6_mendota_loo03_zeroshot_scorecard_report.png)

更完整的本轮总结见 [`归档/第六版/实验总结.md`](./归档/第六版/实验总结.md)。

## 分层评价原则

模型比较按以下顺序进行：

1. 物理底线：冬季逆温、夏季分层、秋季翻混、漂移和温跃层结构。
2. 关键季节过程：5 月升温、7 月表层高温、秋季翻混时间、冬季结构。
3. 数值精度：RMSE、MAE、bias、0-3 m 表层 RMSE、温跃层 RMSE。
4. 稳定性：重跑波动和 train/predict 一致性。
5. 图像观感：热图是否自然，作为最后参考。

## 热收支 A 线结论

热收支实验线的主要观察：

- A 组直接接入能量版热收支后，主线明显变差。
- A2 延迟弱权重后不再完全崩，但 drift 和秋季翻混仍有问题。
- A3 使用区间平均通量后，drift 明显改善，RMSE 接近主线。
- A4 在训练侧通过物理底线，但预测侧仍存在秋季翻混或漂移不稳。

当前判断：

```text
热收支 A 线值得继续，但暂不替代 11维测试/九 主线。
```

下一步建议优先做预测侧单因素对照，检查 embedded PPO policy、Kalman 参数和 heat-budget loss 在预测阶段的耦合。

## 第五版 raw PINN 结论

第五版曾将研究重心调整为 raw PINN 主线：优先通过训练侧结构约束学习物理形态，而不是依赖 rolling、Kalman 或 PPO 预测后处理。当前源码已移入 `归档/第五版/lake_pinn/`，对应本地 `T34_rawPINN_noRolling_noKalman_noPPO_20260511` 分支。

主要判断：

- 默认输入维度升级到 27 维，支持扩展 forcing、past-only weather memory 和 previous-state memory。
- 预测侧默认输出 raw PINN；Kalman、rolling 和 PPO 仍保留为可选对照或诊断。
- 训练侧新增 profile-grid physics、density regularization 和 bottom slow-change，用来把热图形态约束前移到训练阶段。
- scorecard 升级到 v2，重点检查季节覆盖、物理失败项和候选输出排序。
- 当前代表实验 `T34_rawPINN_noRolling_noKalman_noPPO_20260511` 的 RMSE 为 1.250，MAE 为 0.830，bias 为 -0.069，scorecard v2 为 80.71。
- T34 的数值精度已明显好于第三版和第四版对照，但 density stability 仍未通过，是下一步优化重点。
- T34 完整输出仍保留在本地，暂不提交 checkpoint、预测 CSV 或批量图像。

## 第四版模块化归档结论

第四版将 run9 后续实验整理为 `lake_pinn` 包，便于把训练、预测、Kalman 同化、PPO 调度、标准输入构建和分层评分拆开维护。当前更新只提交源码和说明，不提交 checkpoint、完整预测 CSV 或训练输出。

主要判断：

- `run9_resume_train_20260509` 相比 `11维测试/九` 有小幅数值提升，RMSE 约为 1.478，MAE 约为 0.999，bias 约为 0.060。
- 官方预测输出中，`pinn_rolling` 保留为 PINN 预测诊断；启用 Kalman 时，`kalman_assimilated` 作为展示和评分优先输出。
- 第四版默认输入维度面向 17 维扩展 forcing；复用 11 维 checkpoint 或 run9 兼容训练时，需要显式设置 `--model-input-dim 11`。
- 该版本已移入 `归档/第四版/`，作为第五版前的模块化对照保留。

## checkpoint 定位

11 维归档对照复现建议使用第三版 11 维模型 checkpoint：

```text
11维测试/9/mohonk_lake_2017_pinn_model_checkpoint.pt
```

该 checkpoint 为 11 维输入模型，包含嵌入的 PPO policy bundle，可直接用于 `归档/第三版/PPO策略调控_11维主线_20260426.py --mode predict`，或在归档第四版中配合 `--model-input-dim 11` 做兼容预测。公开复现模板见 [`REPRODUCE.md`](./REPRODUCE.md)。

热收支 A 线可参考：

- `11维测试/16/mohonk_lake_2017_pinn_model_checkpoint.pt`
- `11维测试/17/mohonk_lake_2017_pinn_model_checkpoint.pt`

## 结果文件管理

建议把完整实验输出保留在本地，例如：

- `11维测试/`
- `策略测试/`
- `score_outputs_*`
- checkpoint 文件
- 批量预测热图和预测 CSV

GitHub 仓库只保存代码、文档、精简结论和少量代表性图表。需要公开更多结果时，再把评分摘要整理进论文或报告。
