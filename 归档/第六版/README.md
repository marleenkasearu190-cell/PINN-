# 第六版：多湖 Global Adapter 与 Few-shot 迁移

第六版是在第五版 raw PINN 主线上的扩展版本，现在已归档为第七版前的 multi-lake / few-shot 迁移基线。它继续保留单湖 raw PINN 训练能力，同时新增多湖 global adapter、湖泊静态属性输入、warm/deep lake 物理门控、few-shot 目标湖残差适配，以及 Richardson-number-dependent eddy diffusivity 的物理实现与测试。

本目录只提交源码、测试和实验摘要，不包含 checkpoint、预测 CSV 或完整实验输出。第六版代表图像已整理到根目录 `docs/figures/`。

## 目录结构

```text
归档/第六版/
|-- README.md
|-- 更新说明.md
|-- 实验总结.md
|-- lake_pinn/
`-- tests/
```

## 主要变化

- 新增 `global_adapter` 架构：共享 backbone + lake-attribute FiLM/adapter residual，用于多湖联合训练。
- 新增 `multilake_global_adapter.py` 和 `fewshot_adapter.py`，支持 leave-one-lake zero-shot 与少量目标湖剖面日期的 few-shot 适配。
- 输入从单湖 forcing 扩展到湖泊静态属性、LST 质量权重、地理/湖型特征和 warm-deep-lake 门控。
- 物理项强化到密度稳定、温跃层形态、深层慢变化、Jan deep memory、时间步长热收支和 Richardson 数扩散。
- 新增 `tests/test_physics_diffusivity.py`，用于检查稳定/不稳定密度梯度下扩散系数响应方向。

## 本轮最佳实验记录

| 实验线 | 最佳/代表实验 | RMSE | MAE | bias | 物理底线 | 代表图 |
|---|---|---:|---:|---:|---|---|
| Kinneret 单湖 raw PINN | `T54_Kinneret修复时间步长后T51微调` | 0.670 | 0.443 | -0.159 | 通过 | [年度热图](../../docs/figures/lakepinn_v6_kinneret_t54_year_heatmap.png)、[scorecard](../../docs/figures/lakepinn_v6_kinneret_t54_scorecard_report.png) |
| Sparkling few-shot | `lakeSpecificResidual_softBound_surfaceUniform_20d` | 1.402 | 1.078 | 0.340 | 通过 | [年度热图](../../docs/figures/lakepinn_v6_sparkling_fewshot_surfaceuniform_year_heatmap.png)、[scorecard](../../docs/figures/lakepinn_v6_sparkling_fewshot_surfaceuniform_scorecard_report.png) |
| Mendota leave-one-lake zero-shot | `LOO_03_train_Mohonk_Sparkling_Erken2019_test_Mendota` | 1.079 | - | - | 训练摘要口径 | [年度热图](../../docs/figures/lakepinn_v6_mendota_loo03_zeroshot_year_heatmap.png)、[scorecard](../../docs/figures/lakepinn_v6_mendota_loo03_zeroshot_scorecard_report.png) |
| Mohonk 对照 | `T34_rawPINN_noRolling_noKalman_noPPO_20260511` 离散点评估 | 1.227 | 0.807 | -0.090 | 第五版基线对照 | [双湖离散点评估](../../docs/figures/lakepinn_v6_two_lake_discrete_point_evaluation.png) |

更完整的本轮实验说明见 [`实验总结.md`](./实验总结.md)。

## 运行入口

```powershell
Push-Location ".\归档\第六版"
python -m lake_pinn --help
Pop-Location
```

## 测试

```powershell
Push-Location ".\归档\第六版"
python .\tests\test_physics_diffusivity.py
python -m compileall -q .\lake_pinn
Pop-Location
```

## 结果管理

以下内容保留在本地第六版实验目录，或单独的数据发布位置，不直接进入 GitHub：

- `*.pt` checkpoint。
- 预测输出 `*.csv`。
- 批量 scorecard、bias contour、discrete evaluation 和年度热图。
- smoke、LOO、few-shot 和 Kinneret T 系列的完整实验目录。

GitHub 中只保留源码、测试、摘要表和少量代表图像。
