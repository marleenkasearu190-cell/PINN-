# 第七版：Reconstruction-State LakePINN 主线

第七版把主线从直接预测 `T(z,t)` 的 PINN，切换为 reconstruction-state / state-space forecaster：模型学习从当前剖面状态、forcing 和湖泊属性推演下一步温度剖面，并用热含量、垂向扩散、密度稳定、free-roll 和 segment rollout 约束长时段稳定性。

本目录只保存源码、测试、云端运行脚本和说明文档；完整 `experiments/`、checkpoint、预测 CSV、压缩包和缓存文件不进入 GitHub。代表图和关键指标整理在根目录 `docs/figures/` 与本文档中。

## 目录结构

```text
第七版/
|-- README.md
|-- 更新说明.md
|-- 实验总结.md
|-- CLOUD_GPU_README.md
|-- requirements.txt
|-- lake_pinn/
|-- scripts/
`-- tests/
```

## 主要变化

- 新增 reconstruction-state forecaster，以完整剖面状态为模型状态变量，而不是独立逐点拟合温度。
- 支持多湖 manifest 训练、held-out lake / lake group 评估、state reconstruction 和 future-year free-roll。
- 加入 heat-content transition loss、bulk turbulent flux、hypsometry-aware diffusion、latent-reservoir freezing mode、hard density stability 和 rolling horizon evaluation。
- 增加批量化训练与评估开关，支持云端 GPU 长实验。
- 保留 scorecard、年度热图、bias contour、离散观测点评估和热闭合诊断输出。

## 当前代表实验

| 实验 | 口径 | RMSE | MAE | bias | 当前判断 |
|---|---|---:|---:|---:|---|
| `T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud` | Mendota 2020 full free-roll | 1.190 | 0.837 | -0.466 | 第七版主结果，长时段 free-roll/reconstruction 最稳 |
| `T5_Mendota2020_bulkFlux_kz15_heat005_200ep_cloud` | Mendota 2020 heldout transition | 0.331 | - | - | 单步 transition 精度接近 T4 |
| `T4_Mendota2020_bulkFlux_kz20_heat005_200ep_sample4_eval10_20260524_rerun1` | Mendota 2020 full free-roll | 1.529 | 1.043 | -0.762 | transition 略低，但长时段稳定性不如 T5 |
| `R_free_roll_fix/sparkling_R2_reslim025_20260517` | Sparkling 2003 future-year free-roll | 3.266 | 2.504 | 0.250 | Sparkling future-year 对照 |

T5 的 30 天 free-roll RMSE 为 1.000，说明 bulk-flux + Kz15 + heat-content 配置比 T4 更适合作为第七版公开主结果。

## 代表图像

![Mendota 2020 v7 T5 year heatmap](../../docs/figures/lakepinn_v7_mendota_t5_year_heatmap.png)

![Mendota 2020 v7 T5 scorecard report](../../docs/figures/lakepinn_v7_mendota_t5_scorecard_report.png)

![Mendota 2020 v7 T5 discrete point evaluation](../../docs/figures/lakepinn_v7_mendota_t5_discrete_point_evaluation.png)

![Sparkling 2003 v7 R2 year heatmap](../../docs/figures/lakepinn_v7_sparkling_r2_year_heatmap.png)

## 运行入口

```powershell
Push-Location ".\归档\第七版"
python -m lake_pinn --help
Pop-Location
```

训练或导出时需要提供 manifest：

```powershell
Push-Location ".\归档\第七版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --output-dir "..\outputs\v7_run" `
  --epochs 200 `
  --device cpu
Pop-Location
```

## 结果管理

以下内容不进入 GitHub：

- `experiments/` 完整实验目录。
- `*.pt`、`*.pth`、`*.ckpt` checkpoint。
- 预测 CSV、scorecard CSV、diagnostics CSV 和批量热图。
- `*.zip` 云端运行包。
- `__pycache__/`、`.pytest_cache/` 和 `.pyc` 缓存文件。
