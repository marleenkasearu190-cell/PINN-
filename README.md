# PINN-

湖泊温度剖面预测与物理约束 LakePINN 实验仓库。项目以 ERA5 气象强迫、MODIS/LST 表面温度、湖泊静态属性和剖面观测为输入，预测逐日水温深度剖面 `T(z,t)`，并围绕物理一致性、跨湖泛化、长时段滚动稳定性和观测点误差进行评估。

当前推荐继续开发的主线是 [`第八版/lake_pinn`](./第八版/lake_pinn)。第八版延续 reconstruction-state / state-space forecaster 路线，重点推进跨湖泛化、扩展 metadata、temporal adaptive、LST dropout、segment LST weak loss、warm-column heat-content loss、roll60/export25 导出，以及 PGDL-WRR benchmark 对照工具化。

## 项目状态

这是研究型实验仓库，不是已发布的 Python 包。仓库只保存源码、测试、脚本、说明文档和少量代表图；完整实验目录、原始数据、checkpoint、预测 CSV、日志和压缩包保留在本地或单独发布位置。

主要目录：

```text
PINN-
|-- 第八版/              # 当前主线
|   |-- lake_pinn/
|   |-- tests/
|   |-- scripts/
|   `-- benchmarks/
|-- 归档/                # 历史版本
|-- docs/figures/        # 精选公开图像
|-- DATA.md
|-- MODEL.md
|-- EXPERIMENTS.md
`-- REPRODUCE.md
```

## 第八版结果快照

第八版主结果采用 `R9_WARMCOL_LST_SURFICE_ALL38_holdout3_120ep_roll60_export25_20260605` 的 `epoch0099` 导出结果。该实验使用 38 个 lake-year，其中 35 个训练、3 个整湖 heldout，rollout 约束为 60 天，并统一在 0-25 m 范围导出和评估。

| Heldout lake-year | RMSE | bias | 说明 |
|---|---:|---:|---|
| `lacawac_2016` | 2.405 | -0.244 | 三湖中最稳定，系统偏差较小 |
| `carvins_cove_2022` | 5.456 | -4.047 | 仍存在明显冷偏，是下一版重点问题 |
| `lake_maggiore_2024` | 3.708 | -2.191 | 比 R7 改善，但仍有冷偏 |
| 三湖平均 | 3.856 | 2.161 absolute bias | 相比 R7 的 4.158 / 2.455 有整体改善 |

![LakePINN v8 Lacawac year heatmap](./docs/figures/lakepinn_v8_r9_epoch0099_lacawac_year_heatmap.png)

![LakePINN v8 Carvins Cove scorecard](./docs/figures/lakepinn_v8_r9_epoch0099_carvins_cove_scorecard_report.png)

![LakePINN v8 Lake Maggiore bias contour](./docs/figures/lakepinn_v8_r9_epoch0099_lake_maggiore_bias_contour_heatmap.png)

更多指标和代表图见 [`第八版/实验总结.md`](./第八版/实验总结.md) 与 [`EXPERIMENTS.md`](./EXPERIMENTS.md)。

## 版本演进

| 版本 | 位置 | 定位 | 最佳/代表记录 |
|---|---|---|---|
| 第八版 | [`第八版/`](./第八版) | 当前跨湖泛化主线，强化 extended metadata、temporal adaptive、LST dropout、warm-column heat-content 和 roll60 | R9 epoch0099，三湖平均 RMSE 3.856 |
| 第七版 | [`归档/第七版/`](./归档/第七版) | reconstruction-state / state-space forecaster 前一代主线 | Mendota T5 full free-roll RMSE 1.190 |
| 第六版 | [`归档/第六版/`](./归档/第六版) | multi-lake / few-shot 迁移基线 | Kinneret T54 RMSE 0.670，Sparkling few-shot RMSE 1.402 |
| 第五版 | [`归档/第五版/`](./归档/第五版) | Mohonk raw PINN 单湖基线 | T34 raw PINN RMSE 1.250 |
| 第四版 | [`归档/第四版/`](./归档/第四版) | 模块化 LakePINN 对照 | run9 resume RMSE 1.478 |
| 第三版 | [`归档/第三版/`](./归档/第三版) | 11 维 PINN + PPO/Kalman 单文件主线 | 11维测试/九 RMSE 1.498 |
| 第二版 | [`归档/第二版/`](./归档/第二版) | 旧输入结构 PPO 数值对照 | 策略测试/七 RMSE 1.051 |
| 第一版/第零版 | [`归档/`](./归档) | 早期数据处理、下载和建模流程 | 历史留档 |

## 快速开始

```powershell
pip install -r requirements.txt
Push-Location ".\第八版"
python -m lake_pinn --help
Pop-Location
```

训练或导出需要准备 manifest，示例：

```powershell
Push-Location ".\第八版"
python -m lake_pinn `
  --manifest "..\path\to\manifest.json" `
  --output-dir "..\outputs\v8_run" `
  --epochs 120 `
  --device cpu
Pop-Location
```

## 不提交的内容

以下内容保留在本地实验目录，不直接提交到 GitHub：

- 原始数据、验证数据和标准化输入数据。
- 训练 checkpoint，例如 `*.pt`、`*.pth`、`*.ckpt`。
- 完整 `experiments/`、`external/`、`_archive/` 目录。
- 预测 CSV、scorecard CSV、diagnostics CSV、批量 PNG、日志、zip 包和进程文件。
- `__pycache__/`、`.pytest_cache/`、`.pyc` 等缓存。

仓库中的 PNG 仅限 `docs/figures/` 下精选代表图。
