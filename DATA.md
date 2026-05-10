# 数据说明

本仓库不直接上传原始数据、验证数据和训练输出。运行第四版模块化入口或归档第三版脚本前，需要在本地准备 ERA5 forcing、LST 表面温度和剖面观测数据。

## 数据角色

| 数据 | 是否必须 | 用途 |
|---|---|---|
| ERA5 daily forcing | 必须 | 提供气温、风速、短波辐射等外部强迫 |
| LST surface observation | 必须 | 提供湖泊表面温度观测或表层约束 |
| Profile observations | 训练时建议提供 | 用于 PINN observation loss、validation、Kalman assimilation 和 test 评价切分 |
| Static lake metadata | 可选 | 最大深度、面积、纬度；代码对 Mohonk 和 Mendota 有默认值 |
| Bottom temperature / FLake fields | 可选 | 用于底温、混合层深度等结构诊断或辅助约束 |

## ERA5 / Forcing CSV

脚本通过 `--era5` 读取日尺度 forcing CSV。建议至少包含：

| 字段 | 含义 |
|---|---|
| `Date` | 日期，格式如 `2017-01-01` |
| 气温列 | 会被整理为 `T_air_C` 或 `surface_air_temp` |
| 风速列 | 会被整理为 `wind_speed_m_per_s` |
| 短波辐射列 | 会被整理为 `Solar_W_m2` |

如果存在下列列，脚本也会尝试使用：

| 字段 | 含义 |
|---|---|
| `relative_humidity` | 相对湿度 |
| `surface_pressure` | 表面气压 |
| `lmld_m` / `MixedLayerDepth_m` | 混合层深度 |
| `lblt_C` / `BottomTemp_C` | 湖底温度 |
| `ltlt_C` | 全湖平均温度或相关 FLake 诊断量 |

不同数据源的列名可能不完全相同，脚本内部会做一定的列名兼容和缺失值填补。

## LST CSV

脚本通过 `--lst` 读取湖泊表面温度 CSV。建议包含：

| 字段 | 含义 |
|---|---|
| `Date` | 日期 |
| `LST_surface_C` 或可转换为 LST 的温度列 | 湖泊表面温度，单位摄氏度 |
| `LST_surface_K` | 可选，单位 Kelvin；脚本可转换为摄氏度 |

如果存在 `SurfaceBulkTarget_C`，脚本会优先使用它作为表层 bulk 目标。它比皮肤温度更接近 0-1 m 表层水温，适合用于 Kalman 表层同化和表层约束。

## Profile Observation CSV

脚本通过 `--profile-obs` 读取剖面观测。支持两种格式。

长表格式：

```csv
Date,Depth_m,Temperature_C
2017-01-01,0,1.2
2017-01-01,1,3.4
```

宽表格式：

```csv
Date,Temp_0m,Temp_1m,Temp_2m,Temp_3m
2017-01-01,1.2,3.4,3.5,3.4
```

评分脚本 `lake_profile_scorecard.py` 同样支持这两种格式。

## Profile Split

第四版和归档第三版默认使用 `--profile-split-mode time_blocked`。这会把剖面观测按时间块切分为：

| 子集 | 用途 |
|---|---|
| `train` | 进入 PINN observation loss |
| `val` | 用于模型选择、早停和 PPO reward 对齐 |
| `assim` | 用于 Kalman profile assimilation |
| `test` | 只用于最终评价，不参与训练或同化 |

这个切分方式可以降低时间泄露风险，比随机深度交错切分更接近真实预测任务。

## Kalman 同化数据

Kalman 预测阶段可以同化：

- `assim` profile observations：剖面温度观测。
- `SurfaceBulkTarget_C` 或 `LST_surface_C`：表层观测链路。
- `BottomTemp_C`：可选底温信息。

`test` profile truth 不应在预测阶段被同化，只用于最终评分。

## 不上传数据的原因

原始数据、ERA5、LST、验证剖面、预测 CSV、热图和 checkpoint 通常体积较大，并且来源、许可和版本需要单独说明。因此仓库只保存代码和文档，数据保留在本地或单独的数据发布位置。
