# 模型说明

第五版路线延续前序版本的条件化 Lake PINN，并将研究重心调整为 raw PINN + 训练侧结构约束：神经网络输出湖泊温度剖面 `T(z,t)`，物理损失负责约束热扩散、表面能量平衡、密度稳定、整剖面形态和边界条件。Kalman、rolling 和 PPO 仍可用于对照或诊断，但不再是默认预测主线。

## 输入与输出

第五版默认面向 27 维扩展输入，覆盖 forcing、past-only weather memory 和 previous-state memory。复用归档第三版 run9/11 维 checkpoint 或做兼容训练时，需要显式设置 `--model-input-dim 11`。下表是 11 维兼容输入的核心字段。

| 输入 | 含义 |
|---|---|
| `t_norm` | 归一化时间 |
| `z_norm` | 归一化深度 |
| `doy_sin` | 年周期正弦项 |
| `doy_cos` | 年周期余弦项 |
| `LST_surface` | 湖泊表面温度观测或表层目标 |
| `T_air` | 近地面气温 |
| `wind_speed` | 风速 |
| `shortwave` | 短波辐射 |
| `max_depth_norm` | 归一化最大水深 |
| `log_area` | 湖泊面积的对数特征 |
| `latitude` | 纬度特征 |

输出为：

| 输出 | 含义 |
|---|---|
| `T(z,t)` | 指定时间和深度处的水温，单位摄氏度 |

## PINN 主体

`LakePINN` 是一个多层感知机。网络不直接输出整张热图，而是学习函数：

```text
T = f(t_norm, z_norm, forcing, lake_metadata)
```

预测年度热图时，脚本会在每日时间网格和深度网格上调用模型，得到完整温度剖面序列。

## 物理约束

LakePINN 脚本中的主要物理项包括：

| 物理项 | 作用 |
|---|---|
| PDE residual | 约束一维垂向热扩散和短波穿透加热 |
| Surface boundary condition | 用表面能量平衡约束湖面热通量 |
| Bottom boundary condition | 底部近似零通量或弱通量条件 |
| Initial condition | 约束初始温度剖面 |
| Observation loss | 约束模型贴近剖面观测或表层观测 |
| Smoothness / structure terms | 抑制不合理振荡，辅助维持季节结构 |

热收支 A 线进一步实验了能量版整柱热收支：

```text
d(heat content)/dt ≈ surface energy flux + penetrating shortwave
```

该实验线有价值，但目前作为归档对照保留。

## Kalman 同化

Kalman 部分用于预测和在线修正阶段。它不是替代 PINN，而是在 PINN 给出的剖面预测基础上融合观测信息。

可同化数据包括：

- `assim` 剖面观测。
- `SurfaceBulkTarget_C` 或 `LST_surface_C` 表层观测链路。
- 可选 `BottomTemp_C` 底温信息。

当前 Kalman 参数包括：

| 参数 | 含义 |
|---|---|
| `process` | 过程噪声缩放 |
| `obs` | 观测噪声缩放 |
| `correlation_length` | 垂向相关长度 |
| `forecast_blend` | forecast 与同化更新之间的混合比例 |

## PPO 调度

PPO 不直接预测温度剖面。温度剖面由 PINN 输出，PPO 的角色是学习调度策略：

- 动态调节 PINN 损失权重。
- 在允许时调节 Kalman 参数。
- 根据物理诊断量、验证误差和代理指标改进训练/预测流程。

因此更准确的说法是：

```text
PINN 学 T(z,t)，PPO 学怎样调度物理约束和同化参数。
```

## 当前主线判断

`第五版/lake_pinn/` 是当前推荐继续研究的模块化版本。它承接第四版包结构，把 raw PINN 作为默认预测输出，并将更多物理形态约束前移到训练阶段。

`归档/第四版/lake_pinn/` 是第四版模块化对照。它承接 run9 后续实验，把训练、预测、Kalman 同化、PPO 调度和评分工具拆开维护。

`归档/第三版/PPO策略调控_11维主线_20260426.py` 是已归档的 11 维单文件主线。它不是所有实验中 RMSE 最低的版本，但在 11 维输入、物理形态和季节过程之间更平衡，适合作为稳定对照。

`归档/第三版/PPO策略调控_热收支A线_20260428.py` 是已归档实验线。A3/A4 说明能量版热收支有潜力，但预测阶段仍需继续检查 PPO / Kalman / heat-budget 的耦合。
