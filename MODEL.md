# 模型说明

第七版路线将主线切换为 reconstruction-state / state-space forecaster：模型不再只学习独立的 `T(z,t)` 函数，而是从当前温度剖面状态、forcing 历史和湖泊静态属性推进下一步剖面。物理约束重点转为热含量转移、垂向扩散、bulk turbulent flux、密度稳定、free-roll 漂移和 segment rollout 稳定性。

## 输入与输出

第七版默认通过 manifest 读取 lake-year 输入。每个 lake-year 至少需要 ERA5 forcing、LST、profile observations 和 metadata；模型内部会构建 forcing history、静态湖泊属性、深度网格、hypsometry 和当前剖面状态。归档第五版/第六版的直接 PINN 输入维度不再是第七版主接口。

| 输入 | 含义 |
|---|---|
| Manifest | lake-year 列表、heldout lake 设置、路径和训练参数 |
| ERA5 / forcing | 气温、风、短波、长波或可派生 flux 项 |
| LST | 表层观测、初始化和可选表层同化 |
| Profile observations | transition loss、state initialization、validation 和 scorecard |
| Metadata / hypsometry | 最大水深、面积、纬度、体积、透明度等静态属性 |

输出为：

| 输出 | 含义 |
|---|---|
| `T_next(z)` | 从当前剖面状态推进得到的下一步水温剖面，单位摄氏度 |

## PINN 主体

`LakePINN` 是一个多层感知机。网络不直接输出整张热图，而是学习函数：

```text
T = f(t_norm, z_norm, forcing, lake_metadata)
```

预测年度热图时，脚本会在每日时间网格和深度网格上调用模型，得到完整温度剖面序列。

第六版归档版本新增 `GlobalAdaptiveLakePINN`：

```text
T = global_backbone(t, z, forcing, lake_metadata) + lake_adapter(lake_metadata, shared_state)
```

其中 global backbone 学跨湖共享结构，lake adapter / residual 学湖泊属性驱动的差异。few-shot 模块可以冻结 global checkpoint，只用少量目标湖剖面日期训练 residual adapter。

第七版新增 `LakeStateForecaster`：

```text
T(t + dt, z) = M(T(t, z), forcing[t:t+dt], lake_metadata, hypsometry)
```

其中 state model 负责预测剖面增量和物理尺度，vertical solver 负责垂向扩散和面积加权热源，reconstruction 模块负责初始化、spinup、LST 同化和 free-roll 导出。

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
| Density / Richardson diffusivity | 根据密度梯度和 Richardson 数调节垂向扩散，稳定分层降低混合，密度倒置增强混合 |
| Warm/deep lake constraints | 针对 Kinneret 等暖深湖约束冬季弱梯度、Jan deep memory 和深层慢变化 |
| Heat-content transition | 约束相邻剖面之间的整柱热含量变化 |
| Bulk turbulent flux | 用 bulk formula 或已提供 flux 驱动表面热通量 |
| Segment rollout / free-roll | 用多日滚动误差约束状态推进稳定性 |
| Latent reservoir freezing | 用 latent heat reservoir 处理冰点附近能量滞留 |

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

`第七版/lake_pinn/` 是当前推荐继续研究的模块化版本。它承接第六版多湖输入和评分资产，但将核心模型切换为 reconstruction-state forecaster，重点优化长时段 free-roll 稳定性。

`归档/第六版/lake_pinn/` 是 multi-lake / few-shot 归档基线。它承接第五版 raw PINN 包结构，新增多湖 global adapter、few-shot 适配和 warm/deep lake 物理约束。

`归档/第五版/lake_pinn/` 是已归档的 Mohonk raw PINN 基线。它承接第四版包结构，把 raw PINN 作为默认预测输出，并将更多物理形态约束前移到训练阶段。

`归档/第四版/lake_pinn/` 是第四版模块化对照。它承接 run9 后续实验，把训练、预测、Kalman 同化、PPO 调度和评分工具拆开维护。

`归档/第三版/PPO策略调控_11维主线_20260426.py` 是已归档的 11 维单文件主线。它不是所有实验中 RMSE 最低的版本，但在 11 维输入、物理形态和季节过程之间更平衡，适合作为稳定对照。

`归档/第三版/PPO策略调控_热收支A线_20260428.py` 是已归档实验线。A3/A4 说明能量版热收支有潜力，但预测阶段仍需继续检查 PPO / Kalman / heat-budget 的耦合。
