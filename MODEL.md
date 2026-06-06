# 模型说明

当前主线是 `第八版/lake_pinn`。模型不再把所有任务都视为静态函数拟合 `T(z,t)`，而是采用 reconstruction-state / state-space forecaster：以当前温度剖面状态、气象强迫、LST 信息和湖泊属性为输入，预测下一步剖面演化，并在滚动过程中约束物理一致性。

## 核心结构

- `state_model.py` 定义状态推进网络、forcing batch、lake static features、residual tendency 和物理缩放参数。
- `state_multilake.py` 负责多湖 manifest 训练、heldout lake 评估、segment rollout、rolling horizon、导出和 scorecard 集成。
- `vertical_solver.py` 与 `hypsometry.py` 提供垂向扩散、层厚和湖盆面积剖面处理。
- `physics.py`、`forcing.py`、`conditional_priors.py` 提供水密度、湍流通量、表层强迫修正、条件先验和 warm/deep lake 约束。
- `standard_inputs.py` 与 `scripts/prepare_v8_global_generalization_inputs.py` 用于标准输入和第八版跨湖泛化数据准备。

## 第八版新增重点

- extended metadata：扩展湖泊静态属性，支持更多 lake-year 的跨湖泛化。
- temporal adaptive：让部分自适应参数随时间和状态变化，而不是只依赖固定湖泊属性。
- LST dropout 与 segment LST weak loss：降低 LST 依赖，同时保留表层观测对滚动段的弱约束。
- surface/ice latent reservoir：处理表层和结冰期热量滞留。
- warm-column heat-content loss：约束暖季水柱热含量，减少夏秋季系统漂移。
- roll60/export25：训练和导出更关注长时段滚动稳定性，并统一 0-25 m 公开评估口径。
- PGDL-WRR benchmark：提供无 LST 条件下与官方 PGDL-WRR 2019 Mendota 结果对比的工具脚本。

## 公共接口

主要入口：

```powershell
Push-Location ".\第八版"
python -m lake_pinn --manifest "..\path\to\manifest.json" --output-dir "..\outputs\v8_run"
Pop-Location
```

Python API 入口集中在 `lake_pinn.api`，CLI 入口由 `lake_pinn.__main__` 调用 `state_multilake.main()`。

## 评估逻辑

第八版不只看单步 transition RMSE。公开结果优先关注：

- heldout lake-year 的 observed-point RMSE 和 bias。
- 年度热图是否保持合理季节结构。
- bias contour 是否显示系统性冷偏或热偏。
- scorecard 中的分层误差、季节误差和观测点匹配结果。
- 长时段 rollout 是否出现漂移、翻混异常或不合理密度结构。
