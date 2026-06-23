# 模型说明

当前主线是 `第十版/lake_pinn`。模型延续 reconstruction-state / state-space forecaster：先构造或重建当前温度剖面状态，再结合气象强迫、LST 信息和湖泊静态属性预测后续剖面演化，并在训练和诊断中加入物理一致性约束。

## 核心结构

- `state_model.py` 定义状态推进网络、forcing batch、lake static features、residual tendency 和物理缩放参数。
- `state_multilake.py` 负责多湖 manifest 训练、group-kfold split、heldout diagnostic、rolling metrics、zero-profile export validation 和 CLI。
- `state_reconstruction.py` 提供 initial state reconstruction、support assimilation、zero-profile prior、EOF/PCA basis 和 LSWT observer 工具。
- `unlabeled_heat_closure.py` 构造无剖面日期 heat-closure windows。
- `vertical_solver.py`、`hypsometry.py`、`physics.py`、`forcing.py` 和 `conditional_priors.py` 提供垂向扩散、湖盆面积、密度、通量和条件先验支持。

## 第十版新增重点

- zero-profile EOF/PCA init-net：从 train-only thermal basis 构造低秩初始剖面，并允许小幅 train-supervised correction。
- daily-memory 分支：学习逐日低秩 profile memory，用于和 `init_physics_rollout` 主线比较。
- model-mainline 解析：显式区分 `init_physics_rollout` 与 `daily_memory`，减少隐式分支配置造成的误读。
- unlabeled heat-closure：尝试在无剖面日期使用外部热收支约束训练信号。
- GPU batch autotune 和 target matrix cache：提升多湖训练的批量选择和重复计算稳定性。
- source package hygiene：用 allow-list 导出可公开源码包，避免夹带本地实验输出。

## 公共接口

主要 CLI：

```powershell
Push-Location ".\第十版"
python -m lake_pinn --manifest "..\path\to\manifest.json" --output-dir "..\outputs\v10_run"
Pop-Location
```

Python API 入口集中在 `lake_pinn.api`；`lake_pinn.__main__` 调用 `state_multilake.main()`。

## 评估逻辑

第十版评估时区分三类结果：

- main diagnostic record：R42 group-kfold epoch-2 diagnostic，用于验证工程主线和 split 诊断。
- transition / rolling / zero-profile diagnostics：用于定位 initial state、daily memory 和 physics rollout 的误差来源。
- heat-closure diagnostics：用于判断无剖面物理约束是否形成有效训练信号。

当前 R42 是 diagnostic-only，不替代第八版 R9 的跨湖 heldout 长时段结论。
