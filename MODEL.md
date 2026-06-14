# 模型说明

当前主线是 `第九版/lake_pinn`。模型延续 reconstruction-state / state-space forecaster：以当前温度剖面状态、气象强迫、LST 信息和湖泊属性为输入，预测下一步剖面演化，并在滚动过程中约束物理一致性。第九版的重点不是新增一个单纯数值最优热图，而是围绕 zero-profile reconstruction、support profile 校正和 few-shot 迁移失败来源做系统诊断。

## 核心结构

- `state_model.py` 定义状态推进网络、forcing batch、lake static features、residual tendency 和物理缩放参数。
- `state_multilake.py` 负责多湖 manifest 训练、heldout lake 评估、segment rollout、rolling horizon、reconstruction export 和 scorecard 集成。
- `state_reconstruction.py` 提供 initial state reconstruction、support assimilation、LSWT observer update、zero-profile 初始化和 rollout state 工具。
- `vertical_solver.py` 与 `hypsometry.py` 提供垂向扩散、层厚和湖盆面积剖面处理。
- `physics.py`、`forcing.py`、`conditional_priors.py` 提供水密度、湍流通量、表层强迫修正、条件先验和 warm/deep lake 约束。
- `scripts/` 下的 pipeline controller、roadmap、tiered smoke 和 R19-R35 诊断脚本用于组织第九版 reconstruction 实验。

## 第九版新增重点

- zero-profile reconstruction：在缺少目标剖面初始化时，显式评估从先验状态进入 rollout 的误差来源。
- support profile assimilation：用少量 support profile 校正 query-start 初始状态，并检查校正是否朝观测方向移动。
- sparse observer / LSWT observer autopsy：分析表层 LST 更新是否过深、过强，或在不同湖泊类型上产生相反偏差。
- tiered experiment control：通过 L1/L2/L3/L4/L7 分层 smoke、diagnostic 和 overnight 任务减少盲目长跑。
- diagnostic-only export modes：区分 free、support_train、profile_train、support_all、profile_all 等 export 口径，避免把带观测锚定的诊断结果误当 formal transfer claim。

## 公共接口

主要入口：

```powershell
Push-Location ".\第九版"
python -m lake_pinn --manifest "..\path\to\manifest.json" --output-dir "..\outputs\v9_run"
Pop-Location
```

Python API 入口集中在 `lake_pinn.api`，CLI 入口由 `lake_pinn.__main__` 调用 `state_multilake.main()`。

## 评估逻辑

第九版评估时区分三类结果：

- main record：`RECON_L3_SUPPORT_DELTA_MAGNITUDE_OVERNIGHT_v1`，以 validation few-shot 30d/60d 和 rolling-start 30d/60d 为主。
- diagnostic consistency：1d closed-loop 与 teacher-forced transition 是否一致，support update 是否改善 query-start profile。
- diagnostic-only export：R11 export modes 和 lake-type bias/RMSE 只用于定位误差来源，不作为正式泛化主结果。
