# 归档说明

本目录保存历史版本代码和文档，用于回溯、对照和复现实验结论。当前开发入口在仓库根目录的 `第十版/lake_pinn`。

| 版本 | 定位 | 归档原因 |
|---|---|---|
| 第九版 | reconstruction diagnostic / support transfer 主线，代表记录为 L3 overnight few-shot 30d RMSE 2.486、60d RMSE 2.396 | 已被第十版 zero-profile reconstruction-state 诊断主线替代 |
| 第八版 | 跨湖泛化主线，代表结果为 R9 epoch0099 三湖 heldout 平均 RMSE 3.856 | 作为跨湖泛化 R9 warm-column 基线保留 |
| 第七版 | reconstruction-state / state-space forecaster 前一代主线，代表结果为 Mendota T5 full free-roll RMSE 1.190 | 已被第八版跨湖泛化主线替代 |
| 第六版 | multi-lake / few-shot 迁移基线，包含 global adapter、lake-attribute residual、few-shot adapter 和 warm/deep lake 物理约束 | 作为第七版和第八版之前的多湖迁移对照 |
| 第五版 | Mohonk raw PINN 单湖基线，包含扩展输入、profile-grid physics、density regularization 和 scorecard v2 | 作为 raw PINN 单湖基线 |
| 第四版 | 模块化 LakePINN 对照，包含 PINN、PPO、Kalman、rolling 预测和分层评分流程 | 作为第五版之前的模块化对照 |
| 第三版 | 11 维 PINN + PPO/Kalman 单文件主线、热收支 A 线和剖面评分工具 | 作为稳定单文件对照 |
| 第二版 | 旧输入结构 PPO 版本，对应历史数值最优 `策略测试/七` | 作为旧输入结构数值基线 |
| 第一版 | 早期数据处理和预测流程 | 历史留档 |
| 第零版 | 最早期集中式脚本和数据处理尝试 | 历史留档 |

归档目录仍遵守仓库文件策略：不提交完整实验输出、checkpoint、CSV、外部数据、日志和缓存。
