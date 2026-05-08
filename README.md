# PINN-

湖泊温度剖面预测与物理约束 PINN / PPO 实验仓库。

当前主入口是 `第三版/`。第二版虽然在 Mohonk 2017 上取得过最低 RMSE，但它属于旧输入结构，已移入 `归档/`，仅作为历史对照。

## 快速入口

- 当前 11 维主线：[`第三版/PPO策略调控_11维主线_20260426.py`](./第三版/PPO策略调控_11维主线_20260426.py)
- 热收支实验线：[`第三版/PPO策略调控_热收支A线_20260428.py`](./第三版/PPO策略调控_热收支A线_20260428.py)
- 评分工具：[`第三版/lake_profile_scorecard.py`](./第三版/lake_profile_scorecard.py)
- 数据格式说明：[`DATA.md`](./DATA.md)
- 模型原理说明：[`MODEL.md`](./MODEL.md)
- 实验结论摘要：[`EXPERIMENTS.md`](./EXPERIMENTS.md)

## 环境

建议使用 Python 3.10 或更新版本。

```powershell
pip install -r requirements.txt
```

本仓库不包含原始数据、训练 checkpoint 和大批量实验输出。运行训练或预测前，需要按 [`DATA.md`](./DATA.md) 准备 ERA5、LST 和剖面观测数据。

## 最小运行示例

训练主线模型：

```powershell
python ".\第三版\PPO策略调控_11维主线_20260426.py" `
  --mode train `
  --era5 "path\to\ERA5_daily.csv" `
  --lst "path\to\LST_2017.csv" `
  --profile-obs "path\to\profile_observations.csv" `
  --profile-split-mode time_blocked `
  --epochs 600 `
  --output-dir "outputs\run_main"
```

使用已训练 checkpoint 预测：

```powershell
python ".\第三版\PPO策略调控_11维主线_20260426.py" `
  --mode predict `
  --era5 "path\to\ERA5_daily.csv" `
  --lst "path\to\LST_2017.csv" `
  --model-checkpoint-path "outputs\run_main\mohonk_lake_2017_pinn_model_checkpoint.pt" `
  --output-dir "outputs\predict_main"
```

评分预测结果：

```powershell
python ".\第三版\lake_profile_scorecard.py" `
  --truth "path\to\profile_truth.csv" `
  --pred "outputs\predict_main\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" `
  --label "main_predict" `
  --out-dir "outputs\score_main"
```

## 版本定位

| 目录/脚本 | 定位 | 对应实验 | 当前判断 |
|---|---|---|---|
| `第三版/PPO策略调控_11维主线_20260426.py` | 11 维 PINN + PPO/Kalman 主线 | `11维测试/九` | 当前推荐继续研究 |
| `第三版/PPO策略调控_热收支A线_20260428.py` | 能量版热收支实验线 | `11维测试/十一` 到 `11维测试/十七` | 有价值，但暂不替代主线 |
| `归档/第二版/PPO策略控制.py` | 旧输入结构 PPO | `策略测试/七` | 数值 RMSE 最低，但不作为当前主线 |
| `归档/第一版`、`归档/第零版` | 早期历史版本 | 早期流程 | 仅用于回溯 |

## 仓库结构

```text
PINN-
|-- README.md
|-- DATA.md
|-- MODEL.md
|-- EXPERIMENTS.md
|-- requirements.txt
|-- 第三版/
|   |-- PPO策略调控_11维主线_20260426.py
|   |-- PPO策略调控_热收支A线_20260428.py
|   |-- lake_profile_scorecard.py
|   `-- README.md
`-- 归档/
    |-- 第二版/
    |-- 第一版/
    `-- 第零版/
```

## 不建议提交的内容

以下内容建议保留在本地实验目录，不直接提交到 GitHub：

- 原始数据和验证数据。
- 训练 checkpoint，例如 `*.pt`、`*.pth`。
- 预测输出 CSV 和热图 PNG。
- `11维测试/`、`策略测试/` 等大批量实验结果目录。

这些内容更适合在论文、报告或发布版本中整理为摘要表、评分表和代表性图像。
