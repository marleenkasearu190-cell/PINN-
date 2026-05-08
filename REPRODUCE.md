# 本地复现说明

本仓库只提交代码和文档，不提交原始数据、训练 checkpoint 和大批量实验输出。当前本机 `E:\pycharm\PINN` 里已经保留了可直接用于预测和评分的 Mohonk 2017 资产。

## 推荐复现路线

优先使用第三版 11 维主线：

```text
E:\pycharm\PINN\第三版\11维测试\9\mohonk_lake_2017_pinn_model_checkpoint.pt
```

该 checkpoint 属于当前 11 维 PINN + PPO/Kalman 主线，模型输入维度为 11，并且 checkpoint 内包含嵌入的 `ppo_policy_bundle`。它对应文档中的 `11维测试/九`。

## 本机资产清单

| 资产 | 本机路径 | 用途 |
|---|---|---|
| ERA5 forcing | `E:\pycharm\PINN\数据\mohonk\ERA5_mohonk_2017_Daily.csv` | 预测输入 |
| MODIS LST | `E:\pycharm\PINN\数据\mohonk\Mohonk-lst-2017-.csv` | 预测输入 |
| profile truth | `E:\pycharm\PINN\数据\mohonk\验证\MohonkLake_temp_2017_filled_from_2014_2017.csv` | 评分真值 |
| 11 维主线 checkpoint | `E:\pycharm\PINN\第三版\11维测试\9\mohonk_lake_2017_pinn_model_checkpoint.pt` | 推荐复现 |
| 热收支 A3 checkpoint | `E:\pycharm\PINN\第三版\11维测试\16\mohonk_lake_2017_pinn_model_checkpoint.pt` | 热收支实验对照 |
| 热收支 A4 checkpoint | `E:\pycharm\PINN\第三版\11维测试\17\mohonk_lake_2017_pinn_model_checkpoint.pt` | 热收支实验对照 |
| 第二版旧输入 checkpoint | `E:\pycharm\PINN\策略测试\7\mohonk_lake_2017_pinn_model_checkpoint.pt` | 数值最优历史对照 |

`策略测试/7` 的 RMSE 最低，但它属于旧输入结构，不作为当前主线。当前继续研究和对外说明时，建议以 `第三版/11维测试/9` 为主线，以 `策略测试/7` 为历史对照。

## 预测命令

在任意工作目录运行：

```powershell
python "E:\pycharm\PINN\github仓库\PINN-\第三版\PPO策略调控_11维主线_20260426.py" `
  --mode predict `
  --era5 "E:\pycharm\PINN\数据\mohonk\ERA5_mohonk_2017_Daily.csv" `
  --lst "E:\pycharm\PINN\数据\mohonk\Mohonk-lst-2017-.csv" `
  --model-checkpoint-path "E:\pycharm\PINN\第三版\11维测试\9\mohonk_lake_2017_pinn_model_checkpoint.pt" `
  --output-dir "E:\pycharm\PINN\本地复现\predict_11d9" `
  --device cpu
```

运行后会生成：

- `mohonk_lake_2017_pinn_temperature_depth_predictions.csv`
- `mohonk_lake_2017_year_heatmap.png`
- `mohonk_lake_2017_monthly_heatmaps.png`

## 评分命令

```powershell
python "E:\pycharm\PINN\github仓库\PINN-\第三版\lake_profile_scorecard.py" `
  --truth "E:\pycharm\PINN\数据\mohonk\验证\MohonkLake_temp_2017_filled_from_2014_2017.csv" `
  --pred "E:\pycharm\PINN\本地复现\predict_11d9\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" `
  --label "11d9_checkpoint_predict" `
  --out-dir "E:\pycharm\PINN\本地复现\score_11d9"
```

如果只想复查已有预测结果，也可以直接评分 `第三版/11维测试/九` 中已保存的预测 CSV：

```powershell
python "E:\pycharm\PINN\github仓库\PINN-\第三版\lake_profile_scorecard.py" `
  --truth "E:\pycharm\PINN\数据\mohonk\验证\MohonkLake_temp_2017_filled_from_2014_2017.csv" `
  --pred "E:\pycharm\PINN\第三版\11维测试\九\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" `
  --label "11d9_saved_prediction" `
  --out-dir "E:\pycharm\PINN\本地复现\score_11d9_saved"
```

## 已验证的 smoke test

已用 `11维测试/9` checkpoint、Mohonk 2017 ERA5 和原始 LST 在 CPU 上跑通过预测流程。临时验证输出位于：

```text
C:\Users\A\Documents\Playground\PINN_checkpoint_smoke_9
```

该 smoke test 仅用于确认 checkpoint 能被当前第三版脚本加载并生成预测表与热图，不建议提交到 GitHub。
