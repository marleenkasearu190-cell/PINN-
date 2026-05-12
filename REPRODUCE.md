# Reproduction Guide

This repository contains code and documentation only. Raw data, trained checkpoints, prediction CSV files, and figures are intentionally kept out of Git. To reproduce a run, prepare the input data and checkpoint locally, then use the commands below.

For the author's current machine, private absolute paths are kept in `REPRODUCE_LOCAL.md`. That file is ignored by Git.

## Required Inputs

Prepare the following files:

| File | Example path | Purpose |
|---|---|---|
| ERA5 forcing CSV | `data/ERA5_daily.csv` | Daily meteorological forcing |
| MODIS LST CSV | `data/LST_2017.csv` | Lake surface temperature observation |
| PINN checkpoint | `checkpoints/mohonk_lake_2017_pinn_model_checkpoint.pt` | Trained model for predict mode |
| Profile truth CSV | `data/profile_truth.csv` | Optional truth file for scoring |

The expected data formats are described in [`DATA.md`](./DATA.md). The current recommended model line is the fifth-edition raw PINN pipeline described in [`MODEL.md`](./MODEL.md).

## Environment

Use Python 3.10 or newer.

```powershell
pip install -r requirements.txt
```

## Predict

```powershell
Push-Location ".\第五版"
python -m lake_pinn `
  --mode predict `
  --era5 "..\data\ERA5_daily.csv" `
  --lst "..\data\LST_2017.csv" `
  --model-checkpoint-path "..\checkpoints\mohonk_lake_2017_pinn_model_checkpoint.pt" `
  --output-dir "..\outputs\predict_main" `
  --device cpu
Pop-Location
```

Expected outputs:

- `mohonk_lake_2017_pinn_temperature_depth_predictions.csv`
- `mohonk_lake_2017_year_heatmap.png`
- `mohonk_lake_2017_prediction_outputs_manifest.csv`

## Score

```powershell
python ".\第五版\lake_pinn\lake_profile_scorecard.py" `
  --truth ".\data\profile_truth.csv" `
  --pred ".\outputs\predict_main\mohonk_lake_2017_pinn_temperature_depth_predictions.csv" `
  --label "main_predict" `
  --out-dir ".\outputs\score_main"
```

The score script writes `scorecard_summary.csv`, `scorecard_scores.csv`, `scorecard_vetoes.csv`, and `scorecard_diagnostics.csv`.

## Recommended Public Reporting

When reporting results, keep three things separate:

- The current research mainline: `第五版/lake_pinn/`.
- The archived fourth-edition modular reference: `归档/第四版/lake_pinn/`.
- The archived third-edition single-file reference: `归档/第三版/PPO策略调控_11维主线_20260426.py`.
- Historical numeric baselines: archived second-edition runs such as `策略测试/七`.
- Experimental physics variants: heat-budget A-line runs such as `11维测试/十六` and `11维测试/十七`.

The project position is summarized in [`EXPERIMENTS.md`](./EXPERIMENTS.md): the historical second-edition route has the lowest Mohonk 2017 RMSE, the third- and fourth-edition routes remain archived references, and the fifth-edition raw PINN package is the recommended line for continued research.
