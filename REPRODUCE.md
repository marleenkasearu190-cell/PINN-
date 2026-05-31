# Reproduction Guide

This repository contains code and documentation only. Raw data, trained checkpoints, prediction CSV files, and figures are intentionally kept out of Git. To reproduce a run, prepare the input data and checkpoint locally, then use the commands below.

For the author's current machine, private absolute paths are kept in `REPRODUCE_LOCAL.md`. That file is ignored by Git.

## Required Inputs

Prepare the following files:

| File | Example path | Purpose |
|---|---|---|
| ERA5 forcing CSV | `data/ERA5_daily.csv` | Daily meteorological forcing |
| MODIS LST CSV | `data/LST_2017.csv` | Lake surface temperature observation |
| Manifest JSON | `manifests/mendota_reconstruction.json` | Lake-year inputs and heldout settings for seventh-edition runs |
| State-forecaster checkpoint | `checkpoints/global_state_forecaster_checkpoint.pt` | Optional trained model for export-only mode |
| Profile truth CSV | `data/profile_truth.csv` | Optional truth file for scoring |

The expected data formats are described in [`DATA.md`](./DATA.md). The current recommended model line is the seventh-edition reconstruction-state pipeline described in [`MODEL.md`](./MODEL.md).

## Environment

Use Python 3.10 or newer.

```powershell
pip install -r requirements.txt
```

## Train Or Export

```powershell
Push-Location ".\第七版"
python -m lake_pinn `
  --manifest "..\manifests\mendota_reconstruction.json" `
  --output-dir "..\outputs\v7_train" `
  --epochs 200 `
  --device cpu
Pop-Location
```

Expected outputs:

- `global_state_forecaster_training_history.csv`
- `global_state_forecaster_checkpoint.pt`
- heldout `*_temperature_depth_predictions.csv`
- heldout `*_year_heatmap.png`
- heldout scorecard and diagnostic figures

## Score

```powershell
python ".\第七版\lake_pinn\lake_profile_scorecard.py" `
  --truth ".\data\profile_truth.csv" `
  --pred ".\outputs\v7_train\heldout_temperature_depth_predictions.csv" `
  --label "main_predict" `
  --out-dir ".\outputs\score_main"
```

The score script writes `scorecard_summary.csv`, `scorecard_scores.csv`, `scorecard_vetoes.csv`, and `scorecard_diagnostics.csv`.

## Recommended Public Reporting

When reporting results, keep three things separate:

- The current research mainline: `第七版/lake_pinn/`.
- The archived sixth-edition multi-lake / few-shot baseline: `归档/第六版/lake_pinn/`.
- The archived fifth-edition Mohonk raw PINN baseline: `归档/第五版/lake_pinn/`.
- The archived fourth-edition modular reference: `归档/第四版/lake_pinn/`.
- The archived third-edition single-file reference: `归档/第三版/PPO策略调控_11维主线_20260426.py`.
- Historical numeric baselines: archived second-edition runs such as `策略测试/七`.
- Experimental physics variants: heat-budget A-line runs such as `11维测试/十六` and `11维测试/十七`.

The project position is summarized in [`EXPERIMENTS.md`](./EXPERIMENTS.md): the historical second-edition route has the lowest Mohonk 2017 RMSE, the third-, fourth-, fifth-, and sixth-edition routes remain archived references, and the seventh-edition package is the recommended line for continued reconstruction-state and long free-roll research.
