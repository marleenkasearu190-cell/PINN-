# Lake PINN Workflow

A small workflow for lake temperature modeling with ERA5, MODIS LST, and a physics-informed neural network (PINN).

## What This Project Does

This project helps you:

- download ERA5 lake-related forcing data for a target lake and year
- merge monthly ERA5 files into a yearly NetCDF file
- extract cleaned hourly and daily CSV files from ERA5
- combine ERA5 and MODIS LST data for PINN training
- generate annual temperature-depth heatmaps and prediction tables

## Main Scripts

- `下载提取一体.py`
  - one-stop ERA5 downloader + merger + CSV extractor
- `物理参数.py`
  - PINN training script using ERA5 daily forcing and MODIS LST
- `运行Mendota流程.ps1`
  - convenience entry point for the Mendota 2018 workflow
- `提取.py`
  - standalone extractor for an existing yearly ERA5 NetCDF file
- `数据爬取.py`
  - downloader / merger script kept for separate use

## Recommended Workflow

```text
ERA5 download -> yearly .nc merge -> hourly/daily CSV extraction -> LST preparation -> PINN training -> heatmap output
```

## Installation

Create a Python environment and install dependencies:

```powershell
pip install -r requirements.txt
```

You also need Copernicus CDS access configured for `cdsapi`.

## Example Commands

### 1. Download and extract Mendota 2018 ERA5

```powershell
python 下载提取一体.py --lake mendota --year 2018
```

### 2. Extract from an existing yearly NetCDF file

```powershell
python 下载提取一体.py --lake mendota --year 2018 --merged-file "E:\path\to\ERA5_mendota_2018_full.nc"
```

### 3. Train the PINN model

```powershell
python 物理参数.py --era5 "E:\path\to\ERA5_mendota_2018_Daily.csv" --lst "E:\path\to\Lake-Mendota-MOD11A1-061-results.csv"
```

### 4. Quick test run for Mendota

```powershell
powershell -ExecutionPolicy Bypass -File .\运行Mendota流程.ps1 -QuickTest
```

## Output Files

Typical generated outputs include:

- yearly ERA5 NetCDF files
- hourly ERA5 CSV files
- daily ERA5 CSV files
- yearly lake temperature-depth heatmaps
- monthly heatmaps
- full-year prediction tables
- July prediction tables

These generated files are ignored by Git via `.gitignore`.

## Notes

- Raw ERA5 and LST data are not committed in this repository.
- The current workflow includes built-in presets for `mendota` and `acton`.
- For a new lake, you can use `--bbox north west south east`.
- Some scripts use Chinese filenames; GitHub supports this, but you can add English aliases later if desired.

## Suggested Repository Structure

```text
运行代码/
  下载提取一体.py
  物理参数.py
  提取.py
  数据爬取.py
  运行Mendota流程.ps1
  requirements.txt
  README.md
  .gitignore
```

## Publish To GitHub

This machine currently does not have GitHub CLI installed, so the project is prepared for publishing locally.

Suggested steps:

1. Create a new empty repository on GitHub.
2. In this folder, run:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

## License

Add a license before public release if you want others to reuse the code.
