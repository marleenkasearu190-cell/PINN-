param(
    [switch]$QuickTest,
    [int]$Epochs = 2500,
    [string]$MergedFile = ''
)

$ErrorActionPreference = 'Stop'

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = 'E:\pycharm\.venv\Scripts\python.exe'
$DownloadExtractScript = Join-Path $BaseDir '下载提取一体.py'
$ModelScript = Join-Path $BaseDir '物理参数.py'
$DailyCsv = Join-Path $BaseDir 'ERA5_mendota_2018_Daily.csv'
$LstCsv = Join-Path $BaseDir 'Lake-Mendota-MOD11A1-061-results.csv'
$DefaultMergedNc = Join-Path $BaseDir 'ERA5_mendota_2018_full.nc'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $DownloadExtractScript)) {
    throw "Script not found: $DownloadExtractScript"
}
if (-not (Test-Path -LiteralPath $ModelScript)) {
    throw "Script not found: $ModelScript"
}
if (-not (Test-Path -LiteralPath $LstCsv)) {
    throw "LST file not found: $LstCsv"
}

if ($QuickTest) {
    $Epochs = 5
}

$ResolvedMergedNc = $null
if ($MergedFile) {
    $ResolvedMergedNc = (Resolve-Path -LiteralPath $MergedFile).Path
}
elseif (Test-Path -LiteralPath $DefaultMergedNc) {
    $ResolvedMergedNc = $DefaultMergedNc
}

if (Test-Path -LiteralPath $DailyCsv) {
    Write-Host "Detected existing ERA5 daily CSV, skipping download/extract." -ForegroundColor Cyan
}
elseif ($ResolvedMergedNc) {
    Write-Host "Detected merged NC, extracting CSV files first." -ForegroundColor Cyan
    & $PythonExe $DownloadExtractScript --lake mendota --year 2018 --merged-file $ResolvedMergedNc
    if ($LASTEXITCODE -ne 0) { throw 'Download/extract step failed.' }
}
else {
    Write-Host "No daily CSV or merged NC found. Starting full ERA5 download/extract workflow." -ForegroundColor Yellow
    & $PythonExe $DownloadExtractScript --lake mendota --year 2018
    if ($LASTEXITCODE -ne 0) { throw 'Download/extract step failed.' }
}

if (-not (Test-Path -LiteralPath $DailyCsv)) {
    throw "Daily CSV not found after preparation: $DailyCsv"
}

Write-Host "Starting PINN model run..." -ForegroundColor Green
& $PythonExe $ModelScript --era5 $DailyCsv --lst $LstCsv --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Model run failed.' }

Write-Host "Pipeline completed successfully." -ForegroundColor Green
