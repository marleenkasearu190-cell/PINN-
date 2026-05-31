param(
    [string]$RepoRoot = "",
    [string]$DataRoot = "",
    [string]$OutputZip = ""
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
if (-not $DataRoot) {
    $repoParent = Split-Path -Parent $RepoRoot
    $dataCandidates = Get-ChildItem -LiteralPath $repoParent -Directory |
        ForEach-Object { Join-Path $_.FullName "_standard_inputs" } |
        Where-Object { Test-Path -LiteralPath $_ }
    if (-not $dataCandidates) {
        throw "Could not find a sibling _standard_inputs directory. Pass -DataRoot explicitly."
    }
    $DataRoot = @($dataCandidates)[0]
}
if (-not $OutputZip) {
    $OutputZip = Join-Path $RepoRoot "lakepinn_mendota_cloud_bundle.zip"
}

$stage = Join-Path $env:TEMP ("lakepinn_cloud_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $stage | Out-Null

New-Item -ItemType Directory -Force (Join-Path $stage "LakePINN") | Out-Null
Copy-Item -Recurse -Force (Join-Path $RepoRoot "lake_pinn") (Join-Path $stage "LakePINN\lake_pinn")
New-Item -ItemType Directory -Force (Join-Path $stage "LakePINN\tests") | Out-Null
Copy-Item -Force (Join-Path $RepoRoot "tests\*.py") (Join-Path $stage "LakePINN\tests")
New-Item -ItemType Directory -Force (Join-Path $stage "LakePINN\scripts") | Out-Null
foreach ($script in @("make_cloud_manifest.py", "run_t5_cloud.sh", "package_mendota_cloud.ps1")) {
    Copy-Item -Force (Join-Path $RepoRoot "scripts\$script") (Join-Path $stage "LakePINN\scripts")
}
Copy-Item -Force (Join-Path $RepoRoot "requirements.txt") (Join-Path $stage "LakePINN\requirements.txt")
Copy-Item -Force (Join-Path $RepoRoot "CLOUD_GPU_README.md") (Join-Path $stage "LakePINN\CLOUD_GPU_README.md")

$manifestDst = Join-Path $stage "LakePINN\experiments\manifests_20260522"
New-Item -ItemType Directory -Force $manifestDst | Out-Null
Copy-Item -Force (Join-Path $RepoRoot "experiments\manifests_20260522\T1_mendota_reconstruction_night_20260522.json") $manifestDst
Copy-Item -Force (Join-Path $RepoRoot "experiments\manifests_20260522\T1_mendota_reconstruction_night_cloud.json") $manifestDst

$cloudData = Join-Path $stage "LakePINN\data\_standard_inputs"
New-Item -ItemType Directory -Force $cloudData | Out-Null
foreach ($lake in @("mendota_2018", "mendota_2019", "mendota_2020")) {
    Copy-Item -Recurse -Force (Join-Path $DataRoot $lake) (Join-Path $cloudData $lake)
}

if (Test-Path $OutputZip) {
    Remove-Item -Force $OutputZip
}
Compress-Archive -Path (Join-Path $stage "LakePINN") -DestinationPath $OutputZip
Remove-Item -Recurse -Force $stage
Write-Host "Wrote $OutputZip"
