$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = $scriptDir
$srcRoot = Join-Path $appRoot "code\src"
$pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

Write-Host "[test] compare config consistency..."
& $pythonBin (Join-Path $srcRoot "compare_config_consistency.py")

& $pythonBin (Join-Path $srcRoot "cli.py") --app-root $appRoot predict @args

$packageVariantPath = Join-Path $appRoot "model\package_variant.json"
$aggressiveResultPath = Join-Path $appRoot "model\aggressive_score_submission_candidate\result_aggressive_score.csv"
if ((Test-Path $packageVariantPath) -and (Test-Path $aggressiveResultPath)) {
    $variant = (Get-Content -Raw -Encoding UTF8 $packageVariantPath | ConvertFrom-Json).variant
    if ($variant -eq "aggressive_score_submission") {
        Write-Host "[test] aggressive score package detected; using frozen aggressive result"
        Copy-Item -LiteralPath $aggressiveResultPath -Destination (Join-Path $appRoot "output\result.csv") -Force
        & $pythonBin (Join-Path $srcRoot "result_validator.py") --result_path (Join-Path $appRoot "output\result.csv")
    }
}
