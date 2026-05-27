$ErrorActionPreference = "Stop"

Write-Host "[init.ps1] starting environment checks..."

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = $scriptDir
$codeRoot = Join-Path $appRoot "code"
$srcRoot = Join-Path $codeRoot "src"
$modelDir = Join-Path $appRoot "model"
$dataDir = Join-Path $appRoot "data"
$outputDir = Join-Path $appRoot "output"
$tempDir = Join-Path $appRoot "temp"
$pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

foreach ($dir in @($codeRoot, $srcRoot, $modelDir, $dataDir, $outputDir, $tempDir)) {
    if (-not (Test-Path -LiteralPath $dir)) {
        Write-Host "[init.ps1] creating missing directory: $dir"
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

try {
    $null = & $pythonBin --version
} catch {
    throw "[init.ps1] error: $pythonBin is not available"
}

Write-Host "[init.ps1] app root: $appRoot"
Write-Host "[init.ps1] python version: $(& $pythonBin --version 2>&1)"
Write-Host "[init.ps1] code root: $codeRoot"
Write-Host "[init.ps1] data dir: $dataDir"
Write-Host "[init.ps1] output dir: $outputDir"
Write-Host "[init.ps1] temp dir: $tempDir"

if (-not (Test-Path -LiteralPath (Join-Path $dataDir "train.csv"))) {
    Write-Warning "[init.ps1] $dataDir\train.csv not found yet"
}

if (-not (Test-Path -LiteralPath (Join-Path $dataDir "test.csv"))) {
    Write-Warning "[init.ps1] $dataDir\test.csv not found yet"
}

Write-Host "[init.ps1] environment checks completed."
