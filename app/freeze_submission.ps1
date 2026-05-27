$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = $scriptDir
$srcRoot = Join-Path $appRoot "code\src"
$pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

Write-Host "[freeze_submission] compare config consistency before freeze..."
& $pythonBin (Join-Path $srcRoot "compare_config_consistency.py")

& $pythonBin (Join-Path $srcRoot "cli.py") --app-root $appRoot freeze @args
