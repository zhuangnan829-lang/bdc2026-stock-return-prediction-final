$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = $scriptDir
$srcRoot = Join-Path $appRoot "code\src"
$pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

& $pythonBin (Join-Path $srcRoot "cli.py") --app-root $appRoot train @args
