$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dockerConfig = Join-Path $root ".docker-local"

if (-not (Test-Path $dockerConfig)) {
    New-Item -ItemType Directory -Path $dockerConfig | Out-Null
}

$env:DOCKER_CONFIG = $dockerConfig

Write-Host "[docker_offline_rehearsal] root=$root"
Write-Host "[docker_offline_rehearsal] DOCKER_CONFIG=$dockerConfig"

docker version | Out-Host
docker build -t bdc2026-stock $root | Out-Host
docker run --rm bdc2026-stock | Out-Host

Write-Host "[docker_offline_rehearsal] completed"
