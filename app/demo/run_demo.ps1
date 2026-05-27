$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$preferredPort = if ($args.Length -ge 1) { [int]$args[0] } else { 8501 }

function Test-PortAvailable {
    param([int]$Port)

    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

$candidatePorts = @($preferredPort, 8502, 8503, 8510, 8601)
$port = $null

foreach ($candidate in $candidatePorts) {
    if (Test-PortAvailable -Port $candidate) {
        $port = $candidate
        break
    }
}

if ($null -eq $port) {
    throw "No available port found. Please free one of 8501/8502/8503/8510/8601 and try again."
}

$url = "http://127.0.0.1:$port"
Write-Host "[run_demo.ps1] starting Streamlit demo on port $port"
Write-Host "[run_demo.ps1] url: $url"

python -m streamlit run (Join-Path $scriptDir "streamlit_app.py") --server.address 127.0.0.1 --server.port $port --server.headless true --browser.gatherUsageStats false
