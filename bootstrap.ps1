param(
    [string]$RunCommand = "python -m app.main"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[setup] Creating virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    }
    else {
        python -m venv .venv
    }
}

Write-Host "[setup] Installing dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "[run] Starting project..."
$runParts = $RunCommand -split " " | Where-Object { $_ -ne "" }
if ($runParts.Length -eq 0) {
    throw "RunCommand is empty."
}

if ($runParts[0].ToLower() -eq "python") {
    $args = @()
    if ($runParts.Length -gt 1) {
        $args = $runParts[1..($runParts.Length - 1)]
    }
    & $venvPython @args
}
else {
    $args = @()
    if ($runParts.Length -gt 1) {
        $args = $runParts[1..($runParts.Length - 1)]
    }
    & $runParts[0] @args
}
