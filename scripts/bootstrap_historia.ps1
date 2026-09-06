[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$DataDir = "",
    [string]$ModelDir = "",
    [switch]$SkipModel,
    [switch]$NoLaunch,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string]$FilePath, [string[]]$Arguments = @())

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Get-VenvPython {
    param([Parameter(Mandatory = $true)][string]$VenvDir)

    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        return Join-Path $VenvDir "Scripts\python.exe"
    }
    return Join-Path $VenvDir "bin/python"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $projectRoot ".venv"
$venvPython = Get-VenvPython -VenvDir $venvDir

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "[Historia] Creating virtual environment..."
    Invoke-Checked -FilePath $Python -Arguments @("-m", "venv", $venvDir)
}

Write-Host "[Historia] Installing local dependencies..."
Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
$extra = if ($Dev) { ".[local,dev]" } else { ".[local]" }
Push-Location $projectRoot
try {
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "-e", $extra)
}
finally {
    Pop-Location
}

if ([string]::IsNullOrWhiteSpace($DataDir)) {
    if (Test-Path -LiteralPath "D:\") {
        $DataDir = "D:\HistoriaData"
    }
    else {
        $baseDataDir = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $HOME }
        $DataDir = Join-Path $baseDataDir "Historia\data"
    }
}
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
$env:BYZANTINE_DATA_DIR = [System.IO.Path]::GetFullPath($DataDir)

if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = Join-Path $projectRoot "models\bge-m3"
}
$modelConfig = Join-Path $ModelDir "config.json"
if (-not $SkipModel) {
    if (-not (Test-Path -LiteralPath $modelConfig)) {
        Write-Host "[Historia] Downloading BGE-M3 for the first time. This can take a while..."
        Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "modelscope")
        Invoke-Checked -FilePath $venvPython -Arguments @(
            "-m", "byzantine.indexing.download_model", "--output-dir", $ModelDir
        )
    }
    $env:BYZANTINE_EMBEDDING_MODEL = [System.IO.Path]::GetFullPath($ModelDir)
}

Write-Host "[Historia] Data directory: $env:BYZANTINE_DATA_DIR"
if (Test-Path -LiteralPath $modelConfig) {
    Write-Host "[Historia] BGE-M3: $([System.IO.Path]::GetFullPath($ModelDir))"
}
elseif ($SkipModel) {
    Write-Host "[Historia] Model download skipped. You can download it later with byzantine-download-model."
}

if ($NoLaunch) {
    Write-Host "[Historia] Environment is ready. Page launch skipped."
    exit 0
}

$url = "http://127.0.0.1:8501"
$listener = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Host "[Historia] An app is already listening at $url. Opening it..."
    Start-Process $url
    exit 0
}

$appPath = Join-Path $projectRoot "src\byzantine\app.py"
Write-Host "[Historia] Starting local app..."
$appProcess = Start-Process -FilePath $venvPython `
    -ArgumentList @("-m", "streamlit", "run", $appPath, "--server.address", "127.0.0.1", "--server.port", "8501") `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru

$ready = $false
for ($attempt = 1; $attempt -le 20; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $ready = $true
            break
        }
    }
    catch {
        if ($appProcess.HasExited) {
            throw "Historia failed to start (exit code $($appProcess.ExitCode))."
        }
    }
}

if (-not $ready) {
    throw "Historia did not become ready at $url within 10 seconds."
}

Write-Host "[Historia] Ready at $url"
Start-Process $url
