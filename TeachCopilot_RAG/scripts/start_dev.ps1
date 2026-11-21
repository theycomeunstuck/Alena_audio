param(
    # Audit F-078: the authenticated API listens on loopback by default. Pass
    # -BindHost 0.0.0.0 only when a container (for example Open WebUI in Docker)
    # must reach it from outside the host, and only on a trusted network.
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8099,
    # Directory with docker-compose.open-webui.yml; also read from OPENWEBUI_DIR.
    # When neither is set, the Docker step is skipped.
    [string]$OpenWebUiDir = $env:OPENWEBUI_DIR
)

$ErrorActionPreference = "Stop"

# Audit F-078: resolve the project from this script's location instead of a
# machine-specific absolute path.
$RagRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== TeachCopilot dev start ===" -ForegroundColor Cyan

if ($OpenWebUiDir) {
    Write-Host "Starting Docker Compose: Open WebUI + PostgreSQL/pgvector in $OpenWebUiDir..." -ForegroundColor Cyan
    Push-Location $OpenWebUiDir
    try {
        docker compose -f docker-compose.open-webui.yml up -d
    }
    finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "Docker containers:" -ForegroundColor Cyan
    docker ps
}
else {
    Write-Host "OPENWEBUI_DIR is not set - skipping Docker Compose (pass -OpenWebUiDir to start it)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting TeachCopilot RAG API on ${BindHost}:$Port..." -ForegroundColor Cyan
Set-Location $RagRoot

$env:PYTHONUTF8 = "1"

uv run uvicorn server:app --host $BindHost --port $Port
