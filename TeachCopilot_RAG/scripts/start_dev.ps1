$ErrorActionPreference = "Stop"

Write-Host "=== TeachCopilot dev start ===" -ForegroundColor Cyan

Write-Host "Starting Docker Compose: Open WebUI + PostgreSQL/pgvector..." -ForegroundColor Cyan
cd C:\Users\DataScientist\openwebui
docker compose -f docker-compose.open-webui.yml up -d

Write-Host ""
Write-Host "Docker containers:" -ForegroundColor Cyan
docker ps

Write-Host ""
Write-Host "Starting TeachCopilot RAG API..." -ForegroundColor Cyan
cd E:\PycharmProjects\TeachCopilot_RAG

$env:PYTHONUTF8 = "1"

uv run uvicorn server:app --host 0.0.0.0 --port 8099