param([switch]$DownloadRuntime, [switch]$SkipModels)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
New-Item -ItemType Directory -Path '.local' -Force | Out-Null
if (-not (Test-Path -LiteralPath '.local.env')) {
    $taskPassword = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
    @("POSTGRES_PASSWORD=$taskPassword", "DATABASE_URL=postgresql://teachcopilot:${taskPassword}@127.0.0.1:5434/teachcopilot_local", 'OLLAMA_URL=http://127.0.0.1:11435', 'LOCAL_CHAT_MODEL=gemma4:12b', 'LOCAL_EMBEDDING_MODEL=bge-m3', 'LOCAL_CONTEXT_TOKENS=8192', 'OLLAMA_NO_CLOUD=1') | Set-Content -LiteralPath '.local.env' -Encoding utf8
}
if (-not (Test-Path -LiteralPath '.venv-local\Scripts\python.exe')) { uv venv --python 3.12 .venv-local }
uv pip install --python '.venv-local\Scripts\python.exe' -r requirements-local.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
docker compose --env-file .local.env -f compose.local.yml up -d --wait
if ($LASTEXITCODE -ne 0) { throw 'Database startup failed. Start Docker Desktop first.' }
& '.venv-local\Scripts\python.exe' scripts/manage_local.py migrate
if ($LASTEXITCODE -ne 0) { throw 'Database migration failed.' }
& '.venv-local\Scripts\python.exe' scripts/manage_local.py import
if ($LASTEXITCODE -ne 0) { throw 'Data import failed.' }
if ($DownloadRuntime) {
    $taskArchive = Join-Path $taskRoot '.local\ollama-windows-amd64.zip'
    curl.exe -L --fail --silent --show-error 'https://github.com/ollama/ollama/releases/download/v0.40.3/ollama-windows-amd64.zip' -o $taskArchive
    if ($LASTEXITCODE -ne 0) { throw 'Runtime download failed.' }
    Expand-Archive -LiteralPath $taskArchive -DestinationPath '.local\ollama' -Force
}
if (-not $SkipModels) {
    $env:OLLAMA_HOST = '127.0.0.1:11435'
    $env:OLLAMA_NO_CLOUD = '1'
    $env:OLLAMA_CONTEXT_LENGTH = '8192'
    $taskRuntime = Join-Path $taskRoot '.local\ollama\ollama.exe'
    if (-not (Test-Path -LiteralPath $taskRuntime)) { $taskRuntime = (Get-Command ollama).Source }
    try { $null = Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2 }
    catch { Start-Process -FilePath $taskRuntime -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput '.local\ollama.stdout.log' -RedirectStandardError '.local\ollama.stderr.log' }
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 30; $taskAttempt++) {
        try { $null = Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 1; $taskReady = $true; break } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $taskReady) { throw 'Local Ollama did not start.' }
    function Invoke-LocalModelPull([string]$Model, [string]$Label) {
        New-Item -ItemType Directory -Path '.local\ollama' -Force | Out-Null
        $taskModelKey = $Model -replace '[^a-zA-Z0-9._-]', '_'
        $taskStdout = Join-Path $taskRoot ".local\ollama.pull.$taskModelKey.stdout.log"
        $taskStderr = Join-Path $taskRoot ".local\ollama.pull.$taskModelKey.stderr.log"
        $taskPull = Start-Process -FilePath $taskRuntime -ArgumentList @('pull', $Model) -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr
        if ($taskPull.ExitCode -ne 0) {
            $taskErrorText = if (Test-Path -LiteralPath $taskStderr) { Get-Content -LiteralPath $taskStderr -Raw } else { '' }
            throw "$Label download failed (exit $($taskPull.ExitCode)). See $taskStderr. $taskErrorText"
        }
    }
    Invoke-LocalModelPull 'bge-m3' 'Embedding model'
    Invoke-LocalModelPull 'gemma4:12b' 'Chat model'
    & '.venv-local\Scripts\python.exe' scripts/index_local.py
    if ($LASTEXITCODE -ne 0) { throw 'Index construction failed.' }
}
Write-Output 'Local database is prepared. Create a teacher with manage_local.py bootstrap, then run start_local.ps1.'
