param([int]$ApiPort = 8099)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath '.local.env')) { throw 'Run scripts/setup_local.ps1 first.' }
$taskPython = Join-Path $taskRoot '.venv-local\Scripts\python.exe'
docker compose --env-file .local.env -f compose.local.yml up -d --wait
if ($LASTEXITCODE -ne 0) { throw 'Could not start the database. Start Docker Desktop first.' }
& $taskPython (Join-Path $taskRoot 'scripts\manage_local.py') migrate
if ($LASTEXITCODE -ne 0) { throw 'Database migration failed. The local API and memory worker were not started.' }
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '2'
$taskRuntime = Join-Path $taskRoot '.local\ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $taskRuntime)) { $taskRuntime = (Get-Command ollama).Source }
try { $null = Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2 }
catch {
    $taskProcess = Start-Process -FilePath $taskRuntime -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput '.local\ollama.stdout.log' -RedirectStandardError '.local\ollama.stderr.log'
    $taskProcess.Id | Set-Content -LiteralPath '.local\ollama.pid'
}
$taskReady = $false
for ($taskAttempt = 0; $taskAttempt -lt 30; $taskAttempt++) {
    try { $null = Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 1; $taskReady = $true; break }
    catch { Start-Sleep -Seconds 1 }
}
if (-not $taskReady) { throw 'Local Ollama did not become ready.' }
$taskLogDirectory = Join-Path $taskRoot '.local'
New-Item -ItemType Directory -Path $taskLogDirectory -Force | Out-Null
$taskWorkerName = 'memory-worker-' + [guid]::NewGuid().ToString('N')
$taskWorkerStdout = Join-Path $taskLogDirectory ($taskWorkerName + '.stdout.log')
$taskWorkerStderr = Join-Path $taskLogDirectory ($taskWorkerName + '.stderr.log')
$taskWorkerScript = Join-Path $taskRoot 'scripts\memory_worker.py'
$taskMemoryWorker = $null
try {
    $taskMemoryWorker = Start-Process -FilePath $taskPython -ArgumentList @('-u', ('"' + $taskWorkerScript + '"')) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskWorkerStdout -RedirectStandardError $taskWorkerStderr
    if ($taskMemoryWorker.WaitForExit(1000)) {
        throw "Memory worker exited during startup. See $taskWorkerStderr."
    }
    Write-Output "Memory worker logs are saved to $taskWorkerStdout and $taskWorkerStderr."
    & $taskPython -m uvicorn server:app --host 127.0.0.1 --port $ApiPort
    if ($LASTEXITCODE -ne 0) { throw "Local API stopped with exit code $LASTEXITCODE." }
}
finally {
    if ($null -ne $taskMemoryWorker) {
        try {
            if (-not $taskMemoryWorker.HasExited) {
                $taskMemoryWorker.Kill()
                $taskMemoryWorker.WaitForExit()
            }
        }
        finally { $taskMemoryWorker.Dispose() }
    }
}
