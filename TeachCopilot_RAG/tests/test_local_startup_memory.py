"""Exercise the Windows launcher with all service commands replaced locally."""
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_local.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows PowerShell launcher")

HARNESS = r"""
param([string]$Launcher, [string]$Scenario)
$ErrorActionPreference = 'Stop'
$global:TestScenario = $Scenario
$global:TestEvents = [System.Collections.Generic.List[string]]::new()
$taskErrors = $null
$taskTokens = $null
$taskAst = [System.Management.Automation.Language.Parser]::ParseFile($Launcher, [ref]$taskTokens, [ref]$taskErrors)
if ($taskErrors.Count) { throw 'Invalid launcher syntax' }
# Fail before execution if a future launcher adds an unmocked service command.
$taskAllowed = @('docker','Get-Command','Invoke-RestMethod','Join-Path','New-Item','Out-Null',
    'Set-Content','Set-Location','Split-Path','Start-Process','Start-Sleep','Test-Path','Write-Output')
foreach ($taskCommand in $taskAst.FindAll({param($node) $node -is [System.Management.Automation.Language.CommandAst]}, $true)) {
    $taskName = $taskCommand.GetCommandName()
    if ($null -ne $taskName -and $taskName -notin $taskAllowed) { throw "Unmocked command $taskName" }
    if ($null -eq $taskName -and $taskCommand.CommandElements[0].Extent.Text -ne '$taskPython') {
        throw 'Unmocked dynamic command'
    }
}
function docker {
    $global:TestEvents.Add('database')
    $global:LASTEXITCODE = if ($global:TestScenario -eq 'database_failure') { 1 } else { 0 }
}
function Test-Path { return $true }
function New-Item {}
function Invoke-RestMethod { return @{ version = 'test' } }
$global:TestRoot = Split-Path -Parent (Split-Path -Parent $Launcher)
$taskPythonPath = Join-Path $global:TestRoot '.venv-local\Scripts\python.exe'
Set-Item -LiteralPath ('Function:\' + $taskPythonPath) -Value {
    if ($args[-1] -eq 'migrate') {
        $global:TestEvents.Add('migrate')
        $global:LASTEXITCODE = if ($global:TestScenario -eq 'migration_failure') { 1 } else { 0 }
    }
    elseif ($args[0] -eq '-m' -and $args[1] -eq 'uvicorn') {
        if ($args[-1] -ne 8773) { throw 'API port was not forwarded' }
        $global:TestEvents.Add('api')
        $global:LASTEXITCODE = if ($global:TestScenario -eq 'api_failure') { 7 } else { 0 }
    }
    else { throw 'Unexpected Python invocation' }
}
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle, [switch]$PassThru,
          $RedirectStandardOutput, $RedirectStandardError)
    if ($FilePath -ne (Join-Path $global:TestRoot '.venv-local\Scripts\python.exe') -or
        $WorkingDirectory -ne $global:TestRoot -or $WindowStyle -ne 'Hidden' -or -not $PassThru) {
        throw 'Worker process configuration is invalid'
    }
    $taskExpectedScript = '"' + (Join-Path $global:TestRoot 'scripts\memory_worker.py') + '"'
    if ($ArgumentList.Count -ne 2 -or $ArgumentList[0] -ne '-u' -or $ArgumentList[1] -ne $taskExpectedScript) {
        throw 'Worker script path is not safely quoted'
    }
    if ($RedirectStandardOutput -notmatch 'memory-worker-[a-f0-9]{32}\.stdout\.log$' -or
        $RedirectStandardError -notmatch 'memory-worker-[a-f0-9]{32}\.stderr\.log$') {
        throw 'Worker logs must have per-launch names'
    }
    $global:TestEvents.Add('worker')
    $taskProcess = [pscustomobject]@{ HasExited = ($global:TestScenario -eq 'worker_failure') }
    $taskProcess | Add-Member ScriptMethod WaitForExit { if ($args.Count) { return $this.HasExited } }
    $taskProcess | Add-Member ScriptMethod Kill { $global:TestEvents.Add('kill_owned_worker'); $this.HasExited = $true }
    $taskProcess | Add-Member ScriptMethod Dispose { $global:TestEvents.Add('dispose_owned_worker') }
    return $taskProcess
}
$taskFailed = $false
try { & $Launcher -ApiPort 8773 }
catch { $taskFailed = $true }
Write-Output ('TEST_RESULT=' + (@{ failed=$taskFailed; events=@($global:TestEvents) } | ConvertTo-Json -Compress))
"""


@pytest.mark.parametrize(
    ("scenario", "failed", "events"),
    [
        ("success", False, ["database", "migrate", "worker", "api", "kill_owned_worker", "dispose_owned_worker"]),
        ("database_failure", True, ["database"]),
        ("migration_failure", True, ["database", "migrate"]),
        ("worker_failure", True, ["database", "migrate", "worker", "dispose_owned_worker"]),
        ("api_failure", True, ["database", "migrate", "worker", "api", "kill_owned_worker", "dispose_owned_worker"]),
    ],
)
def test_launcher_orders_migration_and_cleans_up_only_its_worker(tmp_path, scenario, failed, events):
    harness = tmp_path / "launcher_harness.ps1"
    harness.write_text(HARNESS, encoding="utf-8")
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
         str(harness), "-Launcher", str(SCRIPT), "-Scenario", scenario],
        capture_output=True, text=True, timeout=20, check=True,
    )
    reports = [line.removeprefix("TEST_RESULT=") for line in result.stdout.splitlines()
               if line.startswith("TEST_RESULT=")]
    assert len(reports) == 1, result.stdout + result.stderr
    assert json.loads(reports[0]) == {"failed": failed, "events": events}
