"""Audit F-078: scripts/start_dev.ps1 hard-coded another machine's absolute
paths and bound the authenticated API to every interface (0.0.0.0)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_dev.ps1"


def _text():
    return SCRIPT.read_text(encoding="utf-8-sig")


def test_no_machine_specific_absolute_paths():
    assert not re.findall(r"\b[A-Za-z]:\\", _text())


def test_api_binds_to_loopback_unless_asked_otherwise():
    text = _text()
    assert re.search(r"\$BindHost\s*=\s*\"127\.0\.0\.1\"", text)
    assert "--host $BindHost" in text
    assert "--host 0.0.0.0" not in text


def test_script_parses_as_powershell():
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell is not available")
    command = ("$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile("
               f"'{SCRIPT}', [ref]$null, [ref]$errors) | Out-Null; $errors.Count")
    result = subprocess.run([shell, "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0", result.stdout
