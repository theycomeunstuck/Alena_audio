# Audit F-061: scripts/dev_run.py imported print_routes at module level, which
# imports app.main – so the uvicorn reload supervisor loaded the whole app (and
# its model stack) and the reloaded worker loaded it again.
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_importing_dev_run_does_not_load_the_app():
    probe = (
        "import sys; sys.path.insert(0, r'%s'); import dev_run; "
        "print('app.main' in sys.modules)" % (ROOT / "scripts")
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"
