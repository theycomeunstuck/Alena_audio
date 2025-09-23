# scripts/dev_run.py
from __future__ import annotations
import argparse
import uvicorn
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser("Dev runner for API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-reload", action="store_true", help="Disable autoreload")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        # следим только за исходниками приложения
        reload_dirs=["app", "core"],
        # исключаем тяжёлые/шумные каталоги (Windows-friendly)
        reload_excludes=[".venv/*", "pretrained_models/*", "**/.no_exist/*", "__pycache__/*"],
        log_level="info",
    )

if __name__ == "__main__":
    main()
