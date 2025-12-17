# scripts/dev_run.py
from __future__ import annotations
import sys, asyncio, uvicorn, argparse
from pathlib import Path
from print_routes import *

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def main():
    parser = argparse.ArgumentParser("(scripts/dev_run.py) Dev run for API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-reload", action="store_true", help="Disable autoreload")
    parser.add_argument("--print-routes", action="store_true", help="Print API routes")

    args = parser.parse_args()

    if args.print_routes:
        print_routes_()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        # следим только за исходниками приложения
        reload_dirs=["app", "core"],
        # исключаем тяжёлые/шумные каталоги
        reload_excludes=[".venv/*", "pretrained_models/*", "**/.no_exist/*", "__pycache__/*"],
        log_level="info",
    )





if __name__ == "__main__":
    main()
