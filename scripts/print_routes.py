# scripts/print_routes.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app
from fastapi.routing import APIRoute


def print_routes_():
    print("\nAvailable API routes (w/o websockets):")

    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = ",".join(sorted(route.methods))
            print(f"{methods:10s} {route.path}")
    print()


