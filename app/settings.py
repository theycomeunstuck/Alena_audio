from pathlib import Path

STORAGE_DIR = (Path(__file__).parent / "storage").resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
