"""Build the local material index from the canonical database."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_app.config import Settings
from local_app.store import Store
from local_app.runtime import OllamaRuntime
from local_app.rag import RagService

if __name__ == '__main__':
    settings = Settings.from_env()
    print(json.dumps(RagService(Store(settings.database_url), OllamaRuntime(settings), settings).reindex()))
