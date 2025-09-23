import httpx
from pathlib import Path

BASE = "http://127.0.0.1:8000"

def verify(probe: Path, ref: Path | None = None):
    files = {"probe": (probe.name, probe.read_bytes(), "audio/wav")}
    if ref is not None:
        files["reference"] = (ref.name, ref.read_bytes(), "audio/wav")
    r = httpx.post(f"{BASE}/speaker/verify", files=files, timeout=120)
    r.raise_for_status()
    print(r.json())  # {"score": ..., "decision": ...}

if __name__ == "__main__":
    verify(Path("probe.wav"), Path("ref.wav"))
