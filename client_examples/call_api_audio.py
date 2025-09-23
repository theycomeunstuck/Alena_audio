import httpx
from pathlib import Path

BASE = "http://127.0.0.1:8000"

def enhance_local(path: Path):
    files = {"file": (path.name, path.read_bytes(), "audio/wav")}
    r = httpx.post(f"{BASE}/audio/enhance", files=files, timeout=120)
    r.raise_for_status()
    out_name = r.json()["output_filename"]

    bin_resp = httpx.get(f"{BASE}/files/download/{out_name}", timeout=120)
    bin_resp.raise_for_status()
    (path.parent / f"ENH_{path.name}").write_bytes(bin_resp.content)
    print("Saved:", f"ENH_{path.name}")

def transcribe_local(path: Path, lang="ru"):
    files = {"file": (path.name, path.read_bytes(), "audio/wav")}
    r = httpx.post(f"{BASE}/audio/transcribe", params={"language": lang}, files=files, timeout=180)
    r.raise_for_status()
    print(r.json()["text"])

if __name__ == "__main__":
    wav = Path("ru_sample.wav")
    enhance_local(wav)
    transcribe_local(wav, "ru")
