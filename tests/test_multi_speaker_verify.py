# tests/test_multi_speaker_verify.py
import io, json, wave
import numpy as np
import torch
import pytest
from fastapi.testclient import TestClient
from app.main import app
from core.config import SAMPLE_RATE

client = TestClient(app)

def _wav_bytes(x: np.ndarray, sr: int) -> bytes:
    """PCM16 mono WAV в памяти через стандартный модуль wave.

    torchaudio в этом окружении не работает (torchcodec не может загрузить свою
    DLL), поэтому и проба, и реестр собираются без него."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x16 = (x * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x16.tobytes())
    buf.seek(0)
    return buf.getvalue()

def _fake_embed(audio: np.ndarray) -> torch.Tensor:
    """Дешёвый детерминированный эмбеддинг по спектру, L2-нормированный."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    # простой спектральный вектор фиксированной длины
    nfft = 2048
    if x.size < nfft:
        pad = np.zeros(nfft - x.size, dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    spec = np.abs(np.fft.rfft(x[:nfft]))[:256].astype(np.float32)  # 256 фичей
    v = torch.from_numpy(spec)
    v = v / (v.norm(p=2) + 1e-8)
    return v

def _pcm16_bytes(x: np.ndarray) -> bytes:
    """1D float32 [-1..1] -> PCM16 little-endian bytes (формат кадров /ws/speaker/verify)."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


class _PassThroughEnhancement:
    """Замена Audio_Enhancement (SepFormer): возвращает аудио без изменений."""

    def __init__(self, audio, ref=None):
        self.audio = np.asarray(audio, dtype=np.float32)
        self.ref = None if ref is None else np.asarray(ref, dtype=np.float32)

    def noise_suppression(self):
        return self.audio


@pytest.fixture(autouse=True)
def fake_registry(tmp_path, monkeypatch):
    """Патчим ровно те биндинги, которые читает production-код, и держим
    реестр в tmp_path вместо настоящего storage/.

    - app.services.multi_speaker_matcher.embed_speechbrain — матчер вызывает
      биндинг из СВОЕГО пространства имён (from ... import embed_speechbrain),
      поэтому патч в app.services.embeddings_utils до него не доходил.
    - app.services.multi_speaker_matcher.Audio_Enhancement — тот же принцип
      (иначе на каждый матч грузится настоящий SepFormer).
    - app.api.routes_speaker.matcher — REST-роуты читают этот module-level
      матчер; заменяем его на матчер над tmp-реестром (.npy в tmp_path).
    - app.services.streaming_verify.get_global_matcher — WS-сессия верификации
      вызывает этот биндинг; возвращаем тот же tmp-матчер.
    """
    import app.services.multi_speaker_matcher as msm
    import app.api.routes_speaker as rs
    import app.services.streaming_verify as sv

    monkeypatch.setattr(msm, "embed_speechbrain", _fake_embed)
    monkeypatch.setattr(msm, "Audio_Enhancement", _PassThroughEnhancement)

    # Два синтетических пользователя в tmp-реестре (EMBEDDINGS_DIR/*.npy):
    # реестр матчера — .npy-эмбеддинги, а не WAV в VOICES_DIR.
    sr = SAMPLE_RATE
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
    v1 = _fake_embed(0.1 * np.sin(2 * np.pi * 220.0 * t))
    v2 = _fake_embed(0.1 * np.sin(2 * np.pi * 330.0 * t))
    np.save(tmp_path / "u1.npy", v1.numpy())
    np.save(tmp_path / "u2.npy", v2.numpy())

    matcher = msm.MultiSpeakerMatcher(tmp_path, sample_rate=sr)
    loaded = matcher.reload()
    assert loaded[1] == 2  # (id_list, count): tmp-реестр подхвачен целиком

    monkeypatch.setattr(rs, "matcher", matcher)
    monkeypatch.setattr(sv, "get_global_matcher", lambda: matcher)
    return matcher

def test_rest_verify_all_returns_matches():
    # Probe tone 220 Hz → ближе к u1
    sr = SAMPLE_RATE
    t = np.linspace(0, 0.5, int(0.5*sr), endpoint=False, dtype=np.float32)
    x = 0.1 * np.sin(2*np.pi*220.0 * t).astype(np.float32)
    files = {"probe": ("probe.wav", _wav_bytes(x, sr), "audio/wav")}
    r = client.post("/speaker/verify/topk?top_k=2", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "matches" in data and isinstance(data["matches"], list)
    assert data["count"] == len(data["matches"])
    assert data["count"] == 2  # tmp-реестр из двух пользователей реально прогружен
    assert all(0.0 <= m["score"] <= 1.0 for m in data["matches"])
    assert data["matches"][0]["user_id"] == "u1"  # 220 Гц ближе к u1, чем к u2

def test_ws_verify_basic_flow():
    # Open WS and send a short chunk, then flush
    def recv_until_any(ws, types=("partial", "final"), max_msgs=20):
        for _ in range(max_msgs):
            msg = ws.receive_json()
            if msg.get("type") in types:
                return msg
        raise AssertionError("Did not receive partial/final within limit")

    sr = SAMPLE_RATE
    rng = np.random.default_rng(42)
    with client.websocket_connect(f"/ws/speaker/verify?sample_rate={sr}&channels=1&top_k=2") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        # 500 мс — больше минимума сессии (300 мс), иначе матчер вернёт пустой список
        x = rng.uniform(-0.1, 0.1, int(0.5*sr)).astype(np.float32)
        ws.send_bytes(_pcm16_bytes(x))
        ws.send_text(json.dumps({"event": "flush"}))

        ws.send_text(json.dumps({"event": "flush"}))
        # Первый partial тикера приходит сразу после connect с пустым буфером
        # (совпадений 0) — ждём partial/final, посчитанный уже ПОСЛЕ приёма аудио.
        msg = recv_until_any(ws)
        for _ in range(20):
            if msg.get("type") in ("partial", "final") and msg.get("matches"):
                break
            msg = ws.receive_json()
        else:
            raise AssertionError("Did not receive a computed partial/final with matches")
        assert msg["type"] in ("partial", "final")
        assert "matches" in msg
        assert len(msg["matches"]) == 2
        assert all(0.0 <= m["score"] <= 1.0 for m in msg["matches"])
