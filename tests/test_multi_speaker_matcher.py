import numpy as np
import torch
import pytest
from pathlib import Path

import app.services.multi_speaker_matcher as msm


def _write_npy(dirpath: Path, name: str, vec):
    arr = np.asarray(vec, dtype=np.float32)
    np.save(dirpath / f"{name}.npy", arr)


@pytest.fixture()
def tmp_registry(tmp_path, monkeypatch):
    # Реестр с 3 эмбеддингами (ненормализованные, класс сам их нормализует)
    _write_npy(tmp_path, "a", [1.0, 0.0, 0.0])
    _write_npy(tmp_path, "b", [0.0, 2.0, 0.0])
    _write_npy(tmp_path, "c", [0.0, 0.0, 3.0])

    # device берём из конфига -> принудительно на CPU
    monkeypatch.setattr(msm, "CFG_DEVICE", "cpu", raising=False)

    # шумодав — no-op
    class DummyEnh:
        def __init__(self, audio): self.audio = np.asarray(audio, dtype=np.float32)
        def noise_suppression(self): return self.audio
    monkeypatch.setattr(msm, "Audio_Enhancement", DummyEnh, raising=True)

    return tmp_path


def test_reload_loads_all_and_places_index_on_device(tmp_registry, monkeypatch):
    m = msm.MultiSpeakerMatcher(embeddings_dir=tmp_registry)
    n = m.reload()
    assert n == 3
    # Индекс на нужном устройстве
    assert isinstance(m._embs, torch.Tensor)
    assert m._embs.device.type == "cpu"  # единый device из конфига
    # Порядок пользователей соответствует сортировке файлов
    assert set(m._user_ids) == {"a", "b", "c"}
    # Размерность [N, D]
    assert m._embs.shape[0] == 3 and m._embs.ndim == 2
# (покрывает требования к reload и device из multi_speaker_matcher.py)


def test_match_top1_correct_and_sorted_desc(tmp_registry, monkeypatch):
    # эмбеддер возвращает «вектор пробы» вдоль "b"
    def fake_embed(audio):
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    monkeypatch.setattr(msm, "embed_speechbrain", fake_embed, raising=True)

    m = msm.MultiSpeakerMatcher(embeddings_dir=tmp_registry)
    m.reload()
    # audio произвольное — эмбеддер замокан
    audio = np.zeros(16000, dtype=np.float32)
    res = m.match_probe_array(audio, top_k=3)

    # top-1 — user "b"
    assert res[0]["user_id"] == "b"
    # Сортировка по убыванию score
    scores = [r["score"] for r in res]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
# (top-1 и сортировка по убыванию) :contentReference[oaicite:5]{index=5}


def test_binary_decision_threshold(tmp_registry, monkeypatch):
    v = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
    v = torch.nn.functional.normalize(v, p=2, dim=-1)
    monkeypatch.setattr(msm, "embed_speechbrain", lambda audio: v, raising=True)

    m = msm.MultiSpeakerMatcher(embeddings_dir=tmp_registry)
    m.reload()
    audio = np.zeros(16000, dtype=np.float32)

    decision, best = m.binary_decision(audio, threshold=0.9)
    assert decision is False
    assert best is not None  # best всегда есть при наличии реестра

    decision2, _ = m.binary_decision(audio, threshold=0.8)
    assert decision2 is True
