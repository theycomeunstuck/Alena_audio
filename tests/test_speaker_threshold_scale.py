# tests/test_speaker_threshold_scale.py
"""Audit F-008 regression tests: sim_threshold lives on the RAW cosine scale.

core/config.sim_threshold = 0.65 is a raw-cosine threshold – that is how
speaker_service.verify_files already treats it. The registry decision paths –
matcher.binary_decision, POST /speaker/verify_registry and the WS path
StreamingVerifySession.current_best_binary – used to compare it against the
[0..1] display score produced by _cos_to01 ((cos + 1) / 2), which effectively
accepted any raw cosine >= 0.30 (mapped 0.65) – wrong speakers were accepted.
After the fix these decisions compare the threshold against the RAW cosine,
while the API `score` fields stay mapped to [0..1] for display (documented
contract "Схожесть в диапазоне [0..1]") and the response shapes are unchanged.

All tests drive the real matcher/decision code with synthetic embeddings:
embed_speechbrain is monkeypatched to return controlled unit vectors, so no
models are loaded.
"""

import asyncio
import math

import numpy as np
import pytest
import torch

import app.services.multi_speaker_matcher as msm
import app.api.routes_speaker as rs
import app.services.streaming_verify as sv
import app.services.speaker_service as ss
from core.config import sim_threshold as DEFAULT_SIM_THRESHOLD

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _unit(vec) -> torch.Tensor:
    t = torch.tensor(vec, dtype=torch.float32)
    return torch.nn.functional.normalize(t, p=2, dim=-1, eps=1e-12)


def _probe_with_raw_cosine(c: float) -> torch.Tensor:
    """Unit probe vector whose raw cosine with the registry vector [1, 0] is c."""
    return _unit([c, math.sqrt(max(0.0, 1.0 - c * c))])


class _NoopEnhancement:
    """Audio_Enhancement stand-in: no SepFormer, audio passes through."""

    def __init__(self, audio, ref=None):
        self.audio = np.asarray(audio, dtype=np.float32)

    def noise_suppression(self):
        return self.audio


class _FakeUpload:
    """Minimal UploadFile stand-in for direct handler calls."""

    def __init__(self, data: bytes = b"unused"):
        self._data = data

    async def read(self) -> bytes:
        return self._data


@pytest.fixture()
def single_user_matcher(tmp_path, monkeypatch):
    """Real MultiSpeakerMatcher over a one-user registry ([1, 0] embedding)."""
    np.save(tmp_path / "u.npy", np.asarray([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(msm, "CFG_DEVICE", _device, raising=False)
    monkeypatch.setattr(msm, "Audio_Enhancement", _NoopEnhancement, raising=True)
    m = msm.MultiSpeakerMatcher(embeddings_dir=tmp_path)
    assert m.reload()[1] == 1
    return m


def _set_probe(monkeypatch, probe_vec: torch.Tensor) -> None:
    """Pin the probe embedding returned by the (mocked) embedder."""
    v = probe_vec.clone()
    monkeypatch.setattr(msm, "embed_speechbrain", lambda audio: v, raising=True)


# --- 1) matcher.binary_decision -------------------------------------------------

def test_binary_decision_rejects_raw_040_and_accepts_raw_080_at_default_threshold(single_user_matcher, monkeypatch):
    m = single_user_matcher
    audio = np.zeros(16000, dtype=np.float32)  # content irrelevant: the embedder is mocked

    _set_probe(monkeypatch, _probe_with_raw_cosine(0.40))
    decision, best = m.binary_decision(audio)  # default threshold = core.config.sim_threshold (0.65)
    assert decision is False  # RED before the fix: mapped display score 0.70 >= 0.65 accepted a raw 0.40
    assert best is not None  # best is still reported when the registry is non-empty
    assert best["user_id"] == "u"
    # The API score field stays mapped to [0..1] for display (contract unchanged)
    assert best["score"] == pytest.approx((0.40 + 1.0) / 2.0, abs=1e-6)

    _set_probe(monkeypatch, _probe_with_raw_cosine(0.80))
    decision2, best2 = m.binary_decision(audio)
    assert decision2 is True  # raw 0.80 >= 0.65
    assert best2["score"] == pytest.approx((0.80 + 1.0) / 2.0, abs=1e-6)


# --- 2) POST /speaker/verify_registry decision path -----------------------------

def test_verify_registry_route_decides_on_raw_cosine(single_user_matcher, monkeypatch):
    monkeypatch.setattr(rs, "matcher", single_user_matcher, raising=True)
    monkeypatch.setattr(rs, "load_and_resample", lambda buf: np.zeros(1600, dtype=np.float32), raising=True)

    _set_probe(monkeypatch, _probe_with_raw_cosine(0.40))
    res = asyncio.run(rs.verify_registry(probe=_FakeUpload(), sim_threshold=DEFAULT_SIM_THRESHOLD, top_k=3))
    assert res["decision"] is False  # RED before the fix
    assert res["best"] is None  # best is only returned when decision is true (contract unchanged)
    # Top-k matches keep the [0..1] display score for the API response
    assert res["matches"][0]["score"] == pytest.approx((0.40 + 1.0) / 2.0, abs=1e-6)

    _set_probe(monkeypatch, _probe_with_raw_cosine(0.80))
    res2 = asyncio.run(rs.verify_registry(probe=_FakeUpload(), sim_threshold=DEFAULT_SIM_THRESHOLD, top_k=3))
    assert res2["decision"] is True
    assert res2["best"] is not None and res2["best"].user_id == "u"  # best comes back as MultiVerifyMatch
    assert res2["best"].score == pytest.approx((0.80 + 1.0) / 2.0, abs=1e-6)


# --- 3) WS decision path (StreamingVerifySession.current_best_binary) -----------

def test_streaming_current_best_binary_decides_on_raw_cosine(single_user_matcher, monkeypatch):
    monkeypatch.setattr(sv, "get_global_matcher", lambda: single_user_matcher, raising=True)
    sess = sv.StreamingVerifySession(sample_rate=16000, channels_hint=1, inactivity_sec=None)
    sess.ingest_pcm16_chunk(np.zeros(int(0.5 * 16000), dtype="<i2").tobytes())  # 500 ms > the 300 ms gate

    _set_probe(monkeypatch, _probe_with_raw_cosine(0.40))
    out = sess.current_best_binary(threshold=DEFAULT_SIM_THRESHOLD, top_k=3)
    assert out["decision"] is False  # RED before the fix
    assert out["best"] is None
    assert out["matches"][0]["score"] == pytest.approx((0.40 + 1.0) / 2.0, abs=1e-6)

    _set_probe(monkeypatch, _probe_with_raw_cosine(0.80))
    out2 = sess.current_best_binary(threshold=DEFAULT_SIM_THRESHOLD, top_k=3)
    assert out2["decision"] is True
    assert out2["best"] is not None and out2["best"]["user_id"] == "u"


# --- 4) /speaker/verify (speaker_service.verify_files) raw behaviour unchanged --

def test_speaker_verify_files_keeps_raw_cosine_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "load_and_resample", lambda src: np.zeros(1600, dtype=np.float32), raising=True)
    ref = _unit([1.0, 0.0])
    service = ss.SpeakerService(tmp_path)

    for raw_cos, expected_decision in ((0.40, False), (0.80, True)):
        probe = _probe_with_raw_cosine(raw_cos)
        seen = []

        def fake_embed(audio, _probe=probe, _ref=ref, _seen=seen):
            # verify_files embeds the probe first, then the reference
            _seen.append(1)
            return _probe if len(_seen) == 1 else _ref

        monkeypatch.setattr(ss, "embed_speechbrain", fake_embed, raising=True)
        res = service.verify_files(tmp_path / "probe.wav", tmp_path / "reference.wav")
        assert res["decision"] is expected_decision  # already raw-cosine; must not change
        assert res["score"] == pytest.approx(raw_cos, abs=1e-6)  # /speaker/verify returns the RAW cosine as score
