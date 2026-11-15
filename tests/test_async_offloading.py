# tests/test_async_offloading.py
# Tests for audit findings F-019, F-024 and F-010.
#
# F-019: heavy models (Whisper, SepFormer, ECAPA) must not be built at
#        core.config import time -- they become lazy singletons with accessor
#        functions; the old module-level names stay importable through lazy
#        proxies so existing `from core.config import asr_model` users work.
# F-024: VoiceStore must reuse one AsrTranscriber instead of constructing a
#        fresh Whisper model per uncached transcription (per language, to keep
#        the language kwarg semantics).
# F-010: async REST handlers must offload blocking model/service calls with
#        asyncio.to_thread (mirror of routes_ws.py), so a concurrent heartbeat
#        coroutine is not starved during inference.
import ast
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# The stub qwen_tts directory the audit test command puts on PYTHONPATH; it is
# needed by the subprocess that imports app.main without the real TTS package.
_AUDIT_STUB_FALLBACK = Path(
    r"C:/Users/name/PycharmProjects/misc/teachCopilot/Alena_audio/.audit/20261115-0002/repro/stub_qwen_tts"
)


def _find_stub_dir() -> Path:
    for entry in sys.path:
        try:
            p = Path(entry)
        except (OSError, ValueError):
            continue
        if p.name == "stub_qwen_tts" and (p / "qwen_tts").is_dir():
            return p
    if _AUDIT_STUB_FALLBACK.is_dir():
        return _AUDIT_STUB_FALLBACK
    pytest.fail("stub_qwen_tts directory not found (check PYTHONPATH)")


# ---------------------------------------------------------------------------
# F-019
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = r'''
import json, sys, types

WORKTREE = r"__WORKTREE__"

counts = {"whisper_load_model": 0, "sepformer_from_hparams": 0, "ecapa_from_hparams": 0}


class _Dummy:
    def transcribe(self, audio, language=None, **kw):
        return {"text": ""}

    def separate_batch(self, audio):
        return audio.unsqueeze(0)

    def encode_batch(self, audio):
        return audio

    def eval(self):
        return self


def _count(key):
    def _make(*args, **kwargs):
        counts[key] += 1
        return _Dummy()
    return _make


# Install counting stubs for whisper and speechbrain BEFORE importing the app,
# so no real weights are touched and every model construction is recorded.
whisper_stub = types.ModuleType("whisper")
whisper_stub.load_model = _count("whisper_load_model")
sys.modules["whisper"] = whisper_stub

speechbrain = types.ModuleType("speechbrain")
speechbrain.__path__ = []
sb_inference = types.ModuleType("speechbrain.inference")
sb_inference.__path__ = []
sb_separation = types.ModuleType("speechbrain.inference.separation")
class _Sepformer:
    from_hparams = staticmethod(_count("sepformer_from_hparams"))
sb_separation.SepformerSeparation = _Sepformer
sb_speaker = types.ModuleType("speechbrain.inference.speaker")
class _SpeakerRecognition:
    from_hparams = staticmethod(_count("ecapa_from_hparams"))
sb_speaker.SpeakerRecognition = _SpeakerRecognition
class _EncoderClassifier:
    pass
sb_speaker.EncoderClassifier = _EncoderClassifier
sys.modules["speechbrain"] = speechbrain
sys.modules["speechbrain.inference"] = sb_inference
sys.modules["speechbrain.inference.separation"] = sb_separation
sys.modules["speechbrain.inference.speaker"] = sb_speaker

sys.path.insert(0, WORKTREE)

# Phase 1: importing app.main must construct zero models (F-019).
import app.main  # noqa: E402
print(json.dumps({"phase": "after_import_app_main", "counts": counts}), flush=True)

# Phase 2: the lazy accessors must build the model on first use and reuse it.
accessor = {"ok": False, "counts_after_use": None, "error": None}
try:
    from core.config import get_asr_model  # noqa: E402
    first = get_asr_model()
    second = get_asr_model()
    assert first is not None and first is second, "get_asr_model must return a singleton"
    accessor["ok"] = True
    accessor["counts_after_use"] = dict(counts)
except Exception as e:  # noqa: BLE001
    accessor["error"] = "%s: %s" % (e.__class__.__name__, e)
print(json.dumps({"phase": "accessor_use", **accessor}), flush=True)
'''.replace("__WORKTREE__", str(ROOT))


def _run_import_subprocess(tmp_path: Path) -> list[dict]:
    script = tmp_path / "_import_app_main_probe.py"
    script.write_text(_SUBPROCESS_SCRIPT, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_find_stub_dir())
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    events = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    by_phase = {e["phase"]: e for e in events}
    if "after_import_app_main" not in by_phase:
        pytest.fail(
            "probe subprocess died before import_app_main finished; "
            f"returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
        )
    return [by_phase["after_import_app_main"], by_phase.get("accessor_use")]


def test_import_app_main_constructs_no_models(tmp_path):
    """F-019: `import app.main` must not build Whisper/SepFormer/ECAPA."""
    after_import, accessor = _run_import_subprocess(tmp_path)
    assert after_import["counts"] == {
        "whisper_load_model": 0,
        "sepformer_from_hparams": 0,
        "ecapa_from_hparams": 0,
    }, f"models constructed at import time: {after_import['counts']}"

    # The lazy accessors still work: first use builds Whisper exactly once.
    assert accessor is not None and accessor["ok"], f"accessor failed: {accessor and accessor['error']}"
    assert accessor["counts_after_use"]["whisper_load_model"] == 1
    assert accessor["counts_after_use"]["sepformer_from_hparams"] == 0
    assert accessor["counts_after_use"]["ecapa_from_hparams"] == 0


def _module_body_calls(path: Path, func_names: set[str]) -> list[str]:
    """Calls to func_names made at module level (outside function/class bodies)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in tree.body:  # module-level statements only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # calls inside functions are lazy, not import-time
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                if call.func.attr in func_names:
                    found.append(f"{call.func.attr}@line {call.lineno}")
    return found


def test_config_has_lazy_accessors_and_no_import_time_model_builds():
    """F-019: accessors exist and no model is built at core.config module level."""
    src = ROOT / "core" / "config.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for accessor in ("get_asr_model", "get_noise_model", "get_speech_verification_model"):
        assert accessor in top_level_functions, f"core.config must expose accessor {accessor}()"

    builds = _module_body_calls(src, {"load_model", "from_hparams"})
    assert builds == [], f"model construction at core.config module level: {builds}"


def test_lazy_proxy_materializes_once_and_supports_attribute_patch():
    """The compatibility proxy constructs the model only on first attribute use
    and supports the conftest patching pattern (setattr on the proxy)."""
    from core import config

    builds = {"n": 0}

    class _FakeModel:
        language = "ru"

        def __init__(self):
            builds["n"] += 1

        def transcribe(self, audio, language=None, **kw):
            return {"text": "real"}

    proxy = config._LazyModel(_FakeModel)
    assert builds["n"] == 0, "proxy creation must not build the model"

    assert proxy.transcribe("a")["text"] == "real"
    assert builds["n"] == 1
    assert proxy.transcribe("b")["text"] == "real"
    assert builds["n"] == 1, "proxy must reuse the single built model"

    # conftest.py-style patching: monkeypatch.setattr(config.asr_model, ...)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(proxy, "transcribe", lambda audio, language=None: {"text": "fake"})
        assert proxy.transcribe("c")["text"] == "fake"
        assert builds["n"] == 1
    finally:
        monkeypatch.undo()
    assert proxy.transcribe("d")["text"] == "real"


# ---------------------------------------------------------------------------
# F-024
# ---------------------------------------------------------------------------

def test_voice_store_reuses_one_transcriber_for_uncached_calls(tmp_path, monkeypatch):
    """F-024: two uncached transcriptions must construct exactly one
    AsrTranscriber; language semantics are kept (cache per language)."""
    import core.TTS.voices as voices_mod
    from core.TTS.voices import VoiceStore

    made = []

    class _CountingTranscriber:
        def __init__(self, *args, **kwargs):
            self.language = kwargs.get("language")
            made.append(self)

        def transcribe(self, path):
            return "text-" + str(id(self))

    monkeypatch.setattr(voices_mod, "AsrTranscriber", _CountingTranscriber)

    def _wav(path: Path, content: bytes) -> Path:
        path.write_bytes(content)  # content is irrelevant: only its sha1 matters
        return path

    w1 = _wav(tmp_path / "a.wav", b"audio-a")
    w2 = _wav(tmp_path / "b.wav", b"audio-b")
    w3 = _wav(tmp_path / "c.wav", b"audio-c")
    w4 = _wav(tmp_path / "d.wav", b"audio-d")

    store = VoiceStore(tmp_path / "voices")

    # Two uncached files, same (default) language -> one transcriber instance.
    t1 = store._transcribe_with_local_cache(w1)
    t2 = store._transcribe_with_local_cache(w2)
    assert len(made) == 1, f"expected one AsrTranscriber, got {len(made)}"
    assert t1 == t2, "both calls must run through the same transcriber instance"

    # A different language must still get its own transcriber (kwarg semantics).
    store._transcribe_with_local_cache(w3, language="ru")
    assert len(made) == 2
    assert [t.language for t in made] == [None, "ru"]

    # Same language again -> still no new construction.
    store._transcribe_with_local_cache(w4, language="ru")
    assert len(made) == 2


# ---------------------------------------------------------------------------
# F-010
# ---------------------------------------------------------------------------

_BLOCK = 0.3  # seconds the fake blocking inference sleeps


class _FakeUpload:
    def __init__(self, filename="test.wav", payload=b"RIFF....WAVEfmt "):
        self.filename = filename
        self._payload = payload

    async def read(self):
        return self._payload


def _run_handler_with_heartbeat(handler_factory, block_sec=_BLOCK, tick=0.05):
    """Run `handler_factory()` as a coroutine next to a 0.05 s heartbeat and
    return (result, max heartbeat gap). Mirrors the V3 cl1_loop_block demo."""
    ticks = []

    async def _heartbeat():
        while True:
            ticks.append(time.perf_counter())
            await asyncio.sleep(tick)

    async def _scenario():
        hb = asyncio.create_task(_heartbeat())
        await asyncio.sleep(tick * 2)  # establish cadence
        try:
            result = await handler_factory()
            # Let the heartbeat tick again after the handler: if the loop was
            # starved, the overdue tick fires right after unblocking and the
            # stall shows up as a gap between recorded ticks (V3 pattern).
            await asyncio.sleep(tick * 2)
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb
        return result

    result = asyncio.run(_scenario())
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    max_gap = max(gaps) if gaps else 0.0
    assert max_gap < block_sec * 0.8, (
        f"event loop starved for {max_gap:.3f}s by a blocking call of {block_sec}s"
    )
    return result, max_gap


def test_audio_transcribe_handler_offloads_blocking_inference(tmp_path, monkeypatch):
    import app.api.routes_audio as ra
    from app.services.audio_service import AudioService

    def fake_transcribe_file(input_wav, language="ru"):
        time.sleep(_BLOCK)  # blocking, like real whisper inference
        return {"text": "ok", "raw": {}}

    monkeypatch.setattr(ra, "STORAGE_DIR", tmp_path)
    fresh = AudioService(tmp_path)
    monkeypatch.setattr(fresh, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(ra, "svc", fresh)

    result, _gap = _run_handler_with_heartbeat(
        lambda: ra.transcribe(file=_FakeUpload(), language="ru", verbose=False)
    )
    assert result.text == "ok"


def test_speaker_verify_handler_offloads_blocking_embedding(tmp_path, monkeypatch):
    import app.api.routes_speaker as rs

    def fake_verify_files(probe_wav, reference_wav=None):
        time.sleep(_BLOCK)  # blocking, like ECAPA inference
        return {"score": 0.9, "decision": True}

    monkeypatch.setattr(rs, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(rs.svc, "verify_files", fake_verify_files)

    result, _gap = _run_handler_with_heartbeat(lambda: rs.verify(probe=_FakeUpload(), reference=None))
    assert result == {"score": 0.9, "decision": True}


def test_speaker_verify_registry_handler_offloads_blocking_matching(tmp_path, monkeypatch):
    import app.api.routes_speaker as rs

    def fake_load_and_resample(buf, target_sr=None):
        return np.zeros(1600, dtype=np.float32)

    def fake_match_probe_array(audio, top_k=5):
        time.sleep(_BLOCK)  # blocking, like ECAPA/SepFormer inference
        return [{"user_id": "u1", "score": 0.99, "ref_path": "x.npy"}]

    monkeypatch.setattr(rs, "load_and_resample", fake_load_and_resample)
    monkeypatch.setattr(rs.matcher, "match_probe_array", fake_match_probe_array)

    result, _gap = _run_handler_with_heartbeat(
        lambda: rs.verify_registry(probe=_FakeUpload(), sim_threshold=0.65, top_k=5)
    )
    assert result["decision"] is True
    assert result["best"].user_id == "u1"


def test_tts_clone_handler_offloads_blocking_transcription(tmp_path, monkeypatch):
    import app.api.routes_TTS as rt

    class _FakeStore:
        def clone_from_upload(self, up_path, sample_rate=24000, language=None):
            time.sleep(_BLOCK)  # blocking, like Whisper + pydub work
            return SimpleNamespace(voice_id="fake-vid")

    monkeypatch.setattr(rt, "store", _FakeStore())

    result, _gap = _run_handler_with_heartbeat(lambda: rt.clone_voice(file=_FakeUpload("voice.wav")))
    assert result == {"voice_id": "fake-vid"}
