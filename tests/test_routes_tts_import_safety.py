# Guards against import-time filesystem checks in routes_TTS (audit F-003).
# The default-voice reference.wav check used to run at module import, which
# crashed the whole API (and pytest collection) on a fresh checkout.
import ast
from pathlib import Path


def _module_body_calls(path: str, func_name: str) -> list[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = []
    for node in tree.body:  # module-level statements only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # calls inside functions are per-request, not import-time
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                target = call.func
                if isinstance(target, ast.Attribute) and target.attr == func_name:
                    names.append(ast.dump(target))
    return names


def test_no_import_time_reference_check():
    src = Path(__file__).resolve().parents[1] / "app" / "api" / "routes_TTS.py"
    calls = _module_body_calls(str(src), "ensure_reference_wav")
    assert calls == [], "routes_TTS must not call ensure_reference_wav at import time"


def test_tts_endpoint_still_checks_voice_per_request():
    src = Path(__file__).resolve().parents[1] / "app" / "api" / "routes_TTS.py"
    text = src.read_text(encoding="utf-8")
    assert "ensure_reference_wav(vid)" in text, "the /tts handler must keep the per-request voice check"
