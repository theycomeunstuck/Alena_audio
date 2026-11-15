# tests/test_routes_speaker_api.py
# (роуты вокруг speaker_service: обработка успешного кейса и ошибок)
import io

import app.api.routes_speaker as rs


def _patch_route_globals(monkeypatch, tmp_path):
    """Направляем временные файлы/сервис/матчер роутов в tmp_path.

    Роуты speaker при импорте собирают svc = SpeakerService(STORAGE_DIR) и
    matcher = get_global_matcher() из app.settings/core.config, а временные
    probe_/ref_-файлы пишут в module-level STORAGE_DIR. Патчим именно эти
    биндинги, чтобы тест не создавал файлы в настоящем storage/.
    """
    from app.services.speaker_service import SpeakerService
    from app.services.multi_speaker_matcher import MultiSpeakerMatcher

    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rs, "STORAGE_DIR", storage)
    monkeypatch.setattr(rs, "svc", SpeakerService(storage))
    monkeypatch.setattr(rs, "matcher", MultiSpeakerMatcher(tmp_path / "embeddings"))
    return storage


def test_verify_ok_and_errors(client, monkeypatch, tmp_path):
    _patch_route_globals(monkeypatch, tmp_path)

    # Успешный ответ сервиса
    monkeypatch.setattr(rs.svc, "verify_files", lambda probe, ref: {"score": 0.87, "decision": True}, raising=True)

    wav = io.BytesIO(b"RIFF....WAVEfmt ")  # контент неважен — мы мокаем сервис
    files = {
        "probe": ("p.wav", wav.getvalue(), "audio/wav"),
        "reference": ("r.wav", wav.getvalue(), "audio/wav"),
    }
    r = client.post("/speaker/verify", files=files)
    assert r.status_code == 200
    assert r.json() == {"score": 0.87, "decision": True}

    # Невалидный ответ сервиса -> 400: это РЕАЛЬНЫЙ контракт роута (его
    # собственная проверка формы ответа бросает ValueError, который роут
    # отображает в 400). Мокать здесь ValueError от сервиса нельзя: настоящий
    # speaker_service.verify_files оборачивает любые свои ошибки (включая
    # «нет дефолтного референса») в generic Exception, и через этот роут они
    # приходят как 500 — мок скрывал этот дефект 500-вместо-400.
    def _invalid_payload(probe, ref):
        return {"score": 0.5}  # нет ключа "decision"

    monkeypatch.setattr(rs.svc, "verify_files", _invalid_payload, raising=True)
    r2 = client.post("/speaker/verify", files={"probe": ("p.wav", wav.getvalue(), "audio/wav")})
    assert r2.status_code == 400

    # Прочая ошибка → 500
    def _raise_any(probe, ref): raise RuntimeError("boom")
    monkeypatch.setattr(rs.svc, "verify_files", _raise_any, raising=True)
    r3 = client.post("/speaker/verify", files={"probe": ("p.wav", wav.getvalue(), "audio/wav")})
    assert r3.status_code == 500


def test_train_microphone_returns_paths(client, monkeypatch, tmp_path):
    _patch_route_globals(monkeypatch, tmp_path)
    ret = {"user_id": "u1", "wav_path": str(tmp_path / "u1.wav"), "npy_path": str(tmp_path / "u1.npy")}
    monkeypatch.setattr(rs.svc, "train_from_microphone", lambda user_id, duration: ret, raising=True)
    r = client.post("/speaker/train/microphone?user_id=u1&duration=1.5")
    assert r.status_code == 200
    js = r.json()
    assert js["user_id"] == "u1" and "wav_path" in js and "npy_path" in js

    # Невалидный user_id -> 400 по настоящему маппингу ошибок роута:
    # validate_resource_id бросает ValueError до всякой записи с микрофона,
    # роут отображает его в 400 (мок сервиса здесь не нужен).
    r_bad = client.post("/speaker/train/microphone?user_id=../escape&duration=1")
    assert r_bad.status_code == 400
