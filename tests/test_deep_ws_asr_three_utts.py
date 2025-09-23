# tests/test_deep_ws_asr_three_utts.py
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.audio_utils import ensure_pcm16_mono_16k

pytestmark = pytest.mark.deep

client = TestClient(app)

SAMPLES = [
    Path("tests/samples/ru_sample.wav"),
    Path("tests/samples/utt2.wav"),
    Path("core/misha_20sec.wav"),
]

TARGET_SR = 16000
CHUNK_BYTES = 32000  # ~1 сек при 16kHz mono 16-bit


def _load_or_generate_pcm16_bytes(path: Path, seconds: float, freq: float) -> bytes:
    """
    Если файл существует — читаем как WAV PCM16 mono 16k и отдаём его data-чанк.
    Если нет — генерируем тон и делаем PCM16 байты.
    """
    if path.exists():
        return ensure_pcm16_mono_16k(path)

    # Генерация синтетики (если файла нет)
    t = np.linspace(0.0, seconds, int(TARGET_SR * seconds), endpoint=False, dtype=np.float32)
    x = 0.1 * np.sin(2 * np.pi * freq * t)  # амплитуда пониже
    x = np.clip(x, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)
    return x_i16.tobytes()


def _send_pcm_in_chunks(ws, pcm: bytes, chunk_bytes: int = CHUNK_BYTES, sleep_s: float = 0.01):
    for i in range(0, len(pcm), chunk_bytes):
        ws.send_bytes(pcm[i:i + chunk_bytes])
        # маленькая пауза, чтобы эмуляция была ближе к реальному стриму
        time.sleep(sleep_s)


def _recv_until_final(ws, expect_utt_id: str, timeout_s: float = 10.0):
    """
    Читает сообщения до получения {"type":"final", ...} или таймаута.
    Возвращает последнюю final-строку (или бросает AssertionError при таймауте).
    """
    deadline = time.time() + timeout_s
    last_partial = None
    while time.time() < deadline:
        msg = ws.receive_json()
        mtype = msg.get("type")
        if mtype == "partial":
            # допустимо, просто запомним
            last_partial = msg
        elif mtype == "final":
            # если сервер поддерживает echo utt_id — сверим
            if "utt_id" in msg:
                assert msg["utt_id"] == expect_utt_id
            assert "text" in msg
            print(msg["text"])
            return msg
        elif mtype == "ready":
            # повторный ready не ожидаем в одном соединении
            continue
        elif mtype == "error":
            pytest.fail(f"WS error: {msg}")
        else:
            # игнорируем прочие служебные типы
            continue
    raise AssertionError("Не дождались 'final' за отведённое время")


def test_ws_asr_three_utterances_long_lived_socket():
    """
    Один вебсокет → 3 фразы подряд.
    Для каждой: отправили аудио, (опц.) flush, stop → получили final.
    Сокет остаётся открытым между фразами.
    """
    # Подготовим PCM для трёх «фраз»: если файлов нет, сгенерируем
    pcm_list = [
        _load_or_generate_pcm16_bytes(SAMPLES[0], seconds=1.5, freq=320.0),
        _load_or_generate_pcm16_bytes(SAMPLES[1], seconds=2.0, freq=440.0),
        _load_or_generate_pcm16_bytes(SAMPLES[2], seconds=1.2, freq=260.0),
    ]

    with client.websocket_connect(
        f"/ws/asr?language=ru&sample_rate={TARGET_SR}&channels=1&window_sec=8&emit_sec=2&inactivity_sec=3"
    ) as ws:
        # ready
        first = ws.receive_json()
        assert first.get("type") == "ready"
        assert first.get("sample_rate") == TARGET_SR

        finals = []

        for idx, pcm in enumerate(pcm_list, start=1):
            utt_id = f"utt{idx}"

            # стримим бинарные куски
            _send_pcm_in_chunks(ws, pcm)

            # (опционально) попросим partial сразу
            ws.send_text(json.dumps({"event": "flush", "utt_id": utt_id}))
            resp = ws.receive_json()
            assert resp.get("type") in ("partial", "final")  # иногда сразу final
            if resp.get("type") == "final":
                # сервер мог сам автофинализировать — считаем это финалом фразы
                finals.append(resp)
                # перед следующей фразой попросим сброс на всякий случай (не обязательно)
                ws.send_text(json.dumps({"event": "reset", "utt_id": utt_id}))
                _ = ws.receive_json()  # {"type":"ok","detail":"reset",...}
                continue

            # теперь финализируем явно
            ws.send_text(json.dumps({"event": "stop", "utt_id": utt_id}))
            final_msg = _recv_until_final(ws, expect_utt_id=utt_id, timeout_s=15.0)
            finals.append(final_msg)

            # после финала сокет остаётся открыт; буфер сервер сбрасывает сам
            # (если хочется — можно явно ws.send_text({"event":"reset"}))

        # Убедимся, что получили три финальных ответа
        assert len(finals) == 3
        for i, fmsg in enumerate(finals, start=1):
            assert fmsg["type"] == "final"
            assert "text" in fmsg
            # допускам пустой текст на синтетике/шуме — главное, что протокол устойчив
            assert isinstance(fmsg["text"], str)
