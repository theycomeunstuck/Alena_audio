# core/tts/tts_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
from fastapi import HTTPException
from pydub import AudioSegment
from pydub.utils import which
from app import settings
import asyncio, io, soundfile as sf
from app.settings import STORAGE_DIR

import torch
from qwen_tts import Qwen3TTSModel

# singleton
_tts_engine: TtsEngine | None = None
def get_tts_engine() -> TtsEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TtsEngine()
    return _tts_engine


class TtsEngine:
    def __init__(self):
        self._QwenTTS = Qwen3TTSModel.from_pretrained(
            "pretrained_models/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )


        self.max_sec = settings.TTS_MAX_SECONDS
        if not which("ffmpeg"):
            raise RuntimeError("FFmpeg не найден в PATH. Установите ffmpeg и перезапустите.")


    async def synth(self, text: str, ref_audio: Path, ref_text: str, vid: str, out_format: Literal["wav", "mp3", "ogg"] = "wav") -> bytes:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Поле 'text' пустое или отсутствует")

        try:
            return await asyncio.wait_for(
                self._synth_api(gen_text=text.strip(), ref_audio=ref_audio,
                                out_format=out_format, ref_text=ref_text, vid=vid),
                timeout=self.max_sec)


        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail=f"Генерация превысила лимит {settings.TTS_MAX_SECONDS} с")


    async def _synth_api(
            self,
            gen_text: str,
            ref_audio: Path,
            ref_text: str,
            out_format: str,
            vid: str
    ) -> bytes:

        audio_np, sr_ref = sf.read(ref_audio) # Уже делали проверку на наличие файла в app/services/routes_TTS.py
        wavs, sr = await asyncio.to_thread(self._QwenTTS.generate_voice_clone,
            text=gen_text,
            language="Auto",
            ref_audio=str(ref_audio),
            ref_text=ref_text.replace("+", "") #todo: Ударения в qwen_tts не работают. Ударения больше не добавляются, но в некоторых files still stress existing
        )

        out_dir = STORAGE_DIR / "out_TTS" / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        # путь к файлу
        out_file = out_dir / f"{vid}.wav"

        if out_file.exists():
            try: out_file.unlink()
            except Exception as e:
                print("Cannot delete existing file:", e)
        # сохраняем звук
        sf.write(out_file, wavs[0], sr, format="WAV")

        # сохраяем в память
        wav_bytes = io.BytesIO()
        wav_bytes.seek(0)
        sf.write(wav_bytes, wavs[0], sr,  format="WAV")
        wav_bytes.seek(0)

        # если нужен WAV — возвращаем прямо его
        if out_format == "wav":
            return wav_bytes.getvalue()

        # иначе конвертируем через pydub
        audio = AudioSegment.from_wav(wav_bytes)
        out_mem = io.BytesIO()

        if out_format == "mp3":
            audio.export(out_mem, format="mp3", bitrate="192k")
        elif out_format == "ogg":
            audio.export(out_mem, format="ogg")
        else:
            raise HTTPException(400, "Неверный выходной формат аудио")

        return out_mem.getvalue()