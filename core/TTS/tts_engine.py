# core/tts/tts_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
from fastapi import HTTPException
from pydub import AudioSegment
from pydub.utils import which
from app import settings
import os, sys, shutil, asyncio, tempfile, contextlib, io, soundfile as sf
from app.settings import STORAGE_DIR


from importlib.resources import files
from f5_tts.api import F5TTS



class TtsEngine:
    def __init__(self):
        self._F5TTS = F5TTS(
            ckpt_file=settings.F5TTS_CKPT_PATH,
            vocab_file=settings.VOCAB_FILE_PATH,
            device=settings.DEVICE
        )
        if not which("ffmpeg"):
            raise RuntimeError("FFmpeg не найден в PATH. Установите ffmpeg и перезапустите.")

    def _estimate_secs(self, text: str) -> float:
        # грубо: ~13 символов/сек
        return max(1.0, len(text) / 13.0)

    async def synth(self, text: str, ref_audio: Path, ref_text: str, vid: str, out_format: Literal["wav", "mp3", "ogg"] = "wav") -> bytes:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Поле 'text' пустое или отсутствует")

        try:
            return await asyncio.wait_for(
                self._synth_api(gen_text=text.strip(), ref_audio=ref_audio,
                                out_format=out_format, ref_text=ref_text)) #vid=vid


        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail=f"Генерация превысила лимит {settings.TTS_MAX_SECONDS} с")


    async def _synth_api(
            self,
            gen_text: str,
            ref_audio: Path,
            ref_text: str,
            out_format: str
    ) -> bytes:

        print(f"😀 ref_text: {ref_text},\n"
              f"ref_audio: {ref_audio},\n"
              f"gen_text: {gen_text},\n"
              f"out_format: {out_format}")
        # --- ИНФЕРЕНС ЧЕРЕЗ API (БЕЗ CLI) ---
        wav_np, sr, _spec = await asyncio.to_thread(self._F5TTS.infer,
            ref_audio,
            ref_text,
            gen_text,
            nfe_step=settings.TTS_NFE_STEPS
        )


        # сохраняем в память
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, wav_np, sr, format="WAV")
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