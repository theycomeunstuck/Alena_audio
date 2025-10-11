# core/tts/tts_engine.py
from __future__ import annotations
import asyncio, tempfile
from pathlib import Path
from typing import Literal
from fastapi import HTTPException
from pydub import AudioSegment
from pydub.utils import which
from app import settings

class TtsEngine:
    def __init__(self):
        self.ckpt = settings.F5TTS_CKPT_PATH
        self.vocoder = settings.F5TTS_VOCODER_NAME
        self.vocoder_ckpt = settings.F5TTS_VOCODER_CKPT
        self.sample_rate = settings.TTS_SAMPLE_RATE
        self.nfe = settings.TTS_NFE_STEPS
        self.max_sec = settings.TTS_MAX_SECONDS
        self.device = self._resolve_device(settings.DEVICE)

        if not which("ffmpeg"):
            raise RuntimeError("FFmpeg не найден в PATH. Установите ffmpeg и перезапустите.")

        if not self.ckpt:
            raise RuntimeError("Не задан F5TTS_CKPT_PATH (путь к .pt/.safetensors).")

    def _resolve_device(self, dev: str) -> str:
        d = dev.lower()
        if d == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return d  # 'cpu', 'cuda', 'cuda:0'...

    def _estimate_secs(self, text: str) -> float:
        # грубо: ~13 символов/сек
        return max(1.0, len(text) / 13.0)

    async def synth(self, text: str, ref_audio: Path, out_format: Literal["wav","mp3","ogg"]="wav") -> bytes:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Поле 'text' пустое или отсутствует")
        if self._estimate_secs(text) > self.max_sec * 1.6:
            raise HTTPException(status_code=400, detail="Слишком длинный текст для лимита 25с")

        try:
            return await asyncio.wait_for(
                self._synth_cli(text.strip(), ref_audio, out_format),
                timeout=self.max_sec
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Генерация превысила лимит 25с")

    async def _synth_cli(self, text: str, ref_audio: Path, out_format: str) -> bytes:
        tmpdir = Path(tempfile.mkdtemp(prefix="f5tts_"))
        out_dir = tmpdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "f5-tts_infer-cli",
            "--model", "F5-TTS",
            "--ref_audio", str(ref_audio),
            "--ref_text", "",                  # пусто -> автотранскриб референса
            "--gen_text", text,
            "--output_dir", str(out_dir),
            "--vocoder_name", self.vocoder,
            "--nfe", str(self.nfe),
            "--ckpt_file", self.ckpt,
            "--language", "ru",                # Explicitly set language to Russian
            "--sample_rate", str(self.sample_rate),  # Ensure correct sample rate
            "--device", self.device,           # Explicitly set device
            "--batch_size", "1",               # Process one at a time for stability
        ]
        if self.vocoder_ckpt:
            cmd += ["--vocoder_ckpt", self.vocoder_ckpt]
        
        # Set environment variables for better stability
        import os
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0" if self.device.startswith("cuda") else "",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        })

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"F5-TTS CLI error: {stderr.decode(errors='ignore')[:4000]}")

        wavs = list(out_dir.glob("*.wav"))
        if not wavs:
            raise HTTPException(status_code=500, detail="F5-TTS не вернул аудио")
        wav_path = wavs[0]

        if out_format == "wav":
            return wav_path.read_bytes()

        audio = AudioSegment.from_wav(wav_path)
        out_path = out_dir / f"out.{out_format}"
        if out_format == "mp3":
            audio.export(out_path, format="mp3", bitrate="192k")
        elif out_format == "ogg":
            audio.export(out_path, format="ogg")
        else:
            raise HTTPException(status_code=400, detail="Неверный формат аудио")
        return out_path.read_bytes()
