# core/tts/tts_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
from fastapi import HTTPException
from pydub import AudioSegment
from pydub.utils import which
from app import settings
import os, sys, shutil, subprocess, asyncio, tempfile, contextlib

from app.settings import STORAGE_DIR


class TtsEngine:
    def __init__(self):
        self.ckpt = settings.F5TTS_CKPT_PATH
        self.vocoder = settings.F5TTS_VOCODER_NAME
        self.vocoder_ckpt = settings.F5TTS_VOCODER_CKPT
        self.sample_rate = settings.TTS_SAMPLE_RATE
        self.nfe = settings.TTS_NFE_STEPS
        self.max_sec = settings.TTS_MAX_SECONDS
        self.device = settings.DEVICE
        self.vocab_file = settings.VOCAB_FILE_PATH

        if not which("ffmpeg"):
            raise RuntimeError("FFmpeg не найден в PATH. Установите ffmpeg и перезапустите.")

        if not self.ckpt:
            raise RuntimeError("Не задан F5TTS_CKPT_PATH (путь к .pt/.safetensors).")

    def _estimate_secs(self, text: str) -> float:
        # грубо: ~13 символов/сек
        return max(1.0, len(text) / 13.0)

    async def synth(self, text: str, ref_audio: Path, ref_text: str, vid: str, out_format: Literal["wav", "mp3", "ogg"] = "wav") -> bytes:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Поле 'text' пустое или отсутствует")
        if self._estimate_secs(text) > self.max_sec * 1.6:
            raise HTTPException(status_code=400, detail=f"Слишком длинный текст для лимита {self.max_sec} сек")

        try:
            return await asyncio.wait_for(
                self._synth_cli(gen_text=text.strip(), ref_audio=ref_audio, out_format=out_format, ref_text=ref_text, vid=vid),
                timeout=self.max_sec)


        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail=f"Генерация превысила лимит {settings.TTS_MAX_SECONDS} с")

    async def _synth_cli(self, gen_text: str, ref_audio: Path, out_format: str, ref_text: str, vid: str) -> bytes:
        out_dir = STORAGE_DIR / "out_TTS" / f"{vid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [ # избегать пробелов в пути
            "f5-tts_infer-cli",
            "--model", "F5TTS_v1_Base",
            "--ref_audio", str(ref_audio),
            "--ref_text", f'"{ref_text}"' or "",
            "--gen_text", f'"{gen_text}"',
            "--output_dir", str(out_dir),
            "--nfe", str(self.nfe),
            "--ckpt_file", self.ckpt,
            "--vocab_file", self.vocab_file,
            "--vocoder_name", self.vocoder,
            "--device", self.device,
        ]

        if self.vocoder_ckpt:
            cmd += ["--vocoder_ckpt", self.vocoder_ckpt]

        env = os.environ.copy()

        print(f"🔧 F5-TTS CLI command: {' '.join(cmd)}")

        # Проверка доступности бинаря
        if shutil.which(cmd[0]) is None:
            raise HTTPException(status_code=500, detail=f"Не найден исполняемый файл '{cmd[0]}' в PATH")

        # Параметры для синхронного запуска
        run_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
        # Чтобы не всплывало консольное окно на Windows
        try:
            run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except AttributeError:
            pass

        if sys.platform == "win32":
            #  Запуск CLI. На Windows всегда уходим в безопасный путь.
            completed = await asyncio.to_thread(subprocess.run, cmd, **run_kwargs)
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode


        else: # На *nix пробуем настоящий асинхронный subprocess
            try:
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
                stdout, stderr = await proc.communicate()
                returncode = proc.returncode
            except NotImplementedError: # Редкий случай: даже тут нет транспорта — откат к синхронному запуску
                completed = await asyncio.to_thread(subprocess.run, cmd, **run_kwargs)
                stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode

        if returncode != 0:
            error_msg = (stderr or b"").decode(errors="ignore")[:4000]
            print(f"❌ F5-TTS CLI error: {error_msg}")
            print(f"📝 F5-TTS CLI stdout: {(stdout or b'').decode(errors='ignore')[:1000]}")
            raise HTTPException(status_code=500, detail=f"F5-TTS CLI error: {error_msg}")

        # Забираем результат
        wavs = list(out_dir.glob("*.wav"))
        if not wavs:
            # Уберём пустую tmp-папку
            with contextlib.suppress(Exception):
                out_dir.rmdir()
            raise HTTPException(status_code=500, detail="F5-TTS не вернул аудио")
        wavs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        wav_path = wavs[0]

        # try:
        if out_format == "wav":
            bytes_data = wav_path.read_bytes()
            # удаляем исходный wav после чтения
            target_wav_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            return bytes_data

        audio = AudioSegment.from_wav(wav_path)
        out_path = out_dir / f"out.{out_format}"
        if out_format == "mp3":
            audio.export(out_path, format="mp3", bitrate="192k")
        elif out_format == "ogg":
            audio.export(out_path, format="ogg")
        else:
            raise HTTPException(status_code=400, detail="Неверный формат аудио")

        bytes_data = out_path.read_bytes()

        # Чистка: удаляем и исходный wav, и перекодированный файл
        out_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        target_wav_path.unlink(missing_ok=True)  # type: ignore[arg-type]

        return bytes_data

        # return out_path.read_bytes()

        # finally:
        #     # 6) Уборка временной папки CLI
        #     with contextlib.suppress(Exception):
        #         for p in out_dir.glob("*"):
        #             p.unlink(missing_ok=True)  # type: ignore[arg-type]
        #         out_dir.rmdir()
