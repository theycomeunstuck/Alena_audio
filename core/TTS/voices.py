# core/tts/voices.py
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
import uuid, shutil
from pydub import AudioSegment  # нужен ffmpeg

class VoiceMeta(BaseModel):
    voice_id: str
    sr: int
    orig_file: str

class VoiceStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _voice_dir(self, voice_id: str) -> Path:
        return self.root / voice_id

    def exists(self, voice_id: str) -> bool:
        return (self._voice_dir(voice_id) / "reference.wav").exists()

    def ensure_default(self, default_id: str = "_default"):
        d = self._voice_dir(default_id); d.mkdir(parents=True, exist_ok=True)
        ref = d / "reference.wav"
        if not ref.exists():
            raise FileNotFoundError(
                f"Нет базового голоса: {ref}. Положите сюда reference.wav "
                f"(используется, если voice_id не передан)."
            )

    def clone_from_upload(self, up_path: Path, sample_rate: int = 24000) -> VoiceMeta:
        voice_id = str(uuid.uuid4())
        vdir = self._voice_dir(voice_id)
        vdir.mkdir(parents=True, exist_ok=True)

        audio = AudioSegment.from_file(up_path)
        audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)

        ref_path = vdir / "reference.wav"
        audio.export(ref_path, format="wav")

        meta = VoiceMeta(voice_id=voice_id, sr=sample_rate, orig_file=up_path.name)
        (vdir / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")



        return meta

    def reference_wav(self, voice_id: str) -> Path:
        p = self._voice_dir(voice_id) / "reference.wav"
        if not p.exists():
            raise FileNotFoundError(f"Голос {voice_id} не найден")
        return p
