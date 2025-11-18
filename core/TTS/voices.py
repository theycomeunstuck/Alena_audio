# core/TTS/voices.py
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
import uuid, shutil, json, hashlib
from typing import Optional
from pydub import AudioSegment, silence  # требует ffmpeg

from core.ASR.transcriber import AsrTranscriber
from core.config import VOICES_DIR

class VoiceMeta(BaseModel):
    voice_id: str
    sr: int
    orig_file: str
    ref_text: str = ""  # транскрипция референс-аудио

class VoiceStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)


    def _voice_dir(self, voice_id: str) -> Path:
        return self.root / voice_id

    def meta_path(self, voice_id: str) -> Path:
        return self._voice_dir(voice_id) / "meta.json"

    def _cache_path(self) -> Path:
        return self.root / "whisper_transcriptions.json"

    def _load_cache(self) -> dict:
        p = self._cache_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: dict) -> None:
        p = self._cache_path()
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")



    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def exists(self, voice_id: str) -> bool:
        return self._voice_dir(voice_id).exists()

    def read_meta(self, voice_id: str) -> VoiceMeta:
        p = self.meta_path(voice_id)
        if not p.exists():
            return VoiceMeta(voice_id=voice_id, sr=0, orig_file="", ref_text="")
        return VoiceMeta.model_validate_json(p.read_text(encoding="utf-8"))

    def write_meta(self, meta: VoiceMeta) -> None:
        self.meta_path(meta.voice_id).write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    def ensure_reference_wav(self, voice_id: str) -> Path:
        vdir = self._voice_dir(voice_id)
        ref = vdir / "reference.wav"
        if not ref.exists():
            raise FileNotFoundError(f"reference.wav не найден для '{voice_id}'. \nref: {ref} \nvdir: {vdir}'")
        return ref # вовзращает путь к ref.wav

    def _sha1(self, path: Path) -> str:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def _transcribe_with_local_cache(self, wav_path: Path, language: Optional[str] = None) -> str:
        """
        Локальный кэш транскрипций в VOICES_DIR/whisper_transcriptions.json.
        Использует openai/whisper (через AsrTranscriber).
        """
        cache = self._load_cache()
        sha1 = self._sha1(wav_path)
        by_sha = cache.setdefault("by_sha1", {})
        if sha1 in by_sha:
            return by_sha[sha1]

        transcriber = AsrTranscriber(language=language)
        text = transcriber.transcribe(wav_path)
        text = (text or "").strip()
        by_sha[sha1] = text
        cache.setdefault("by_path", {})[str(wav_path)] = text
        self._save_cache(cache)
        return text

    def clone_from_upload(self, up_path: Path, sample_rate: int = 24000) -> VoiceMeta:
        """
        Создаёт новый voice_id из загруженного WAV/MP3/OGG и заполняет meta.json.
        Дублирование транскрипций избегается за счёт локального кэша в VOICES_DIR.
        """
        up_path = Path(up_path)
        voice_id = str(uuid.uuid4().hex)
        vdir = self._voice_dir(voice_id)
        vdir.mkdir(parents=True, exist_ok=True)

        # сконвертировать в reference.wav с нужным sample_rate
        audio = AudioSegment.from_file(up_path) # загрузка
        audio = audio.set_channels(1).set_frame_rate(sample_rate)

        cleaned = self._clean_reference_audio(audio) # обрезка аудио, если оно больше 12 секунд

        # сохраняем trimmed reference.wav
        ref_path = vdir / "reference.wav"
        audio.export(ref_path, format="wav")


        # meta
        ref_text = self._transcribe_with_local_cache(ref_path)
        meta = VoiceMeta(voice_id=voice_id, sr=int(sample_rate), orig_file=str(up_path.name), ref_text=ref_text)




        self.write_meta(meta)
        return meta

    def _clean_reference_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Повторяет логику оригинального F5-TTS preprocess_ref_audio_text:
        1) Поиск длинных тишин — попытка ограничить до 12 секунд
        2) Фолбэк — короткие тишины
        3) Жёсткое ограничение в 12s
        4) Удаление тишины с краёв
        """

        # 1. Попытка найти длинные тишины
        non_silent_segs = silence.split_on_silence(
            audio,
            min_silence_len=1000,
            silence_thresh=-50,
            keep_silence=1000,
            seek_step=10
        )

        non_silent_wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 12000:
                print("Audio is over 12s, clipping short. (1)")
                break
            non_silent_wave += seg

        # 2. fallback — короткие тишины, если всё еще > 12s
        if len(non_silent_wave) > 12000:
            non_silent_segs = silence.split_on_silence(
                audio,
                min_silence_len=100,
                silence_thresh=-40,
                keep_silence=1000,
                seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 12000:
                    print("Audio is over 12s, clipping short. (2)")
                    break
                non_silent_wave += seg

        # 3. Жёсткое отсечение, если все равно > 12 секунд
        if len(non_silent_wave) > 12000:
            print("Audio is over 12s, clipping short. (3)")
            non_silent_wave = non_silent_wave[:12000]

        # 4. Удаление тишины с краёв
        non_silent_wave = self._remove_silence_edges(non_silent_wave)

        # +50ms
        non_silent_wave += AudioSegment.silent(duration=50)

        return non_silent_wave

    def _remove_silence_edges(self, audio, silence_threshold=-42):
        # Удаляем тишину спереди
        start = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
        audio = audio[start:]

        # Удаляем тишину с конца
        end = len(audio)
        for ms in reversed(audio):
            if ms.dBFS > silence_threshold:
                break
            end -= 1
        return audio[:end]


# convenience factory для маршрутизатора
def get_default_store() -> VoiceStore:
    return VoiceStore(VOICES_DIR)
