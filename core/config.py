# config.py
from __future__  import annotations
import warnings, os
from pathlib import Path
if True:
    # Корень проекта = родитель каталога core/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Единая папка для весов/кэшей
    MODELS_DIR = PROJECT_ROOT / "pretrained_models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # единая папка для эмбеддингов пользователей (зарегистрованные пользователи, которых будем верифицировать по голосу)
    EMBEDDINGS_DIR = PROJECT_ROOT / "storage" / "embeddings"
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # папка для .wav пользователей (зарегистрованные пользователи, которых будем верифицировать по голосу)
    EMBEDDINGS_WAV_DIR = PROJECT_ROOT / "storage" / "embeddings_wav"
    EMBEDDINGS_WAV_DIR.mkdir(parents=True, exist_ok=True)

    # единая папка для клонированных голосов
    VOICES_DIR = PROJECT_ROOT / "storage" / "voices_TTS"
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Опционально: стандартизируем кэши HF/torch (чтобы всё летело сюда же)
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(MODELS_DIR / ".cache"))

    # Отключаем предупреждения Whisper, PyTorch, SpeechBrain
    warnings.filterwarnings("ignore", category=UserWarning, message=".*FP16 is not supported on CPU; using FP32 instead.*", module="whisper")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Module 'speechbrain.pretrained' was deprecated.*")
    # Audit F-066: the torch weights_only=False (pickle loading) FutureWarning stays visible.
    warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"^speechbrain\.utils\.parameter_transfer$")
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning, module="torch._utils")
    warnings.filterwarnings("ignore", message=".*symlinks on Windows.*", category=UserWarning, module="speechbrain.utils.parameter_transfer")
    warnings.filterwarnings("ignore", message=".*SYMLINK strategy on Windows.*", category=UserWarning, module="speechbrain.utils.fetching")





import threading

from torch.cuda import is_available


# ПАРАМЕТРЫ
SAMPLE_RATE         = 16000
VAD_AGGR_MODE       = 1         # от 0 (мягко) до 3 (агрессивно) (voice activity detection)
FRAME_MS            = 30        # размер VAD-фрейма в мс
SPK_WINDOW_S        = 3         # размер окна для верификации, сек
STEP_S              = 1         # шаг сдвига окна, сек.  должно быть целым числом; в API недействительно, вроде
MIN_VOICE_RATIO     = 0.5       # минимальная доля реальные речи в окне для ASR. # Параметры гейтинга ASR:
MAX_ASR_FAILURES    = 5         # необязательный: макс. подряд «фоновых» окон до сброса. # Параметры гейтинга ASR:
TARGET_DBFS         = -18.0     # dBFS для RMS-нормализации
TRAIN_USER_VOICE_S  = 15        # Длительность записи эталона
sim_threshold = 0.65     # Пороговое значение совпадения (уверенность) пользователя по косинусному расстоянию

# ASR (Whisper)
ASR_LANGUAGE = None          # язык по умолчанию
ASR_WINDOW_SEC = 8.0         # сколько секунд держим в буфере (StreamingASRSession)
ASR_EMIT_SEC = 2.0           # как часто выдаём partial



REFERENCE_FILE      = "_Old/reference.npy"
REFERENCE_FILE_WAV  = "reference.wav"

device = "cuda" if is_available() else "cpu"

if device == "cuda":
    WHISPER_MODEL = "turbo"  # tiny, base, small, medium, turbo, large
else:
    WHISPER_MODEL = "small"

# ————————————————————————————————————————————————————————————————————

# Audit F-019: the heavy models (Whisper, SepFormer, ECAPA) used to be built
# here at module level, so `import core.config` (and therefore
# `import app.main`) paid the full model load on every boot/test/reload run.
# They are now lazy singletons: builders + accessor functions below, built on
# first use. The old module-level names (`asr_model`, `noise_Model`,
# `speech_verification_model`) are kept as lazy proxies so existing
# `from core.config import asr_model` importers keep working unchanged.
_model_build_lock = threading.Lock()


def _build_asr_model():
    import whisper  # lazy: the heavy package is only needed on first build
    return whisper.load_model(WHISPER_MODEL, device=device)


def _build_noise_model():
    from speechbrain.inference.separation import SepformerSeparation as separator
    return separator.from_hparams(
        source="speechbrain/sepformer-dns4-16k-enhancement",
        savedir=MODELS_DIR / "SpeechBrain" / "sepformer-dns4-16k-enhancement",
        run_opts={"device":device}).eval()


def _build_speech_verification_model():
    from speechbrain.inference.speaker import SpeakerRecognition
    # NOTE (audit F-019): this copy is only consumed by
    # Audio_Enhancement.speech_verification, which has no live call sites; the
    # live ECAPA encoder is app/services/embeddings_utils.get_encoder().
    # With the lazy singleton it stays unloaded until someone calls it.
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=MODELS_DIR / "SpeechBrain" / "spkrec-ecapa-voxceleb",
        run_opts={"device":device}).eval()


_asr_model = None


def get_asr_model():
    """Whisper ASR model; built on first use (audit F-019)."""
    global _asr_model
    if _asr_model is None:
        with _model_build_lock:
            if _asr_model is None:
                _asr_model = _build_asr_model()
    return _asr_model


_noise_Model = None


def get_noise_model():
    """SepFormer enhancement model; built on first use (audit F-019)."""
    global _noise_Model
    if _noise_Model is None:
        with _model_build_lock:
            if _noise_Model is None:
                _noise_Model = _build_noise_model()
    return _noise_Model


_speech_verification_model = None


def get_speech_verification_model():
    """ECAPA SpeakerRecognition copy; built on first use (audit F-019)."""
    global _speech_verification_model
    if _speech_verification_model is None:
        with _model_build_lock:
            if _speech_verification_model is None:
                _speech_verification_model = _build_speech_verification_model()
    return _speech_verification_model


class _LazyModel:
    """Compatibility proxy over a lazy model singleton.

    Builds the underlying model on the first attribute access and delegates
    everything to it. Setting attributes on the proxy (e.g. the
    `monkeypatch.setattr(config.asr_model, "transcribe", fake)` pattern used
    by tests/conftest.py) shadows the delegated attribute.
    """

    def __init__(self, builder):
        self._builder = builder
        self._obj = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        obj = self._obj
        if obj is None:
            with self._lock:
                if self._obj is None:
                    self._obj = self._builder()
                obj = self._obj
        return getattr(obj, name)


asr_model = _LazyModel(get_asr_model)
noise_Model = _LazyModel(get_noise_model)
speech_verification_model = _LazyModel(get_speech_verification_model)


try:
    from app import settings
    SAMPLE_RATE = getattr(settings, "ASR_SAMPLE_RATE", 16000)
except Exception:
    SAMPLE_RATE = 16000

# (Если нужен TTS_SAMPLE_RATE — он в app/settings: settings.TTS_SAMPLE_RATE = 24000)
