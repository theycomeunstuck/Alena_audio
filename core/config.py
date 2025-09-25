# config.py
import warnings, os
from pathlib import Path
if True:
    # Корень проекта = родитель каталога core/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Единая папка для весов/кэшей
    MODELS_DIR = PROJECT_ROOT / "pretrained_models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # единая папка для эмбеддингов пользователей
    EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


    # Опционально: стандартизируем кэши HF/torch (чтобы всё летело сюда же)
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(MODELS_DIR / ".cache"))

    # Отключаем предупреждения Whisper, PyTorch, SpeechBrain
    warnings.filterwarnings("ignore", category=UserWarning, message=".*FP16 is not supported on CPU; using FP32 instead.*", module="whisper")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Module 'speechbrain.pretrained' was deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
    warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"^speechbrain\.utils\.parameter_transfer$")
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning, module="torch._utils")
    warnings.filterwarnings("ignore", message=".*symlinks on Windows.*", category=UserWarning, module="speechbrain.utils.parameter_transfer")
    warnings.filterwarnings("ignore", message=".*SYMLINK strategy on Windows.*", category=UserWarning, module="speechbrain.utils.fetching")





import whisper
from torch.cuda import is_available
from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.inference.speaker import SpeakerRecognition


# ——— ПАРАМЕТРЫ —————————————————————————————————————————————————————
SAMPLE_RATE         = 16000
VAD_AGGR_MODE       = 1         # от 0 (мягко) до 3 (агрессивно) (voice activity detection)
FRAME_MS            = 30        # размер VAD-фрейма в мс
SPK_WINDOW_S        = 3         # размер окна для верификации, сек
STEP_S              = 1         # шаг сдвига окна, сек.  должно быть целым числом
MIN_VOICE_RATIO     = 0.5       # минимальная доля реальные речи в окне для ASR. # Параметры гейтинга ASR:
MAX_ASR_FAILURES    = 5         # необязательный: макс. подряд «фоновых» окон до сброса. # Параметры гейтинга ASR:
TARGET_DBFS         = -18.0     # dBFS для RMS-нормализации
TRAIN_USER_VOICE_S  = 15        # Длительность записи эталона
sim_threshold = 0.6     # Пороговое значение совпадения (уверенность) пользователя по косинусному расстоянию

# ====== ASR (Whisper) ======
ASR_LANGUAGE = "ru"          # язык по умолчанию
ASR_WINDOW_SEC = 8.0         # сколько секунд держим в буфере (StreamingASRSession)
ASR_EMIT_SEC = 2.0           # как часто выдаём partial


REFERENCE_FILE      = "reference.npy"
REFERENCE_FILE_WAV  = "reference.wav"

device = "cuda" if is_available() else "cpu"

if device == "cuda":
    WHISPER_MODEL = "turbo"  # tiny, base, small, medium, turbo, large
else:
    WHISPER_MODEL = "small"

# ————————————————————————————————————————————————————————————————————

# Загрузка модели ASR
asr_model = whisper.load_model(WHISPER_MODEL, device=device)

noise_Model = separator.from_hparams(
    source="speechbrain/sepformer-dns4-16k-enhancement",
    savedir=MODELS_DIR / "SpeechBrain" / "sepformer-dns4-16k-enhancement",
    run_opts={"device":device}).eval()


'''
в API pipeline сейчас используется только энкодер от spkrec-ecapa-voxceleb. 
Расположение app/services/speaker_service.py (def _get_encoder)
'''
speech_verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=MODELS_DIR / "SpeechBrain" / "spkrec-ecapa-voxceleb",
    run_opts={"device":device}).eval()



