# config.py
import warnings
import whisper
from torch.cuda import is_available
from speechbrain.inference.separation import SepformerSeparation as separator

# ——— ПАРАМЕТРЫ —————————————————————————————————————————————————————
SAMPLE_RATE         = 16000
NOISE_DURATION      = 2        # сек для профиля шума
VAD_AGGR_MODE       = 1       # от 0 (мягко) до 3 (агрессивно) (voice activity detection)
FRAME_MS            = 30       # размер VAD-фрейма в мс
SPK_WINDOW_S        = 3        # размер окна для верификации, сек
STEP_S              = 1       # шаг сдвига окна, сек.  должно быть целым числом
MIN_VOICE_RATIO     = 0.5         # минимальная доля реальные речи в окне для ASR. # Параметры гейтинга ASR:
MAX_ASR_FAILURES    = 4          # необязательный: макс. подряд «фоновых» окон до сброса. # Параметры гейтинга ASR:
TARGET_DBFS         = -18.0       # dBFS для RMS-нормализации
REFERENCE_FILE      = "reference.npy"
REFERENCE_FILE_WAV  = "reference.wav"

WHISPER_MODEL       = "small"      # tiny, base, small, medium, turbo, large
device = "cuda" if is_available() else "cpu"

if device == "cuda":
    WHISPER_MODEL = "turbo"  # tiny, base, small, medium, turbo, large

# ————————————————————————————————————————————————————————————————————

# Загрузка модели ASR
asr_model = whisper.load_model(WHISPER_MODEL, device=device)
noise_Model = separator.from_hparams(
    source="speechbrain/sepformer-dns4-16k-enhancement",
    savedir="pretrained_models/sepformer-dns4-16k-enhancement",
    run_opts={"device":device}).eval()


# Отключаем предупреждения Whisper и PyTorch
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*FP16 is not supported on CPU; using FP32 instead.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*weights_only=False.*"
)
