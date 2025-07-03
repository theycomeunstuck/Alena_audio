# config.py
import warnings
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
similarity_threshold = 0.75     # Пороговое значение совпадения (уверенность) пользователя по косинусному расстоянию
REFERENCE_FILE      = "reference.npy"
REFERENCE_FILE_WAV  = "reference.wav"

WHISPER_MODEL       = "small"      # tiny, base, small, medium, turbo, large
device = "cuda" if is_available() else "cpu"

if device == "cuda":
    WHISPER_MODEL = "turbo"  # tiny, base, small, medium, turbo, large

# ————————————————————————————————————————————————————————————————————

# Загрузка модели ASR
asr_model = whisper.load_model(WHISPER_MODEL, device=device)

# noise_Model = separator.from_hparams(
#     source="speechbrain/sepformer-dns4-16k-enhancement",
#     savedir="pretrained_models/sepformer-dns4-16k-enhancement",
#     run_opts={"device":device}).eval()

speech_verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device":device}).eval()




noise_Model = separator.from_hparams(
    source="speechbrain/sepformer-whamr-enhancement",
    savedir="pretrained_models/sepformer-whamr-enhancement",
    run_opts={"device":device, "mask_threshold": 0.8}).eval()
'''

the solution with whamr rn is quite bad cause low sim and asr is much worse than origin file
mask_threshold Действительно роляет, но качество всё равно очень плачевное.
Плюс whamr работает с 8к частотой, что, видимо, уже слишком сильно портит звук.
Обычный wham себя тоже показал с недостойной стороны
 
'''

# print(noise_Model)


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
