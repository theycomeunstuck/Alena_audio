# core/audio_enhancement.py
import torch, torchaudio
import torch.nn.functional as F
import numpy as np

from app.services.audio_utils import load_and_resample
from core.config import device, SAMPLE_RATE, TARGET_DBFS, noise_Model
from core.audio_utils import normalize_rms


MIN_WAV_SAMPLES = int(0.9 * SAMPLE_RATE) # заглушка. надо, чтобы стриминг эмбеддинги были достаточной длины

def handle_exceptions(func):
    """
    Декоратор для обработки исключений в методах классов
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print(f"[ERROR] Ошибка в методе '{func.__name__}': {e}")
            raise
    return wrapper

def to_tensor(x, pad_to_min=False) -> torch.Tensor:
    # 1) Приводим np.ndarray → Tensor, или проверяем, что это уже Tensor
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif torch.is_tensor(x):
        t = x
    else: raise TypeError(f"{__file__} должен быть np.ndarray или torch.Tensor, получен {type(x)}")

    if t.dim() == 1: t = t.unsqueeze(0)
    if pad_to_min == True: # дополняем нулями, если слишком короткая запись
        if t.shape[1] < MIN_WAV_SAMPLES:
            pad = MIN_WAV_SAMPLES - t.shape[1]
            t = F.pad(t, (0, pad))
    return t.to(device, dtype=torch.float32)

def to_numpy(audio) -> np.ndarray:
    # Подготовка numpy-массива из тензора
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    elif isinstance(audio, np.ndarray):
        audio_np = audio
    else: raise TypeError(f"{__file__} должен быть np.ndarray или torch.Tensor, получен {type(audio)}")
    return audio_np

class Audio_Enhancement:
    """
    Класс для обработки аудио: шумоподавление и нормализация громкости.
    Автоматически конвертирует вход в тензор и обрабатывает ошибки.
    Сравнение голосов выполняет app/services/speaker_service.verify_files
    (audit F-067: прежний speech_verification был мёртвым кодом).
    """

    @handle_exceptions
    def __init__(self, audio):
        # Конвертация входного сигнала с защитой от ошибок
        self.audio = to_tensor(audio, pad_to_min=True)

    @handle_exceptions
    def noise_suppression(self) -> np.ndarray:
        """
        Применяет SepFormer для подавления шума и нормализует RMS.
        Возвращает: улучшенный аудиосигнал на CPU, 1D-тензор.
        """
        with torch.inference_mode():
            est_sources = noise_Model.separate_batch(self.audio) # Разделяем на источники: [speech, noise]
        enhanced = est_sources[:, :].detach().cpu().squeeze()
        enhanced = normalize_rms(enhanced, target_dBFS=TARGET_DBFS)
        return enhanced

