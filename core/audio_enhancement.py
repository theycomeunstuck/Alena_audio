# audio_enhancement.py
import torch, torchaudio
import torch.nn.functional as F
import numpy as np

from app.services.audio_utils import load_and_resample
from core.config import (device, SAMPLE_RATE, TARGET_DBFS,
                    noise_Model, speech_verification_model,

                    )
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
    Класс для обработки аудио: шумоподавление и верификация по голосу.
    Автоматически конвертирует входы в тензоры и обрабатывает ошибки.
    """

    @handle_exceptions
    def __init__(self, audio, audio_ref=None):
        # Конвертация входных сигналов с защитой от ошибок
        self.audio = to_tensor(audio, pad_to_min=True)
        self.audio_ref = None
        if audio_ref is not None:
            self.audio_ref = to_tensor(audio_ref, pad_to_min=True)

    @handle_exceptions
    def noise_suppression(self) -> np.ndarray:
        """
        Применяет SepFormer для подавления шума и нормализует RMS.
        Возвращает: улучшенный аудиосигнал на CPU, 1D-тензор.
        """
        with torch.no_grad(): # do not math grad and save memory
            est_sources = noise_Model.separate_batch(self.audio) # Разделяем на источники: [speech, noise]
        enhanced = est_sources[:, :].detach().cpu().squeeze()
        enhanced = normalize_rms(enhanced, target_dBFS=TARGET_DBFS)
        return enhanced

    @handle_exceptions
    def speech_verification(self) -> np.ndarray:
        """
        Сравнивает self.audio и self.audio_ref по голосовым эмбеддингам.
        Возвращает (score: float, prediction: bool).
        """

        if self.audio_ref is None: #todo: исправить в будущем. тут должен быть поиск по голосам
            # print("core/audio_enhancement :: 86. audio_ref не задан для верификации. \nСтоит придумать решение для поиска похожего голоса. Сейчас используется файл одного пользователя (misha_20sec.wav)")
            # audio_ref = load_and_resample("misha_20sec.wav")
            # self.audio_ref = to_tensor(audio_ref, pad_to_min=True)

            raise ValueError("audio_ref не задан для верификации. "
                             "Передайте reference файл или настройте поиск по базе.")



        with torch.no_grad():
            audio_emb = speech_verification_model.encode_batch(self.audio)
            audio_emb = audio_emb.squeeze(1) #torch.Size([1, 192])

        # Фиксация несовпадения размерностей (256->192)
        if audio_emb.shape[1] != self.audio_ref.shape[1]:
            target_dim = min(audio_emb.shape[1], self.audio_ref.shape[1], 192)
            audio_emb = audio_emb[:, :target_dim]
            ref_emb = self.audio_ref[:, :target_dim]
        else:
            ref_emb = self.audio_ref

        emb1 = F.normalize(audio_emb, p=2, dim=1)  # [1, 192]
        emb2 = F.normalize(ref_emb, p=2, dim=1)
        score_t = F.cosine_similarity(emb1, emb2)

        return score_t[0].item()

