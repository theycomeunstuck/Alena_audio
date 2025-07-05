# audio_enhancement.py
import torch, torchaudio
import torch.nn.functional as F
import numpy as np
from config import (device, noise_Model, speech_verification_model,
                    SAMPLE_RATE, TARGET_DBFS, )
from audio_utils import normalize_rms


#todo: SB реализовать через класс, чтобы можно было отлавливать ошибки и не было трижды повторения. тут код одинаковый

MIN_WAV_SAMPLES = int(0.9 * SAMPLE_RATE) # заглушка. надо, чтобы стриминг эмбеддинги были достаточной длины





def handle_exceptions(func):
    """
    Декоратор для обработки исключений в методах класса Enhancement.
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
    else:
        raise TypeError(f"{__file__} должен быть np.ndarray или torch.Tensor, получен {type(x)}")

    if t.dim() == 1: t = t.unsqueeze(0)
    # дополняем нулями, если слишком коротко

    if pad_to_min == True:
        if t.shape[1] < MIN_WAV_SAMPLES:
            pad = MIN_WAV_SAMPLES - t.shape[1]
            t = F.pad(t, (0, pad))
    return t.to(device, dtype=torch.float32)

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
        if self.audio_ref is None:
            raise ValueError("audio_ref не задан для верификации")

        with torch.no_grad():
            audio_emb = speech_verification_model.encode_batch(self.audio)
            audio_emb = audio_emb.squeeze(1) #torch.Size([1, 192])

        # Фиксация несовпадения размерностей (256->192)
        if audio_emb.shape[1] != self.audio_ref.shape[1]:
            target_dim = min(audio_emb.shape[1], self.audio_ref.shape[1], 192)
            audio_emb = audio_emb[:, :target_dim]
            ref_emb = self.audio_ref[:, :target_dim]

        emb1 = F.normalize(audio_emb, p=2, dim=1)  # [1, 192]
        emb2 = F.normalize(ref_emb, p=2, dim=1)
        score_t = F.cosine_similarity(emb1, emb2)

        return score_t[0].item()



