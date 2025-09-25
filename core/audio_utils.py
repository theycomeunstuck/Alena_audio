# audio_utils.py
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from core.config import SAMPLE_RATE, TARGET_DBFS


def normalize_rms(y, target_dBFS: float = TARGET_DBFS):
    """
    Нормализация RMS: поддерживает и numpy.ndarray, и torch.Tensor.
    """
    # приведение к нужному типу
    if isinstance(y, np.ndarray):
        rms = np.sqrt(np.mean(y**2))
        target = 10**(target_dBFS / 20)
        scale = target / (rms + 1e-9) if rms > 0 else 1.0
        y_norm = y * scale
        return np.clip(y_norm, -1.0, 1.0)

    elif isinstance(y, torch.Tensor):
        rms = torch.sqrt(torch.mean(y**2))
        target = 10 ** (target_dBFS / 20)
        scale = target / (rms + 1e-9) if rms > 0 else torch.tensor(1.0, device=y.device)
        y_norm = y * scale
        return y_norm.clamp(-1.0, 1.0)

    else:
        raise TypeError(f"Unsupported type {type(y)} for normalize_rms")


