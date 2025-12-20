# app/services/embedding_utils.py
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Union
from speechbrain.inference.speaker import EncoderClassifier
from core.config import device


# Поднимем один раз энкодер для референс-проверки
_ENCODER: Optional[EncoderClassifier] = None

def get_encoder() -> EncoderClassifier:
    """Загружает (один раз) модель SpeechBrain encoder. Singleton"""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(
                (Path(__file__).resolve().parents[2] / "pretrained_models" / "SpeechBrain" / "spkrec-ecapa-voxceleb")
            ),
            run_opts={"device": device}
        ).to(device).eval()
    return _ENCODER


def to_tensor_1d(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    # 1) Превращаем во float32 numpy 1D
    if isinstance(x, torch.Tensor):
        x = x.detach().to('cpu').numpy()
    elif not isinstance(x, np.ndarray):
        try:
            x = np.asarray(x)
        except Exception:
            raise ValueError(f"ожидался np.ndarray-подобный объект, получено {type(x)}")

    # 2) Если целочисленный сигнал — нормируем в [-1, 1]
    if np.issubdtype(x.dtype, np.integer):
        max_abs = np.float32(np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / max_abs
    else:
        x = x.astype(np.float32, copy=False)

    # Приводим к 1D: (N,1)->(N,), (2,N)->моно средним
    if x.ndim == 2:
        if 1 in x.shape:
            x = x.reshape(-1)
        else:
            # считаем последнюю ось каналами
            if x.shape[0] in (1, 2) and x.shape[1] > 2:
                x = x.mean(axis=0)
            else:
                x = x.mean(axis=-1)
    if x.ndim != 1:
        raise ValueError(f"ожидался 1D массив, получено shape={x.shape}")


    if np.allclose(x, 0.0, atol=1e-7) or not np.isfinite(x).all(): # пустой или нулевой | сигнал содержит NaN/Inf
        return None



    return torch.from_numpy(x).unsqueeze(0).to(device)   # [1, T]

def embed_speechbrain(x: np.ndarray) -> torch.Tensor:
    """Создаёт L2-нормализованный эмбеддинг для аудиосигнала - нормализует по громкости"""
    enc = get_encoder()
    wav = to_tensor_1d(x) # to.device(cuda|cpu)
    if wav is None: # Если пришёл пустой сигнал
        return None

    model_device = next(enc.parameters()).device
    wav = wav.to(model_device)

    with torch.inference_mode():
        emb = enc.encode_batch(wav)
        emb = emb.squeeze()
        emb = F.normalize(emb, p=2, dim=-1)
    return emb