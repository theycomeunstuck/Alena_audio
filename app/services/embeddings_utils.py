# app/services/embedding_utils.py
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
from speechbrain.inference.speaker import EncoderClassifier



# Поднимем один раз энкодер для референс-проверки
_ENCODER: Optional[EncoderClassifier] = None

def get_encoder() -> EncoderClassifier:
    """Загружает (один раз) модель SpeechBrain encoder"""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(
                (Path(__file__).resolve().parents[2] / "pretrained_models" / "SpeechBrain" / "spkrec-ecapa-voxceleb")
            ),
        ).eval()
    return _ENCODER


def to_tensor_1d(x: np.ndarray) -> torch.Tensor:
    if not isinstance(x, np.ndarray):
        raise ValueError("ожидался np.ndarray")
    if x.ndim != 1:
        raise ValueError(f"ожидался 1D массив, получено shape={x.shape}")
    if np.allclose(x, 0.0, atol=1e-7):
        raise ValueError("пустой/нулевой сигнал")
    return torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0)


def embed_speechbrain(x: np.ndarray) -> torch.Tensor:
    """Создаёт L2-нормализованный эмбеддинг для аудиосигнала"""
    enc = get_encoder()
    wav = to_tensor_1d(x)
    with torch.no_grad():
        emb = enc.encode_batch(wav)
        emb = emb.squeeze()
        emb = F.normalize(emb, p=2, dim=-1)
    return emb