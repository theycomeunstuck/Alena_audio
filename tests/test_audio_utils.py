# tests/test_audio_utils.py
import numpy as np
import torchaudio
import torch

from app.services.audio_utils import load_and_resample
from core.config import SAMPLE_RATE


def test_load_and_resample_stereo_to_mono(tmp_path):
    sr_in = 44100
    dur = 0.3

    # Левый канал — синус 440 Гц, правый — тишина -> в моно ампл. станет ~в 2 раза меньше (среднее)
    t = torch.linspace(0, dur, int(sr_in * dur), dtype=torch.float32, requires_grad=False)[:-1]
    left = 0.8 * torch.sin(2 * np.pi * 440.0 * t)
    right = torch.zeros_like(left)
    stereo = torch.stack([left, right], dim=0)  # [2, N]

    in_wav = tmp_path / "stereo.wav"
    torchaudio.save(str(in_wav), stereo, sr_in, format="wav")

    y = load_and_resample(str(in_wav), target_sr=SAMPLE_RATE)

    assert y.ndim == 1
    assert y.dtype == np.float32
    # длина около dur * SAMPLE_RATE
    assert abs(len(y) - int(dur * SAMPLE_RATE)) <= 2

    # амплитуда ~ в 2 раза меньше исходной левой (0.8 -> ~0.4)
    peak = float(np.max(np.abs(y)))
    assert 0.25 <= peak <= 0.6  # широкая «физическая» граница, учитывая фильтры ресэмплера


def test_load_and_resample_down_and_up(tmp_path):
    # проверим, что 48k -> 16k и 8k -> 16k работают и возвращают 1D float32
    for sr_in in (48000, 8000):
        t = torch.linspace(0, 0.2, int(sr_in * 0.2), dtype=torch.float32)[:-1]
        sig = 0.5 * torch.sin(2 * np.pi * 220.0 * t)
        in_wav = tmp_path / f"in_{sr_in}.wav"
        torchaudio.save(str(in_wav), sig.unsqueeze(0), sr_in, format="wav")

        y = load_and_resample(str(in_wav), target_sr=SAMPLE_RATE)
        assert y.ndim == 1
        assert y.dtype == np.float32
        assert abs(len(y) - int(0.2 * SAMPLE_RATE)) <= 2
