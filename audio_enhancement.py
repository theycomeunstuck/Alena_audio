# audio_enhancement.py
import torch, torchaudio
import numpy as np
from speechbrain.inference.separation import SepformerSeparation as separator
from config import device
from audio_utils import normalize_rms

# 1. Загружаем предобученную модель
model = separator.from_hparams(
    source="speechbrain/sepformer-dns4-16k-enhancement",
    savedir="pretrained_models/sepformer-dns4-16k-enhancement",
    run_opts={"device":device}
)

# 2. Разделяем аудиофайл на источники (здесь — «усиленная» речь и «остаточный шум»)
# est_sources = model.separate_file(
#     path="speechbrain/sepformer-dns4-16k-enhancement/example_dns4-16k.wav"
# )


def noise_suppresion_SB(signal: np.ndarray) -> np.ndarray:
    """
    Улучшение речи с помощью SepFormer.
    signal: 1D np.float32-массив, диапазон [-1, 1], fs=16 000 З
    возвращает: улучшенный сигнал того же формата.
    """

    # Переводим в тензор (batch=1)
    wav = torch.from_numpy(signal).unsqueeze(0).to(device)  # (1, N)

    # 2) Отключаем градиенты, чтобы не тратить память и не считать бэкап для backward
    with torch.no_grad():
        # 3) SepFormer делает всю работу на GPU, потому что и модель, и вход на cuda
        # Разделяем на источники: [speech, noise]
        est_sources = model.separate_batch(wav)  # (#batch, source, samples) на устройстве device

    # Берём первый источник (речь)
    enhanced = est_sources[0, 0, :].detach().cpu().numpy()

    # 4) Подгоняем RMS-уровень
    enhanced = normalize_rms(enhanced)

    return enhanced
