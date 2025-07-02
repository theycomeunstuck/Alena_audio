# audio_enhancement.py
import torch, torchaudio
import numpy as np
import soundfile as sf
# from speechbrain.inference.VAD import VAD
from config import device, noise_Model
from audio_utils import normalize_rms




count = 0

def noise_suppresion_SB(audio: np.ndarray) -> np.ndarray:
    """
    Улучшение речи с помощью SepFormer.
    signal: 1D np.float32-массив, диапазон [-1, 1], fs=16 000 З
    возвращает: улучшенный сигнал того же формата.
    """
    try:
        # Переводим в тензор (batch=1)
        Wav = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, N)

        # 2) Отключаем градиенты, чтобы не тратить память и не считать бэкап для backward
        with torch.no_grad():
            # 3) SepFormer делает всю работу на GPU, потому что и модель, и вход на cuda
            # Разделяем на источники: [speech, noise]
            est_sources = noise_Model.separate_batch(Wav)  # (#batch, source, samples) на устройстве device

        # Берём первый источник (речь)
        enhanced = est_sources[:, :].detach().cpu().squeeze()

        # 4) Подгоняем RMS-уровень
        enhanced = normalize_rms(enhanced, target_dBFS=-25)
        global count
        sf.write(f'wav/enhanced_torch{count}.wav', enhanced, 16000)
        # torchaudio.save(f'wav/enhanced_torch{count}.wav', enhanced, 16000, encoding='PCM_S', bits_per_sample=16)
        count += 1
        return enhanced
    except Exception as e:
        print(f"[ERROR] Ошибка вызвана в файле {__file__} \n\n{e}")
        raise Exception

