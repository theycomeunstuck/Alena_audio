# audio_enhancement.py
import torch, torchaudio
import torch.nn.functional as F
import numpy as np
from config import (device, noise_Model, speech_verification_model,
                    similarity_threshold, SAMPLE_RATE)
from audio_utils import normalize_rms


#todo: SB реализовать через класс, чтобы можно было отлавливать ошибки и не было трижды повторения. тут код одинаковый

MIN_WAV_SAMPLES = int(0.9 * SAMPLE_RATE) # заглушка. надо, чтобы стриминг эмбеддинги были достаточной длины
def to_tensor(x, fill_to_Min_Wav_Samples=0) -> torch.Tensor:
    # 1) Приводим np.ndarray → Tensor, или проверяем, что это уже Tensor
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif torch.is_tensor(x):
        t = x
    else:
        raise TypeError(f"{__file__} должен быть np.ndarray или torch.Tensor, получен {type(x)}")
    # 2) Добавляем batch-ось, если надо
    if t.dim() == 1:
        t = t.unsqueeze(0)
    # дополняем нулями, если слишком коротко ?
    if fill_to_Min_Wav_Samples == 1:
        if t.shape[1] < MIN_WAV_SAMPLES:
            pad = MIN_WAV_SAMPLES - t.shape[1]
            t = F.pad(t, (0, pad))
    return t.to(device, dtype=torch.float32)

def noise_suppresion_SB(audio: np.ndarray) -> np.ndarray:
    """
    Улучшение речи с помощью SepFormer.
    signal: 1D np.float32-массив, диапазон [-1, 1], fs=16 000 З
    возвращает: улучшенный сигнал того же формата.
    """
    try:
        resampler_to8k = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=8000).to(device)
        resampler_toTARGET = torchaudio.transforms.Resample(orig_freq=8000, new_freq=SAMPLE_RATE).to(device)

        # Применяем ресэмплинг
        Wav_8k = torch.from_numpy(audio).unsqueeze(0).to(device)  # Переводим в тензор (batch=1)
        Wav_8k = resampler_to8k(Wav_8k)
        with torch.no_grad(): # do not math grad and save memory
            # Разделяем на источники: [speech, noise]
            est_sources = noise_Model.separate_batch(Wav_8k)  # (#batch, source, samples) на устройстве device

        enhanced = est_sources[:, :].detach().cpu().squeeze()
        # 4) Подгоняем RMS-уровень
        enhanced = normalize_rms(enhanced, target_dBFS=-25)
        return enhanced
    except Exception as e:
        print(f"[ERROR] Ошибка вызвана в файле {__file__} \n\n{e}")
        raise Exception

def speech_verification_SB(audio: np.array, audio_ref: np.ndarray) -> np.ndarray:
    try:
        #todo: жалуется на различие (audio: wav, audio_ref: emb) -> should be (wav, wav)
        audio_t = to_tensor(audio, 1)
        ref_t = to_tensor(audio_ref)

        with torch.no_grad():
            audio_emb = speech_verification_model.encode_batch(audio_t)
            audio_emb = audio_emb.squeeze(1) #torch.Size([1, 192])
            ref_emb = ref_t
            # L2-нормализация
            emb1 = F.normalize(audio_emb, p=2, dim=1) # [1, 192]
            emb2 = F.normalize(ref_emb, p=2, dim=1)


            score_t = F.cosine_similarity(emb1, emb2)  # shape [1]
            pred_t = (score_t >= similarity_threshold).float()  # shape [1]

        score = score_t[0].item()
        pred = bool(pred_t[0].item())

        return score, pred

    except Exception as e:
        print(f"[ERROR] Ошибка вызвана в файле {__file__}")
        raise Exception

