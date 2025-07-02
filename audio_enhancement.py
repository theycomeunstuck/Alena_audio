# audio_enhancement.py
import torch, torchaudio
import numpy as np
from config import device, noise_Model, speech_separation_model, speech_verification_model
from audio_utils import normalize_rms


#todo: SB реализовать через класс, чтобы можно было отлавливать ошибки и не было трижды повторения. тут код одинаковый


def noise_suppresion_SB(audio: np.ndarray) -> np.ndarray:
    """
    Улучшение речи с помощью SepFormer.
    signal: 1D np.float32-массив, диапазон [-1, 1], fs=16 000 З
    возвращает: улучшенный сигнал того же формата.
    """
    try:

        Wav = torch.from_numpy(audio).unsqueeze(0).to(device)  # Переводим в тензор (batch=1)
        with torch.no_grad(): # do not math grad and save memory
            # Разделяем на источники: [speech, noise]
            est_sources = noise_Model.separate_batch(Wav)  # (#batch, source, samples) на устройстве device

        enhanced = est_sources[:, :].detach().cpu().squeeze()
        # 4) Подгоняем RMS-уровень
        enhanced = normalize_rms(enhanced, target_dBFS=-25)

        return enhanced
    except Exception as e:
        print(f"[ERROR] Ошибка вызвана в файле {__file__} \n\n{e}")
        raise Exception



def speech_verification_SB(audio: np.array) -> np.ndarray:
    pass
    # score, prediction = speech_verification_model.verify_files("/content/example1.wav", "/content/example2.flac")


'''
    Временно отказано от speech separation. probably i should change the model (not speechbrain)
    cause of quality and underwater stones (recognize always (2,3) speakers and solution via 
    cosinus formulation -> bruh, so why should i use this separation?
'''
# def speech_separation_SB(audio, speaker=0) -> torch.Tensor:
#     '''Разделение аудио по speaker-id. Хорошо работает до 3 голосов включительно'''
#
#     # speech_separation_model
#     try:
#         if isinstance(audio, torch.Tensor):
#             audio_np = audio.detach().cpu().numpy()
#         elif isinstance(audio, np.ndarray):
#             audio_np = audio
#         else:
#             raise TypeError(f"{__file__} Unsupported audio type: {type(audio)}")
#
#         if audio_np.ndim > 1:
#             audio_np = audio_np.reshape(-1)
#
#         audio_np = audio_np.astype(np.float32)  # Добавляем батч
#         Wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # (1, N)
#         with torch.no_grad():
#             est_sources = speech_separation_model.separate_batch(Wav)
#
#         enhanced = est_sources[:, :].detach().cpu().squeeze()  # n
#         return enhanced  # torch.Size([samples, speakerCount])
#
#     except Exception as e:
#         print(f"[ERROR] Ошибка вызвана в файле {__file__} \n\n{e}")
#         raise