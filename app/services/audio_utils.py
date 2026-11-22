# app/services/audio_utils.py
import numpy as np
import soundfile as sf
import torch, torchaudio
import io
from typing import Union
from pathlib import Path
from core.config import SAMPLE_RATE


# F-020, user decision 2026-11-22 (option B): torchaudio 2.9+ reads and writes
# audio only through torchcodec + FFmpeg, which the requirements do not ship.
# File IO goes through soundfile (libsndfile: WAV/FLAC/OGG/MP3); torchaudio is
# kept for resampling only.
def _read_audio(source) -> tuple[torch.Tensor, int]:
    """Read a file path or a file-like object as a float32 [C,N] tensor."""
    data, sr = sf.read(source, dtype="float32", always_2d=True)  # [N,C]
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)


def save_wav_pcm16(path: Union[str, Path], audio, sample_rate: int) -> None:
    """Write mono audio (1D or [1,N], tensor or array) as 16-bit PCM WAV."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(str(path), np.clip(samples, -1.0, 1.0), int(sample_rate), subtype="PCM_16", format="WAV")



def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    """[C,N] -> [1,N] усреднением по каналам, если нужно."""
    # Handle 1D tensors by adding a channel dimension
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # [N] -> [1,N]
    elif wav.dim() != 2:
        raise ValueError(f"Ожидается тензор [C,N] или [N], получен {wav.dim()}D, shape={tuple(wav.shape)}")
    
    # Convert to mono if stereo
    return wav if wav.size(0) == 1 else wav.mean(dim=0, keepdim=True)

def _resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    return wav if src_sr == dst_sr else torchaudio.transforms.Resample(src_sr, dst_sr)(wav)

def load_and_resample(src: Union[str, Path, bytes, np.ndarray], *, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Загружает аудио и приводит к: mono, target_sr, float32 ∈ [-1..1].
    Поддерживает:
      - str | Path : путь к файлу (WAV/FLAC/OGG/MP3 — что умеет soundfile/libsndfile)
      - bytes      : байты файла (контейнерный формат, не «сырой» PCM)
      - np.ndarray : 1D/2D массив float32/float64, интерпретируется как моно/стерео @ target_sr
    Возврат: 1D np.float32 длиной N_target.
    """
    # 1) загрузка источника
    if isinstance(src, (str, Path)):
        wav, sr = _read_audio(str(src))            # [C,N] float32
    elif isinstance(src, (bytes, bytearray, memoryview)):
        wav, sr = _read_audio(io.BytesIO(bytes(src)))  # [C,N] float32
    elif isinstance(src, np.ndarray):
        # считаем, что это уже «числа» @ target_sr
        arr = np.asarray(src)
        if arr.ndim == 1:
            wav = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0)  # [1,N]
        elif arr.ndim == 2:
            wav = torch.from_numpy(arr.astype(np.float32, copy=False))               # [C,N]
        else:
            raise ValueError(f"np.ndarray должен быть 1D/2D, получено {arr.ndim}D")
        sr = target_sr
    else:
        raise TypeError("src должен быть Path|str|bytes|np.ndarray")

    # 2) к float32, моно, ресэмпл
    if wav.dtype != torch.float32:
        wav = wav.to(torch.float32)
    wav = _to_mono(wav)                       # [1,N]
    wav = _resample(wav, sr, target_sr)       # [1,M]

    # 3) → 1D float32 [-1..1]
    audio = wav.squeeze(0).contiguous().cpu().numpy().astype(np.float32)
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio


def ensure_pcm16_mono_16k(src: Union[str, Path, bytes]) -> bytes:
    """
    Универсальная: файл/байты → float моно 16k → PCM16 LE bytes.
    Удобно, когда нужен «правильный» WAV/WS-поток.
    """
    audio = load_and_resample(src)  # 1D float32 @16k
    x = np.asarray(audio, dtype=np.float32, order="C")
    np.nan_to_num(x, copy=False)
    np.clip(x, -1.0, 1.0, out=x)
    x_int16 = (x * 32767.0).astype(np.int16, copy=False)
    return x_int16.tobytes(order="C")

def pcm16_bytes_to_float1d(pcm: bytes) -> np.ndarray:
    """PCM16 LE → 1D float32 [-1..1] (частота задаётся снаружи)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    x *= (1.0 / 32767.0)
    return x

def ensure_float_mono_16k_from_pcm16(pcm: bytes, src_sr: int, channels: int = 1) -> np.ndarray:
    """
    Сырые PCM16 (любой SR/каналы) → 1D float32 @ 16k моно.
    Если каналов >1 — сведём попарно усреднением; затем ресэмплим.
    """
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    t = torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0)  # [1,N]
    t = _resample(t, src_sr, SAMPLE_RATE)       # [1,M]

    y = t.squeeze(0).contiguous().cpu().numpy().astype(np.float32)
    np.clip(y, -1.0, 1.0, out=y)


    return y

