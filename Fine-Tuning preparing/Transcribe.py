# Transcribe.py
# -*- coding: utf-8 -*-
"""
Быстрая транскрибация с faster-whisper + (опционально) пунктуация и RUAccent для русского.
- Переключатель USE_GPU = True/False (одна переменная) для всего пайплайна
- Многопоточность
- Чанкинг длинных аудио
- VAD
- Резюмирование
- Логи ошибок
"""

import os
import sys
import csv
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from threading import Lock

from tqdm import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline

# =========================
#          CONFIG
# =========================

# ЕДИНЫЙ ТУМБЛЕР ДЛЯ ВСЕГО ПРОЕКТА:
USE_GPU = True   # True = GPU, False = CPU
print("GPU using\n" if USE_GPU else "CPU using\n")

# Опционально: включить/выключить восстановление пунктуации и ударений
USE_PUNCTUATION = True
USE_RUACCENT    = True

AUDIO_DIR  = r"E:\Speech_Datasets\SOVA\RuDevicesAudioBooks"
#Убедись, что metadata.csv отсутствует в папке; и что папка wavs существует
OUTPUT_CSV = r"E:\PycharmProjects\AI-Teach\Speech\libs\F5-TTS\data\f5-tts_ru_char\metadata.csv" #Папкой проекта считается та, что указана в данной var.
ERROR_LOG  = r"E:\PycharmProjects\AI-Teach\Speech\libs\F5-TTS\data\f5-tts_ru_char\errors.log"


# Модель whisper: tiny / base / small / medium / large-v2 / large-v3 / distil-* и т.д.
MODEL_NAME   = "turbo"   #| downloaded: meduium gpu, large-v2 cpu
LANGUAGE     = "ru"       # "ru" для русского, None — автоопределение
CHUNK_LENGTH = 30
MAX_WORKERS  = 8
MAX_WORKERS_COPYING  = 4
BATCH_SIZE = 32

BEAM_SIZE   = 1 # 1 = жадный декодинг, максимально быстро
TEMPERATURE = 0.0
VAD_FILTER  = False # Voice activity detector

PROJECT_DIR = ""
if (AUDIO_DIR.split("\\")[:-1] != "wavs"): #.../wavs/...mp3
    PROJECT_DIR = ("/".join(OUTPUT_CSV.split("\\")[:-1]))+"/wavs"


# =========================
# Принудительно задаём режим до импортов моделей пунктуации
# =========================
# Чтобы и HuggingFace pipeline, и др. модули не трогали GPU при USE_GPU=False:
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # полностью скрыть GPU
# Для Punctuation.py — он прочитает этот флаг при импорте
os.environ["APP_FORCE_DEVICE"] = "cuda" if USE_GPU else "cpu"

# Теперь можно импортировать пунктуацию и RUAccent
from Punctuation import punctuation as punct_restore  # noqa: E402
from ruaccent import RUAccent  # noqa: E402

# Потокобезопасность вызовов моделей
_punct_lock  = Lock()
_accent_lock = Lock()


def list_audio_files(root: Path, exts=(".wav", ".mp3", ".flac", ".m4a", ".ogg")) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return files


def load_done_set(csv_path: Path) -> set:
    done = set()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if not row or not row[0]:
                    continue
                done.add(Path(row[0]).name.lower())
    return done


def init_model() -> WhisperModel:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    device = "cuda" if (USE_GPU and has_cuda) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"  # CPU: int8 экономит память

    model = WhisperModel(
        MODEL_NAME,
        device=device,
        compute_type=compute_type,
        # cpu_threads=os.cpu_count()-2,  # можно раскомментировать для CPU. надо протестить todo:
    )

    batched = BatchedInferencePipeline(model=model)
    return model, batched


def init_accentizer() -> RUAccent:
    if not USE_RUACCENT:
        return None
    acc = RUAccent()
    acc.load(
        omograph_model_size="turbo3.1",
        use_dictionary=True,
        custom_dict={},
        device=("cuda" if USE_GPU else "cpu"),
        workdir=None,
    )
    return acc


def copy_batch(paths_to_copy, dst_dir: Path, max_workers=1):
    """
    Копирует список файлов в dst_dir, сохраняя метаданные (copy2).
    max_workers=1 — чаще всего оптимален для одного диска.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    def _copy_one(src: Path):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return str(dst)
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_copy_one, paths_to_copy))
    else:
        for p in paths_to_copy:
            _copy_one(p)


def transcribe_one(
    batched: BatchedInferencePipeline,
    wav_path: Path,
    lang_hint: str,
    accentizer: RUAccent,
) -> Tuple[str, str]:
    """
    Возвращает (path_str, text) — с уже применённой пунктуацией+ударениями для RU.
    """
    segments, info = batched.transcribe(
        str(wav_path),
        language=lang_hint,
        beam_size=BEAM_SIZE,
        temperature=TEMPERATURE,
        vad_filter=VAD_FILTER,
        chunk_length=CHUNK_LENGTH,
        batch_size=BATCH_SIZE,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    text = "".join(seg.text for seg in segments).strip()

    # Определяем, русский ли текст
    detected_lang = info.language  # ISO-639-1
    is_russian = (lang_hint == "ru") or (lang_hint is None and detected_lang == "ru")

    if is_russian:
        if USE_PUNCTUATION:
            try:
                with _punct_lock:
                    text = punct_restore(text)
            except Exception:
                # тихий фоллбек — не срываем пайплайн
                pass

        # 2) Ударения
        if USE_RUACCENT and accentizer is not None:
            try:
                with _accent_lock:
                    # у разных версий: process_all или process
                    if hasattr(accentizer, "process_all"):
                        text = accentizer.process_all(text)
                    else:
                        text = accentizer.process(text)
            except Exception:
                pass


    # if PROJECT_DIR != "":
    #
    #
    #     shutil.copy2(wav_path, (os.path.join(PROJECT_DIR, wav_path.name)).replace("\\", "/"))


    return str(wav_path.name).replace("\\", "/"), text


def main():
    audio_root = Path(AUDIO_DIR)
    csv_path = Path(OUTPUT_CSV)
    err_path = Path(ERROR_LOG)

    audio_files = list_audio_files(audio_root)
    if not audio_files:
        print(f"[ERR] Не найдено аудиофайлов в: {audio_root}")
        sys.exit(1)

    # Резюмирование
    done_list = load_done_set(csv_path)
    todo = [p for p in audio_files if p.name.lower() not in done_list]
    print(f"[INFO] Всего файлов: {len(audio_files)} | Уже готово: {len(done_list)} | К обработке: {len(todo)}")

    model, batched = init_model()
    accentizer = init_accentizer()

    # Открываем CSV в режиме добавления
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.writer(csv_file, delimiter="|")

    # Лог ошибок
    err_file = err_path.open("a", encoding="utf-8")

    start = time.time()
    n_ok, n_bad = 0, 0
    to_copy_buffer = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(transcribe_one, batched, p, LANGUAGE, accentizer): p
            for p in todo
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Transcribing", unit="file"):
            path = futures[fut]
            try:
                p_str, text = fut.result()
                writer.writerow([p_str, text])
                n_ok += 1
                if (n_ok + n_bad) % 500 == 0:
                    csv_file.flush()

                if len(to_copy_buffer) >= 500:
                    copy_batch(to_copy_buffer, PROJECT_DIR, max_workers=MAX_WORKERS_COPYING)
                    to_copy_buffer.clear()

            except Exception as e:
                n_bad += 1
                err_file.write(f"{path} :: {repr(e)}\n")
                if (n_ok + n_bad) % 500 == 0:
                    err_file.flush()

    if to_copy_buffer:
        copy_batch(to_copy_buffer, PROJECT_DIR, max_workers=MAX_WORKERS_COPYING)
        to_copy_buffer.clear()

    csv_file.close()
    err_file.close()

    dur = time.time() - start
    print(f"\n[DONE] Успешно: {n_ok} | Ошибок: {n_bad} | Время: {dur/60:.1f} мин "
          f"| Скорость: { (n_ok+n_bad)/max(dur,1):.2f} файлов/с")


if __name__ == "__main__":
    main()
