# Transcribe.py
# -*- coding: utf-8 -*-
"""
Быстрая транскрибация с faster-whisper + (опционально) пунктуация и RUAccent для русского.
Плюс режим подготовки датасета из TSV (Common Voice и т.п.), где делаем только:
- конвертацию/копирование аудио
- расстановку ударений (пунктуация уже расставлена)

Единый флаг TSV_FLAG выбирает режим работы:
- TSV_FLAG = False -> обычная транскрипция (как раньше)
- TSV_FLAG = True  -> режим "prepareTSV": без транскрипции, только ударения + подготовка датасета
"""

import os
import sys
import csv
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from threading import Lock
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pydub import AudioSegment

# =========================
#          CONFIG
# =========================

USE_GPU = True   # True = GPU, False = CPU
print("GPU using\n" if USE_GPU else "CPU using\n")

# Режим работы:
TSV_FLAG = True  # True -> использовать режим подготовки из TSV (без Whisper)

# Опционально: включить/выключить восстановление пунктуации и ударений
USE_PUNCTUATION = True   # используется только в режиме транскрипции
USE_RUACCENT    = True


# ---------- Конфиг для обычной транскрипции ----------
AUDIO_DIR  = r"E:\Speech_Datasets\SOVA\RuDevicesAudioBooks"
OUTPUT_CSV = r"E:\PycharmProjects\AI-Teach\Speech\libs\F5-TTS\data\f5-tts_ru_char\metadata.csv"
ERROR_LOG  = r"E:\PycharmProjects\AI-Teach\Speech\libs\F5-TTS\data\f5-tts_ru_char\errors.log"

# Модель whisper: tiny / base / small / medium / large-v2 / large-v3 / distil-* и т.д.
MODEL_NAME   = "turbo"
LANGUAGE     = "ru"       # "ru" для русского, None — автоопределение
CHUNK_LENGTH = 30
MAX_WORKERS  = 8
MAX_WORKERS_COPYING  = 4
WHISPER_BATCH_SIZE = 32

BEAM_SIZE   = 1  # 1 = жадный декодинг, максимально быстро
TEMPERATURE = 0.0
VAD_FILTER  = False  # Voice activity detector
print(f'use_gpu: {USE_GPU}; USE_PUNCTUATION: {USE_PUNCTUATION} \nUSE_RUACCENT: {USE_RUACCENT}; VAD_FILTER: {VAD_FILTER} \nTSV_FLAG: {TSV_FLAG}')

# ---------- Конфиг для режима TSV (пример для Common Voice) ----------
CV_DIR = Path(
    r"D:\Speech_Datasets\Common Voice\cv-corpus-22.0-2025-06-20-ru\cv-corpus-22.0-2025-06-20\ru"
)

TSV_FILES = [
    "train.tsv",
    "dev.tsv",
    "test.tsv",
    "validated.tsv",
    "other.tsv",
]

CLIPS_DIR = CV_DIR / "clips"

TSV_OUT_DIR = Path(r"E:\PycharmProjects\AudioAPI\F5-TTS\data\ru_Sova_CV_char")
TSV_WAVS_DIR = TSV_OUT_DIR / "wavs"
TSV_METADATA_CSV = TSV_OUT_DIR / "metadata.csv"
TSV_PROGRESS_FILE = TSV_OUT_DIR / "progress.txt"

TARGET_SR = 22050
TSV_BATCH_SIZE = 64

# =========================
# Принудительно задаём режим до импортов моделей пунктуации / RUAccent
# =========================

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # полностью скрыть GPU

# Для Punctuation.py и RUAccent — они прочитают этот флаг при импорте
os.environ["APP_FORCE_DEVICE"] = "cuda" if USE_GPU else "cpu"

from Punctuation import punctuation as punct_restore  # noqa: E402
from ruaccent import RUAccent  # noqa: E402

# Потокобезопасность вызовов моделей
_punct_lock  = Lock()
_accent_lock = Lock()


# =========================
#        ОБЩИЕ УТИЛИТЫ
# =========================

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


def init_model() -> Tuple[WhisperModel, BatchedInferencePipeline]:
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
        # cpu_threads=os.cpu_count()-2,
    )

    batched = BatchedInferencePipeline(model=model)
    return model, batched


def init_accentizer() -> Optional[RUAccent]:
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


def postprocess_text(
    text: str,
    is_russian: bool,
    accentizer: Optional[RUAccent],
    use_punctuation: bool,
) -> str:
    """
    Общая постобработка текста:
    - опционально пунктуация
    - опционально RUAccent
    """
    if not is_russian:
        return text

    # 1) Пунктуация (если нужна)
    if use_punctuation:
        try:
            with _punct_lock:
                text = punct_restore(text)
        except Exception:
            # не роняем пайплайн
            pass

    # 2) Ударения
    if USE_RUACCENT and accentizer is not None:
        try:
            with _accent_lock:
                if hasattr(accentizer, "process_all"):
                    text = accentizer.process_all(text)
                else:
                    text = accentizer.process(text)
        except Exception:
            pass

    return text


def copy_batch(paths_to_copy: List[Path], dst_dir: Path, max_workers=1):
    """
    Копирует список файлов в dst_dir, сохраняя метаданные (copy2).
    max_workers=1 — чаще всего оптимален для одного диска.
    """
    if not paths_to_copy:
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    def _copy_one(src: Path):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return str(dst)

    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=max_workers) as ex:
            list(ex.map(_copy_one, paths_to_copy))
    else:
        for p in paths_to_copy:
            _copy_one(p)


# =========================
#   ПАЙПЛАЙН ТРАНСКРИПЦИИ
# =========================

def transcribe_one(
    batched: BatchedInferencePipeline,
    wav_path: Path,
    lang_hint: str,
    accentizer: Optional[RUAccent],
) -> Tuple[str, str]:
    """
    Возвращает (filename, text) — с уже применённой пунктуацией+ударениями для RU.
    """
    segments, info = batched.transcribe(
        str(wav_path),
        language=lang_hint,
        beam_size=BEAM_SIZE,
        temperature=TEMPERATURE,
        vad_filter=VAD_FILTER,
        chunk_length=CHUNK_LENGTH,
        batch_size=WHISPER_BATCH_SIZE,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    text = "".join(seg.text for seg in segments).strip()

    # Определяем, русский ли текст
    detected_lang = info.language  # ISO-639-1
    is_russian = (lang_hint == "ru") or (lang_hint is None and detected_lang == "ru")

    text = postprocess_text(
        text=text,
        is_russian=is_russian,
        accentizer=accentizer,
        use_punctuation=USE_PUNCTUATION,
    )

    return wav_path.name.replace("\\", "/"), text


def run_transcribe_pipeline(accentizer: Optional[RUAccent]):
    audio_root = Path(AUDIO_DIR)
    csv_path = Path(OUTPUT_CSV)
    err_path = Path(ERROR_LOG)

    audio_files = list_audio_files(audio_root)
    if not audio_files:
        print(f"[ERR] Не найдено аудиофайлов в: {audio_root}")
        sys.exit(1)

    # Резюмирование по готовому metadata.csv
    done_list = load_done_set(csv_path)
    todo = [p for p in audio_files if p.name.lower() not in done_list]
    print(f"[INFO] Всего файлов: {len(audio_files)} | Уже готово: {len(done_list)} | К обработке: {len(todo)}")

    model, batched = init_model()

    # Папка проекта для копирования wav (рядом с metadata.csv)
    project_dir: Optional[Path] = None
    try:
        project_dir_candidate = csv_path.parent / "wavs"
        if audio_root.resolve() != project_dir_candidate.resolve():
            project_dir_candidate.mkdir(parents=True, exist_ok=True)
            project_dir = project_dir_candidate
        else:
            print("[INFO] AUDIO_DIR совпадает с папкой проекта /wavs, копирование отключено.")
    except Exception as e:
        print(f"[WARN] Не удалось подготовить папку проекта для копирования wav: {e}")
        project_dir = None

    # Открываем CSV в режиме добавления
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.writer(csv_file, delimiter="|")

    # Лог ошибок
    err_file = err_path.open("a", encoding="utf-8")

    start = time.time()
    n_ok, n_bad = 0, 0
    to_copy_buffer: List[Path] = []

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

                # периодически сбрасываем на диск
                if (n_ok + n_bad) % 500 == 0:
                    csv_file.flush()

                # буфер на копирование в проект
                if project_dir is not None:
                    to_copy_buffer.append(path)
                    if len(to_copy_buffer) >= 500:
                        copy_batch(to_copy_buffer, project_dir, max_workers=MAX_WORKERS_COPYING)
                        to_copy_buffer.clear()

            except Exception as e:
                n_bad += 1
                err_file.write(f"{path} :: {repr(e)}\n")
                if (n_ok + n_bad) % 500 == 0:
                    err_file.flush()

    # докопировать хвост
    if project_dir is not None and to_copy_buffer:
        copy_batch(to_copy_buffer, project_dir, max_workers=MAX_WORKERS_COPYING)
        to_copy_buffer.clear()

    csv_file.close()
    err_file.close()

    dur = time.time() - start
    print(f"\n[DONE] Успешно: {n_ok} | Ошибок: {n_bad} | Время: {dur/60:.1f} мин "
          f"| Скорость: { (n_ok+n_bad)/max(dur,1):.2f} файлов/с")


# =========================
#    TSV / prepareTSV MODE
# =========================

def convert_audio(task) -> bool:
    """
    Функция для multiprocessing:
    task = (mp3_path_str, wav_path_str)

    Возвращает:
        True  - если конвертация успешна или файл уже есть
        False - если ошибка
    """
    mp3_path_str, wav_path_str = task
    mp3_path = Path(mp3_path_str)
    wav_path = Path(wav_path_str)

    try:
        # Если WAV уже есть — не тратим время
        if wav_path.exists():
            return True

        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
        audio.export(wav_path, format="wav")
        return True
    except Exception as e:
        print(f"[ERROR] Не удалось конвертировать {mp3_path}: {e}")
        return False


def load_tsv_progress() -> int:
    """Возвращает индекс, с которого нужно продолжить обработку."""
    if TSV_PROGRESS_FILE.exists():
        try:
            value = int(TSV_PROGRESS_FILE.read_text(encoding="utf-8").strip())
            print(f"[INFO] Прогресс из файла: start_index = {value}")
            return value
        except Exception as e:
            print(f"[WARN] Не удалось прочитать {TSV_PROGRESS_FILE}: {e}")
            return 0
    return 0


def save_tsv_progress(idx: int):
    """Сохраняет индекс, ДО которого всё успешно обработано (следующий к запуску)."""
    try:
        TSV_PROGRESS_FILE.write_text(str(idx), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Не удалось записать {TSV_PROGRESS_FILE}: {e}")


def gather_all_tsv_rows() -> List[Tuple[str, str]]:
    """
    Читает все TSV_FILES, собирает (mp3_path, sentence).
    Удаляет дубликаты по пути к аудио.
    """
    all_rows: List[Tuple[str, str]] = []

    for tsv_name in TSV_FILES:
        tsv_path = CV_DIR / tsv_name
        if not tsv_path.exists():
            print(f"[WARN] Файл не найден: {tsv_path}")
            continue

        print(f"[INFO] Читаю {tsv_name} ...")

        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sentence = row.get("sentence")
                if not sentence:
                    continue

                mp3_path = CLIPS_DIR / row["path"]
                if not mp3_path.exists():
                    continue

                all_rows.append((str(mp3_path), sentence.strip()))

    print(f"[INFO] Всего строк найдено: {len(all_rows)}")

    # Удаляем дубликаты по аудиофайлам
    unique = {}
    for mp3_path_str, text in all_rows:
        unique[mp3_path_str] = text

    all_rows = list(unique.items())
    print(f"[INFO] После удаления дубликатов: {len(all_rows)}")

    # Стабильная сортировка по пути, чтобы порядок был детерминированным
    all_rows.sort(key=lambda x: x[0])

    return all_rows


def process_tsv_batch(
    indices,
    mp3_paths,
    wav_paths,
    texts,
    pool: Pool,
    f_out,
    accentizer: Optional[RUAccent],
):
    """
    Обработка одного батча TSV:
      - параллельная конвертация аудио (CPU, multiprocessing)
      - последовательная расстановка ударений (GPU/CPU)
      - запись metadata.csv
      - сохранение прогресса
    """
    if not indices:
        return

    # 1. Параллельная конвертация
    tasks = list(zip(mp3_paths, wav_paths))
    results = pool.map(convert_audio, tasks)

    # 2. Обработка текста + запись
    for success, idx, mp3_path_str, wav_path_str, raw_text in zip(
        results, indices, mp3_paths, wav_paths, texts
    ):
        if not success:
            # аудио не сконвертировалось — пропускаем эту строку
            continue

        # пунктуация уже есть в TSV, ставим только ударения
        processed_text = postprocess_text(
            text=raw_text,
            is_russian=True,
            accentizer=accentizer,
            use_punctuation=False,
        )
        wav_name = Path(wav_path_str).name
        f_out.write(f"{wav_name}|{processed_text}\n")

    # На всякий случай — флешим данные
    f_out.flush()

    # 3. Сохраняем прогресс: следующий индекс после последнего в батче
    last_index = max(indices)
    save_tsv_progress(last_index + 1)


def run_tsv_pipeline(accentizer: Optional[RUAccent]):
    TSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    TSV_WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # Собираем все строки из TSV
    all_rows = gather_all_tsv_rows()
    total = len(all_rows)
    if total == 0:
        print("[ERROR] Нет данных для обработки TSV.")
        return

    # Читаем прогресс
    start_index = load_tsv_progress()

    # Синхронизируем с уже существующим metadata.csv
    if TSV_METADATA_CSV.exists():
        with open(TSV_METADATA_CSV, "r", encoding="utf-8") as f:
            existing_lines = sum(1 for _ in f)
        if existing_lines > start_index:
            print(
                f"[INFO] В metadata.csv уже {existing_lines} строк. "
                f"Сдвигаю стартовый индекс с {start_index} до {existing_lines}"
            )
        start_index = max(start_index, existing_lines)
    else:
        existing_lines = 0

    # Страхуемся, чтобы не выйти за пределы массива
    if start_index >= total:
        print("[INFO] Похоже, всё уже обработано.")
        print(f"[INFO] Всего строк: {total}, start_index: {start_index}")
        print(f"WAV файлы: {TSV_WAVS_DIR}")
        print(f"metadata.csv: {TSV_METADATA_CSV}")
        print(f"progress.txt: {TSV_PROGRESS_FILE}")
        return

    start_index = min(start_index, total)
    print(f"[INFO] Итоговый start_index: {start_index} / {total}")

    # Настройка пула процессов
    num_workers = max(1, cpu_count() - 6)
    print(f"[INFO] Использую {num_workers} процессов для аудио.")

    # Основной цикл
    with Pool(processes=num_workers) as pool, open(
        TSV_METADATA_CSV, "a", encoding="utf-8"
    ) as f_out:

        batch_indices = []
        batch_mp3_paths = []
        batch_wav_paths = []
        batch_texts = []

        for i, (mp3_path_str, text) in tqdm(
            enumerate(all_rows),
            total=total,
            desc="Конвертация и обработка TSV",
        ):
            # Пропускаем уже обработанную часть
            if i < start_index:
                continue

            wav_name = f"cv_{i:07d}.wav"
            wav_path = TSV_WAVS_DIR / wav_name

            batch_indices.append(i)
            batch_mp3_paths.append(mp3_path_str)
            batch_wav_paths.append(str(wav_path))
            batch_texts.append(text)

            # Если батч набрался — обрабатываем
            if len(batch_indices) >= TSV_BATCH_SIZE:
                process_tsv_batch(
                    batch_indices,
                    batch_mp3_paths,
                    batch_wav_paths,
                    batch_texts,
                    pool,
                    f_out,
                    accentizer,
                )
                batch_indices.clear()
                batch_mp3_paths.clear()
                batch_wav_paths.clear()
                batch_texts.clear()

        # Обработка хвостика (если остались не полные батчи)
        if batch_indices:
            process_tsv_batch(
                batch_indices,
                batch_mp3_paths,
                batch_wav_paths,
                batch_texts,
                pool,
                f_out,
                accentizer,
            )

    print("=====================================")
    print("        ГОТОВО! ДАТАСЕТ СОБРАН       ")
    print("=====================================")
    print(f"Всего строк: {total}")
    print(f"WAV файлы:  {TSV_WAVS_DIR}")
    print(f"metadata.csv: {TSV_METADATA_CSV}")
    print(f"progress.txt: {TSV_PROGRESS_FILE}")


# =========================
#           MAIN
# =========================

def main():
    accentizer = init_accentizer()

    if TSV_FLAG:
        print("[INFO] Запуск в режиме TSV (prepareTSV): без Whisper, только ударения + подготовка датасета (копирование из datasetFolder -> f5ttsProject/wavs).")
        run_tsv_pipeline(accentizer)
    else:
        print("[INFO] Запуск в режиме транскрипции Whisper.")
        run_transcribe_pipeline(accentizer)


if __name__ == "__main__":
    main()
