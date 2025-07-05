import os
from huggingface_hub import snapshot_download
import sounddevice as sd
from bark import SAMPLE_RATE, generate_audio, preload_models


class BarkTTS:
    """
    Класс для синтеза речи с помощью модели bark-small от Suno.
    Пример использования:
        tts = BarkTTS(cache_dir='./bark_small')
        tts.speak("Hello, world!")
    """

    def __init__(self, repo_id: str = "suno/bark-small", cache_dir: str = "./bark_small"):
        # Отключаем загрузку только весов
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

        # Скачиваем модель
        self.local_dir = snapshot_download(
            repo_id="suno/bark-small",
            repo_type="model",
            cache_dir="./bark_small"
        )

        # Предзагружаем модели для ускорения работы
        preload_models(self.local_dir)

    def synthesize(self, text: str):
        """
        Синтезирует аудио из текста и возвращает numpy-массив с сэмплами.
        :param text: Входная строка текста для синтеза.
        :return: numpy.ndarray с аудио-сэмплами.
        """
        return generate_audio(text)

    def to_voice(self, text: str):
        """
        Синтезирует речь и сразу воспроизводит её через звуковое устройство.

        :param text: Входная строка текста для воспроизведения.
        """
        audio = self.synthesize(text)
        sd.play(audio, SAMPLE_RATE)
        sd.wait()


if __name__ == "__main__":
    # Пример использования
    tts = BarkTTS(cache_dir='./bark_small')
    example_text = """
    Приветствую. Данный текст сгенерирован для учебного помощника "Алёна".
    Рада первой встрече! С чего начнём? Быть может, Вы можете предложить выпить мне чая? [laughs]
    """

    tts.to_voice(example_text)