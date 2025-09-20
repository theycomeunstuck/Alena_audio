<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Audio Core API — Документация</title>
<style>
  :root{
    --bg:#0e1320;
    --panel:#141a2a;
    --ink:#e8eefc;
    --muted:#9fb0d3;
    --accent:#6ea8fe;
    --ok:#3ddc97;
    --warn:#ffce57;
    --err:#ff6b6b;
    --code:#0b1020;
    --chip:#202945;
    --border:#253053;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font:16px/1.6 Inter,system-ui,-apple-system,Segoe UI,Roboto,"Helvetica Neue",Arial,"Noto Sans","Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol"}
  a{color:var(--accent);text-decoration:none}
  a:hover{text-decoration:underline}
  header{
    position:sticky;top:0;z-index:10;background:linear-gradient(180deg,var(--bg),rgba(14,19,32,.92));
    border-bottom:1px solid var(--border);backdrop-filter:blur(6px)
  }
  .container{max-width:1080px;margin:0 auto;padding:24px}
  h1{font-size:28px;margin:0 0 8px 0}
  h2{font-size:22px;margin:32px 0 8px 0}
  h3{font-size:18px;margin:24px 0 6px 0}
  p.lead{color:var(--muted);margin-top:0}
  .grid{display:grid;gap:16px}
  @media(min-width:900px){.grid.cols-2{grid-template-columns:1fr 1fr}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:18px}
  .endp{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
  .method{padding:2px 10px;border-radius:999px;font-weight:600}
  .GET{background:#1a2b48;color:#9ad2ff;border:1px solid #2c4b7a}
  .POST{background:#1f2a1a;color:#a7f3a1;border:1px solid #335a30}
  .WS{background:#2a1f1f;color:#ffb2b2;border:1px solid #6b2f2f}
  .path{font-family:ui-monospace,Menlo,Consolas,monospace;background:var(--chip);border:1px solid var(--border);padding:2px 8px;border-radius:8px}
  .tag{font-size:12px;padding:2px 8px;border-radius:999px;background:var(--chip);border:1px solid var(--border);color:var(--muted)}
  .status{font-family:ui-monospace,Menlo,Consolas;display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid var(--border)}
  .ok{color:var(--ok);background:#15261f}
  .warn{color:var(--warn);background:#2a2510}
  .err{color:var(--err);background:#2a1414}
  table{width:100%;border-collapse:collapse;border:1px solid var(--border);border-radius:12px;overflow:hidden}
  th,td{padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:top}
  th{background:#10172a;color:#cfe0ff;text-align:left}
  tr:last-child td{border-bottom:none}
  code,pre{font-family:ui-monospace,Menlo,Consolas,monospace}
  pre{background:var(--code);border:1px solid var(--border);padding:14px;border-radius:12px;overflow:auto}
  .chips{display:flex;flex-wrap:wrap;gap:8px}
  .chip{background:var(--chip);border:1px solid var(--border);padding:2px 10px;border-radius:999px;color:var(--muted)}
  .kbd{border:1px solid var(--border);padding:0 6px;border-radius:6px;background:var(--chip);font:12px/20px ui-monospace,Menlo}
  .callout{border-left:4px solid var(--accent);padding:10px 12px;background:#0f1527;border:1px solid var(--border);border-radius:10px}
  footer{color:var(--muted);border-top:1px solid var(--border);margin-top:32px}
  .toc a{display:block;padding:6px 10px;border-radius:8px}
  .toc a:hover{background:#131a2e}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>Audio Core API</h1>
    <p class="lead">REST + WebSocket API для шумоподавления, ASR (Whisper), верификации диктора и служебных операций.</p>
    <div class="chips">
      <span class="chip">Версия: <strong>1.0.0</strong></span>
      <span class="chip">Базовый URL: <code>http://127.0.0.1:8000</code></span>
      <span class="chip">Документация Swagger: <code>/docs</code></span>
      <span class="chip">ReDoc: <code>/redoc</code></span>
    </div>
  </div>
</header>

<div class="container">
  <div class="panel toc">
    <strong>Навигация</strong>
    <div class="grid cols-2">
      <div>
        <a href="#auth">Аутентификация</a>
        <a href="#health">Health</a>
        <a href="#files">Файлы (upload/download)</a>
        <a href="#audio">Аудио: Enhance / Transcribe</a>
      </div>
      <div>
        <a href="#speaker">Спикер: Verify / Train</a>
        <a href="#ws">WebSocket: /ws/asr</a>
        <a href="#formats">Форматы, ресэмплинг, моно</a>
        <a href="#errors">Ошибки и коды ответов</a>
      </div>
    </div>
  </div>

  <section id="auth">
    <h2>Аутентификация</h2>
    <p>В базовой поставке аутентификация отсутствует (локальная разработка). Для продакшена добавьте JWT/ключи и rate limit на уровне реверс-прокси.</p>
  </section>

  <section id="health">
    <h2>Health</h2>
    <div class="panel">
      <div class="endp">
        <span class="method GET">GET</span>
        <span class="path">/health</span>
        <span class="tag">Service</span>
      </div>
      <p>Проверка доступности сервиса.</p>
      <table>
        <tr><th>Запрос</th><td>Параметры не требуются.</td></tr>
        <tr><th>Ответ</th><td><span class="status ok">200 OK</span> → <code>{"status":"ok"}</code></td></tr>
      </table>
    </div>
  </section>

  <section id="files">
    <h2>Файлы</h2>
    <div class="panel">
      <div class="endp">
        <span class="method POST">POST</span>
        <span class="path">/files/upload</span>
        <span class="tag">Files</span>
      </div>
      <p>Загрузка файла в хранилище API.</p>
      <table>
        <tr><th>Формат</th><td><code>multipart/form-data</code>, поле <code>file</code></td></tr>
        <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"filename":"example.wav"}</code></td></tr>
      </table>
      <h3>Примеры</h3>
      <pre><code class="lang-bash">curl -F "file=@sample.wav" http://127.0.0.1:8000/files/upload</code></pre>
</div>

<div class="panel">
  <div class="endp">
      <span class="method GET">GET</span>
      <span class="path">/files/download/{filename}</span>
      <span class="tag">Files</span>
  </div>
  <p>Скачивание файла из хранилища.</p>
  <table class="kv">
    <tr><th>Параметры пути</th><td><code>filename</code> — имя файла (как вернул <code>/files/upload</code>)</td></tr>
    <tr><th>Ответ</th><td><span class="status ok">200</span> → бинарный поток WAV/любого загруженного; <span class="status err">404</span> если нет файла</td></tr>
  </table>
  <pre><code class="lang-bash">curl -o out.wav http://127.0.0.1:8000/files/download/sample_enhanced.wav</code></pre>
</div>
  </section>

  <section id="audio">
    <h2>Аудио</h2>

<div class="panel">
  <div class="endp">
    <span class="method POST">POST</span>
    <span class="path">/audio/enhance</span>
    <span class="tag">Audio</span>
  </div>
  <p>Шумоподавление/улучшение аудио. Возвращает имя результирующего файла (скачать через <code>/files/download</code>).</p>
  <table>
    <tr><th>Формат</th><td><code>multipart/form-data</code>, поле <code>file</code> (WAV/другое — конвертируется)</td></tr>
    <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"output_filename":"&lt;name&gt;_enhanced.wav"}</code></td></tr>
  </table>
  <pre><code class="lang-bash">curl -F "file=@noisy.wav" http://127.0.0.1:8000/audio/enhance</code></pre>
</div>

<div class="panel" style="margin-top:12px">
  <div class="endp">
    <span class="method POST">POST</span>
    <span class="path">/audio/transcribe?language=ru</span>
    <span class="tag">ASR</span>
  </div>
  <p>ASR (Whisper): возвращает распознанный текст. Вход автоматически приводится к <strong>моно + SAMPLE_RATE + float32</strong>.</p>
  <table>
    <tr><th>Формат</th><td><code>multipart/form-data</code>, поле <code>file</code></td></tr>
    <tr><th>Параметры</th><td><code>language</code> — код языка (по умолчанию из конфигурации)</td></tr>
    <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"text":"...","raw":{...}}</code></td></tr>
  </table>
  <div class="grid cols-2">
    <div>
      <h3>curl</h3>
      <pre><code class="lang-bash">curl -F "file=@speech.wav" "http://127.0.0.1:8000/audio/transcribe?language=ru"</code></pre>
    </div>
    <div>
      <h3>Python (httpx)</h3>
      <pre><code class="lang-py">import httpx, pathlib
f = {"file": ("speech.wav", pathlib.Path("speech.wav").read_bytes(), "audio/wav")}
r = httpx.post("http://127.0.0.1:8000/audio/transcribe?language=ru", files=f)
print(r.json()["text"])</code></pre>
        </div>
      </div>
    </div>
  </section>

  <section id="speaker">
    <h2>Спикер</h2>
    <div class="panel">
      <div class="endp">
            <span class="method POST">POST</span>
            <span class="path">/speaker/verify</span>
            <span class="tag">Speaker</span>
      </div>
          <p>Сравнение <code>probe</code> и <code>reference</code> (оба файла — опционально <code>reference</code>). Возвращает score/decision.</p>
          <table>
            <tr><th>Формат</th><td><code>multipart/form-data</code>: <code>probe</code> (обяз.), <code>reference</code> (опц.)</td></tr>
            <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"score":0.83,"decision":true}</code></td></tr>
          </table>
    </div>
    <div class="panel" style="margin-top:12px">
      <div class="endp">
            <span class="method POST">POST</span>
            <span class="path">/speaker/train/microphone</span>
            <span class="tag">Speaker</span>
      </div>
      <p>Запись эталона с микрофона (на сервере) и сохранение. Используй только в доверенной среде.</p>
          <table>
            <tr><th>Запрос</th><td>Параметры не требуются</td></tr>
            <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"status":"ok","message":"..."}</code></td></tr>
          </table>
    </div>
  </section>

  <section id="ws">
<h2>WebSocket: потоковый ASR</h2>
<div class="panel">
  <div class="endp">
    <span class="method WS">WS</span>
    <span class="path">/ws/asr?language=ru&amp;sample_rate=16000</span>
    <span class="tag">ASR</span>
  </div>
  <p>Принимает бинарные фреймы аудио <strong>PCM16 mono LE</strong>, периодически отдаёт промежуточные и финальный текст.</p>

  <table>
    <tr><th>Параметры</th><td><code>language</code> — язык; <code>sample_rate</code> — частота (обычно из <code>core/config.py</code>)</td></tr>
    <tr><th>Вход (бинарь)</th><td>Сырые байты PCM16 mono LE (напрямую из микрофона). Можно вставлять текстовые управляющие события.</td></tr>
    <tr><th>Управление (JSON)</th><td><code>{"event":"start"}</code> (опц.), <code>{"event":"flush"}</code> (принудительный partial), <code>{"event":"stop"}</code> (финал + закрытие)</td></tr>
    <tr><th>Ответы</th><td>
      <div class="chips">
        <span class="chip"><code>{"type":"ready","sample_rate":16000,"language":"ru"}</code></span>
        <span class="chip"><code>{"type":"partial","text":"..."}</code></span>
        <span class="chip"><code>{"type":"final","text":"..."}</code></span>
        <span class="chip"><code>{"type":"error","detail":"..."}</code></span>
      </div>
    </td></tr>
  </table>

  <div class="grid cols-2" style="margin-top:10px">
    <div>
      <h3>Python (websockets + sounddevice)</h3>
<pre>
<code class="lang-py">import asyncio, json, numpy as np, sounddevice as sd, websockets
SR=16000
async def run():
    async with websockets.connect(f"ws://127.0.0.1:8000/ws/asr?language=ru&sample_rate={SR}") as ws:
        print(await ws.recv())  # ready
        def cb(indata, frames, t, status):
            pcm16=(indata[:,0]*32767).astype("&lt;i2").tobytes()
            asyncio.run_coroutine_threadsafe(ws.send(pcm16), asyncio.get_event_loop())
        with sd.InputStream(samplerate=SR,channels=1,dtype="float32",callback=cb):
            await ws.send(json.dumps({"event":"flush"}))
            print(await ws.recv())
            await asyncio.sleep(3)
            await ws.send(json.dumps({"event":"stop"}))
            print(await ws.recv())
asyncio.run(run())</code></pre>
        </div> 
<div>
          <h3>Браузер (идея)</h3>
          <p>Используйте Web Audio API + AudioWorklet (или ScriptProcessor) для получения PCM16 mono и отправки в WS. Если у вас <em>MediaRecorder</em> (opus), потребуется декодирование/перекодирование.</p>
        </div>
      </div>

  <div class="callout" style="margin-top:12px">
    <strong>Важные параметры стрима</strong><br/>
    <code>ASR_WINDOW_SEC</code> — длина «скользящего окна» аудио (сек).<br/>
    <code>ASR_EMIT_SEC</code> — периодичность выдачи partial (сек).<br/>
    Оба параметра задаются в <code>core/config.py</code>.
  </div>
</div>
  </section>

  <section id="formats">
    <h2>Форматы, ресэмплинг, моно</h2>
    <div class="panel">
      <ul>
        <li>Файлы читаются и приводятся к формату <strong>моно + SAMPLE_RATE + float32</strong> утилитой <code>load_and_resample()</code> (torchaudio) перед подачей в Whisper.</li>
        <li>Для WebSocket поток — <strong>PCM16 mono LE</strong> по сети; сервер самостоятельно приводит к float32.</li>
        <li>Рекомендовано использовать <strong>16 kHz</strong>, чтобы избежать лишних преобразований.</li>
      </ul>
    </div>
  </section>

  <section id="errors">
    <h2>Ошибки и коды ответов</h2>
    <div class="panel">
      <table>
        <tr><th>Код</th><th>Когда</th><th>Пример</th></tr>
        <tr><td><span class="status ok">200</span></td><td>Успех</td><td>Любой валидный запрос</td></tr>
        <tr><td><span class="status warn">400</span></td><td>Плохой запрос / неподдерживаемый формат</td><td><code>{"detail":"invalid sample rate"}</code></td></tr>
        <tr><td><span class="status err">404</span></td><td>Файл не найден</td><td><code>{"detail":"File not found"}</code></td></tr>
        <tr><td><span class="status err">422</span></td><td>Не прошла валидация входных данных</td><td><code>{"detail":[...]}</code></td></tr>
        <tr><td><span class="status err">500</span></td><td>Внутренняя ошибка сервера</td><td><code>{"detail":"..."}</code></td></tr>
      </table>
      <p class="muted">Подсказка: используйте <span class="kbd">/docs</span> для ручной прогона запросов и проверки схем.</p>
    </div>
  </section>

  <footer class="container">
    <p>© Audio Core API. Этот документ относится к архитектуре: <code>core/</code> — доменная логика, <code>app/services/</code> — адаптеры, <code>app/api/</code> — транспорт (REST/WS).</p>
  </footer>
</div>
</body>
</html>


# Установка: 
### gpu version: 
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install requirements_gpu.txt
```