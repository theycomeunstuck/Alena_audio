<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Audio Core API — Документация</title>
<style>
  :root{
    --bg:#0e1320; --panel:#141a2a; --ink:#e8eefc; --muted:#9fb0d3;
    --accent:#6ea8fe; --ok:#3ddc97; --warn:#ffce57; --err:#ff6b6b;
    --code:#0b1020; --chip:#202945; --border:#253053;
  }
  *{box-sizing:border-box} html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,system-ui,"Segoe UI",Roboto,"Helvetica Neue",Arial}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
  header{position:sticky;top:0;z-index:10;background:linear-gradient(180deg,var(--bg),rgba(14,19,32,.92));border-bottom:1px solid var(--border);backdrop-filter:blur(6px)}
  .container{max-width:1100px;margin:0 auto;padding:18px}
  h1{font-size:26px;margin:6px 0 0} h2{font-size:20px;margin:28px 0 10px} h3{font-size:16px;margin:18px 0 8px;color:var(--muted)}
  .grid{display:grid;grid-template-columns:1.2fr 1fr;gap:16px} .panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px}
  .endp{display:flex;align-items:center;gap:9px;margin-bottom:8px}
  .method{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,Consolas;background:var(--chip);padding:4px 8px;border-radius:999px}
  .GET{color:#b7e1ff}.POST{color:#cbf7cb}.WS{color:#ffe3a3}.tag{margin-left:auto;color:var(--muted)}
  code,pre{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas;background:var(--code);color:#d6e2ff;border:1px solid var(--border);border-radius:10px}
  pre{padding:12px;overflow:auto} code.inline{padding:2px 6px;border-radius:6px}
  table{width:100%;border-collapse:separate;border-spacing:0 6px}
  th,td{padding:6px 8px;border-bottom:1px dashed var(--border);vertical-align:top} th{color:var(--muted);font-weight:600;width:180px}
  .status{font:600 12px/1 ui-monospace;padding:4px 8px;border-radius:999px;border:1px solid var(--border)}
  .ok{color:var(--ok)} .warn{color:var(--warn)} .err{color:var(--err)}
  .chip{display:inline-block;margin:2px 6px 0 0;padding:4px 8px;border-radius:999px;background:var(--chip);color:#cfe0ff;font:600 11px/1 ui-monospace}
  .note{color:var(--muted);margin:6px 0 14px 0}.grid{display:grid;grid-template-columns:1fr;gap:16px}


</style>
</head>
<body>
<header><div class="container">
  <h1>Audio Core API</h1>
  <div class="note">REST + WebSocket API: шумоподавление, ASR (Whisper), верификация спикера (SpeechBrain ECAPA). Веб-слой не меняет доменную логику, а только адаптирует её под HTTP/WS.</div>
</div></header>

<div class="container grid" style="margin-top:16px">
  <div class="panel">
    <h2>Быстрый старт</h2>
    <pre>uvicorn app.main:app --reload</pre>
    <div class="note">Swagger: <code class="inline">/docs</code>. Этот HTML можно отдать как статику (<code>docs/index.html</code>).</div>
    <h3>Быстрый запуск (dev)</h3>
    <pre>python scripts/dev_run.py</pre>
<div class="note">
      Dev-раннер следит только за <code class="inline">app/</code> и <code class="inline">core/</code>,
      игнорируя <code class="inline">.venv/</code> и <code class="inline">pretrained_models/</code> — сервер не «дрожит» при изменениях кэшей моделей. 
</div>
<h3>Health</h3>
<div class="endp"><span class="method GET GET">GET</span><span>/health</span><span class="tag">Health</span></div>
<pre>200 → {"status":"ok"}</pre>
  </div>
  <div class="panel">
    <h2>Файлы</h2>

<div class="endp"><span class="method POST POST">POST</span><span>/files/upload</span><span class="tag">Files</span></div>
<table>
  <tr><th>Формат</th><td>multipart/form-data: <code class="inline">file</code> (audio/wav|mp3|flac)</td></tr>
  <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"filename":"..."}</code></td></tr>
</table>

<div class="endp"><span class="method GET GET">GET</span><span>/files/download/{filename}</span><span class="tag">Files</span></div>
<table>
  <tr><th>Ответ</th><td><span class="status ok">200</span> → файл; <span class="status err">404</span> если нет</td></tr>
</table>
  </div>

  <div class="panel">
    <h2>Аудио</h2>

<div class="endp"><span class="method POST POST">POST</span><span>/audio/enhance</span><span class="tag">Audio</span></div>

<table>
  <tr><th>Формат</th><td>multipart/form-data: <code class="inline">file</code> (WAV/MP3/FLAC → внутри конвертируется в mono 16k)</td></tr>
  <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"output_filename":"..._enhanced.wav"}</code></td></tr>
</table>
<div class="note">Файл сохраняется по пути app/storage/{FileName}_enhanced.<strong>wav</strong> (и конвертирует в .wav)</div>






<div class="endp"><span class="method POST POST">POST</span><span>/audio/transcribe</span><span class="tag">Audio</span></div>
<table>
  <tr><th>Параметры</th><td><code class="inline">language</code>=ru|en|...</td></tr>
  <tr><th>Формат</th><td>multipart/form-data: <code class="inline">file</code></td></tr>
  <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"text":"...","raw":{...}}</code></td></tr>
</table>
  </div>

  <div class="panel">
    <h2>Speaker</h2>

<div class="endp"><span class="method POST POST">POST</span><span>/speaker/verify</span><span class="tag">Speaker</span></div>
<table>
  <tr><th>Формат</th><td>
    multipart/form-data: <code class="inline">probe</code> (audio/wav) + <code class="inline">reference</code> (audio/wav, опционально).
  </td></tr>
  <tr><th>Fallback</th><td>Если <code>reference</code> не передан, сервер (через core) может использовать локальный <code>reference.wav</code> (если настроен).</td></tr>
  <tr><th>Ответ</th><td><span class="status ok">200</span> → <code>{"score":0.83,"decision":true}</code>; <span class="status err">400</span> если эталон недоступен.</td></tr>
</table>

<h3>Примеры</h3>
<pre>curl -F "probe=@probe.wav;type=audio/wav" -F "reference=@ref.wav;type=audio/wav" http://127.0.0.1:8000/speaker/verify</pre>
<pre>curl -F "probe=@probe.wav;type=audio/wav" http://127.0.0.1:8000/speaker/verify  # если настроен локальный reference.wav</pre>
  </div>

  <div class="panel">
    <h2>WebSocket: потоковый ASR</h2>

<div class="endp"><span class="method WS WS">WS</span><span>/ws/asr</span><span class="tag">WebSocket</span></div>
<div class="note">
      Долгоживущий сокет для распознавания речи в реальном времени.
</div>
<table>
  <tr><th>Параметры (optional)</th><td><code class="inline">language</code>, <code class="inline">sample_rate</code>, 
<code class="inline">windows_sec</code>, <code class="inline">emit_sec</code>, <code class="inline">inactivity_sec</code></td></tr>
  <tr><th>Handshake</th><td>Сервер шлёт: <code>{"type":"ready","sample_rate":16000,"language":"ru"}</code></td></tr>
  <tr><th>Binary</th><td>Посылаем чанки <strong>PCM16 mono (little-endian)</strong>. Буфер обрезается до окна <em>ASR_WINDOW_SEC</em>.</td></tr>
  <tr><th>Text</th><td><code>{"event":"flush"}</code> → вернуть partial; <code>{"event":"stop"}</code> → final + закрыть соединение.</td></tr>
  <tr><th>Ответ</th><td><code>{"type":"partial","text":"..."}</code> / <code>{"type":"final","text":"..."}</code> / <code>{"type":"error","detail":"..."}</code></td></tr>
</table>

<h3>Query параметры</h3>
<table>
  <tr><th>Параметр</th><th>Описание</th><th>Пример</th></tr>
  <tr><td>language</td><td>Язык модели</td><td>ru</td></tr>
  <tr><td>sample_rate</td><td>Частота входящего PCM16</td><td>16000</td></tr>
  <tr><td>channels</td><td>Каналы (1=моно, 2=стерео)</td><td>1</td></tr>
  <tr><td>window_sec</td><td>Длительность окна для анализа, сек</td><td>8.0</td></tr>
  <tr><td>emit_sec</td><td>Частота опроса клиента к серверу, сек</td><td>2.0</td></tr>
  <tr><td>inactivity_sec</td><td>Завершение при тишине более, сек</td><td>2.5</td></tr>
</table>

<h3>Ответы сервера</h3>
    <pre>{
      "type": "ready", "sample_rate": 16000, "language": "ru"
    }
    {
      "type": "partial", "text": "промежуточный текст", "utt_id": "123"
    }
    {
      "type": "final", "text": "окончательный текст", "utt_id": "124"
    }
    {
      "type": "ok", "detail": "reset", "utt_id": "125"
    }
    {
      "type": "error", "detail": "описание ошибки"
}</pre>


  </div>




  <div class="panel">
    <h2>Примечания</h2>
    <ul>
      <li>Все аудио конвертируются в <strong>mono 16 kHz float32</strong> перед обработкой.</li>
      <li><code>score</code> — косинусная близость эмбеддингов; <code>decision</code> — пороговая интерпретация (по умолчанию 0.5, можно вынести в конфиг).</li>
      <li>WebSocket может отдавать пустой текст на коротких/беззвучных фрагментах — это нормально.</li>
    </ul>

<h2>WS не виден в Swagger</h2>
    <div class="note">
      OpenAPI/Swagger описывает только HTTP-маршруты. WebSocket-эндпоинты FastAPI в <code class="inline">/docs</code> не отображаются — это ожидаемое поведение.
      Для WS используйте раздел «WebSocket: потоковый ASR» на этой странице.
    </div>

<h2>Примечания по недоделанной работе</h2>
    <div class="note">
    <li> Модуль train_voice (регистрация пользователя) пока что не реализована в api.</li>
    <li> Намёков на модуль TTS пока что нет (даже не начинал браться)</li>
    <li> Нет перебора по поиску голосов, то есть верификация сейчас работает только на одного пользователя </li>
    <li> В ws train microphone не обрабатывается гонка пакетов </li>
    <li> verify/speaker реализован не в core, а <strong>целиком</strong> в <code>app/services/speaker_service.py</code> (нагромождено)</li>
    
</div>
  </div>



</body>
</html>