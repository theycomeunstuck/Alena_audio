"""Produce an offline teacher-facing comparison from saved experimental results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(run_dir, histories, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = read_lines(run_dir / "results.jsonl")
    ingestion = read_lines(run_dir / "ingestion.jsonl")
    history_data = read_lines(histories)
    bundle = json.dumps({"manifest": manifest, "rows": rows, "ingestion": ingestion,
                         "histories": history_data}, ensure_ascii=False, default=str).replace("</", "<\\/")
    # This is a display copy; original run files keep the exact model output.
    html = TEMPLATE.replace("__DATA__", bundle).replace("\u2014", "\u2013")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return {"output": str(output), "rows": len(rows), "histories": len(history_data)}


TEMPLATE = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Память ребёнка – сравнение методов</title>
<style>
:root{color-scheme:light;--ink:#183344;--muted:#536975;--paper:#f2f7f8;--white:#fff;--teal:#14676b;--gold:#d1a037;--line:#c7d7da;--bad:#a23d3d}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 "Trebuchet MS",Verdana,sans-serif}
header{background:#dcecee;border-bottom:5px solid var(--teal);padding:28px max(22px,calc((100vw - 1500px)/2)) 24px}
h1{font-family:Georgia,serif;font-weight:normal;font-size:clamp(2rem,4vw,3.3rem);line-height:1.1;margin:0 0 10px}h2{font-family:Georgia,serif;font-weight:normal;font-size:1.65rem;margin:0 0 10px}h3{font-size:1.05rem;margin:0 0 8px}
p{max-width:78ch;margin:8px 0}main{max-width:1544px;margin:auto;padding:24px 22px 60px}.status{color:var(--muted);font-size:.92rem}.toolbar{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-bottom:22px}label{display:grid;gap:4px;font-weight:bold}select,button{font:inherit;color:var(--ink);background:white;border:1px solid #8da9ae;border-radius:5px;padding:8px;max-width:100%}select{min-width:190px}button{cursor:pointer}button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid var(--gold);outline-offset:2px}
.question{padding:17px 20px;background:#fff;border-left:5px solid var(--gold);margin-bottom:22px}.results{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.method{background:var(--white);border:1px solid var(--line);border-top:4px solid var(--teal);padding:16px;min-width:0}.method.error{border-top-color:var(--bad)}.method .small{color:var(--muted);font-size:.86rem}.answer{white-space:pre-wrap}.citation{font-size:.82rem;color:#745814;background:#fff4d9;padding:2px 5px;border-radius:3px;display:inline-block;margin:2px}.source-link{font-size:.82rem;padding:2px 6px;margin:3px 4px 3px 0;color:var(--teal)}.row-error{color:var(--bad)}details{margin-top:12px}summary{font-weight:bold;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef4f5;padding:10px;border-radius:4px;font:13px/1.45 Consolas,monospace;max-height:320px;overflow:auto}.wide{background:white;border:1px solid var(--line);padding:18px;margin-top:22px}.turn{padding:7px 9px;border-left:3px solid #cbd9dc;margin:4px 0}.turn.found{border-color:var(--gold);background:#fff7e6}.turn code{font-size:.78rem;color:#667c83}.fact{padding:7px 0;border-bottom:1px solid #e0eaec}.muted{color:var(--muted)}.graph-edge{margin:8px 0;padding:8px 10px;background:#edf5f5;border-left:3px solid var(--teal)}.graph-edge strong{color:var(--teal)}
@media(max-width:1050px){.results{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.results{grid-template-columns:1fr}.toolbar{display:grid}.toolbar select{width:100%}}
</style></head><body><header><h1>Память ребёнка</h1><p>Сравнение ответов, сохранённых наблюдений и исходных реплик.</p><p class="status" id="status"></p></header>
<main><div class="toolbar"><label>История<select id="history"></select></label><label>Вопрос<select id="question"></select></label><label>Объём памяти<select id="budget"></select></label></div>
<div class="question"><strong>Вопрос</strong><p id="questionText"></p></div><section><h2>Ответы методов</h2><div class="results" id="results"></div></section>
<section class="wide"><h2>Что сохранили индексы</h2><p class="muted">Записи библиотек – кандидаты воспоминаний. Ссылка указывает занятие, но не устанавливает точную реплику: её нужно найти и проверить в исходном тексте. В этом просмотре тире нормализованы, точные данные сохранены в файлах запуска.</p><div id="stored"></div></section>
<section class="wide"><h2>Граф ребёнка</h2><div id="graph"></div></section>
<section class="wide"><details id="historyDetails"><summary>Открыть исходную историю</summary><div id="timeline"></div></details></section></main>
<script id="payload" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);
const labels={profile_only:'Без истории',tail:'Последние реплики',summary:'Резюме',structured:'Структурированная V1',structured_v2:'Структурированная V2',graph:'Граф учебных тем',mem0:'Mem0',graphiti:'Graphiti + LangGraph',graphiti_mem0:'Graphiti + Mem0 + LangGraph'};
const $=id=>document.getElementById(id);function element(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x}
function options(select,values,render){select.replaceChildren(...values.map(v=>{const x=element('option',render(v));x.value=v;return x}))}
const histories=data.histories.filter(h=>data.manifest.history_ids.includes(h.history_id));options($('history'),histories.map(h=>h.history_id),x=>x);
function refreshQuestions(){const h=$('history').value;const r=data.rows.filter(x=>x.history_id===h);options($('question'),[...new Set(r.map(x=>x.request_id))],x=>x);options($('budget'),[...new Set(r.map(x=>x.budget))],x=>x==='fixed'?'До 4096 токенов':'Расширенный контекст');render()}
function render(){const h=$('history').value,q=$('question').value,b=$('budget').value;const rows=data.rows.filter(x=>x.history_id===h&&x.request_id===q&&x.budget===b);$('questionText').textContent=rows[0]?.question||'Для этого сочетания результатов нет.';$('results').replaceChildren();const used=new Set();const history=histories.find(x=>x.history_id===h);const lessonViews=new Map(),turnViews=new Map(),turnLessons=new Map();if(history)for(const lesson of history.lessons)for(const turn of lesson.turns)turnLessons.set(turn.turn_id,lesson.lesson_id);
function jump(lessonId,turnId){const lesson=lessonViews.get(lessonId);if(!lesson){$('status').textContent='Указанное занятие отсутствует в этой истории.';return}$('historyDetails').open=true;lesson.open=true;const target=turnViews.get(turnId)||lesson;target.scrollIntoView({behavior:'smooth',block:'center'});target.focus({preventScroll:true})}
function sourceButton(label,lessonId,turnId,cls){const button=element('button',label,cls||'source-link');button.type='button';button.addEventListener('click',()=>jump(lessonId,turnId));return button}
for(const row of rows){const card=element('article',undefined,'method'+(row.status==='ok'?'':' error'));card.append(element('h3',labels[row.method]||row.method));card.append(element('div',row.status==='ok'?`Память: ${row.memory_tokens??'–'} токенов; ${row.elapsed_seconds} с`:`Технический отказ: ${row.error_type||'ошибка'}`,'small'));
if(row.answer){card.append(element('p',row.answer.answer,'answer'));for(const id of row.answer.source_ids||[]){used.add(id);card.append(turnLessons.has(id)?sourceButton(id,turnLessons.get(id),id,'citation'):element('span',id+' – реплика не найдена','citation'))}if(row.answer.uncertain)card.append(element('p','Ответ содержит неопределённость.','small'))}else card.append(element('p',row.error||'Ответ не получен.','row-error'));
const found=element('details');found.append(element('summary','Найденные сведения и выбранная память'));const pre=element('pre',JSON.stringify({hits:row.hits||[],unresolved:row.unresolved||[],selected_memory:row.memory||'',skipped:row.skipped||[],events:row.events||[]},null,2));found.append(pre);card.append(found);$('results').append(card)}
const events=data.ingestion.filter(x=>x.history_id===h&&x.status==='ok');const saved=events.filter(x=>x.stage==='mem0').flatMap(x=>(x.outcome?.results||[]).map(y=>({lesson:x.lesson_id,text:y.memory})));$('stored').replaceChildren();if(!saved.length)$('stored').append(element('p','Сохранённых записей Mem0 в этом прогоне нет.','muted'));for(const x of saved){const p=element('div',undefined,'fact');p.append(element('div',x.text||''),sourceButton('Открыть занятие '+x.lesson,x.lesson));$('stored').append(p)}
const graph=events.filter(x=>x.stage==='graphiti');const nodes=new Map(),episodes=new Map(),edges=new Map();for(const g of graph){for(const id of (g.outcome?.episode_uuids||[g.outcome?.episode_uuid]).filter(Boolean))episodes.set(id,g.lesson_id);for(const n of g.outcome?.graph_nodes||[])nodes.set(n.id,n.name);for(const e of g.outcome?.graph_edges||[])edges.set(e.id,e)}$('graph').replaceChildren();if(!edges.size)$('graph').append(element('p','Данные рёбер не сохранены в этом прогоне.','muted'));for(const e of edges.values()){const p=element('div',undefined,'graph-edge');p.append(element('strong',(nodes.get(e.source)||'Сущность')+' – '+(nodes.get(e.target)||'сущность')));p.append(element('div',e.fact));const refs=[...new Set((e.episodes||[]).map(x=>episodes.get(x)||null))];if(!refs.length)p.append(element('div','Источник не указан','small'));for(const ref of refs)p.append(ref?sourceButton('Открыть занятие '+ref,ref):element('div','Эпизод не разрешён','small'));$('graph').append(p)}
const timeline=$('timeline');timeline.replaceChildren();if(history){for(const lesson of history.lessons){const d=element('details');d.tabIndex=-1;lessonViews.set(lesson.lesson_id,d);d.append(element('summary',lesson.observed_on+' – '+lesson.lesson_id));for(const t of lesson.turns){const div=element('div',undefined,'turn'+(used.has(t.turn_id)?' found':''));div.tabIndex=-1;turnViews.set(t.turn_id,div);const role=t.role==='teacher'?'Учитель: ':t.role==='assistant'?'Ассистент, только контекст: ':'Ребёнок: ';div.append(element('code',t.turn_id+' '),document.createTextNode(role+t.text));d.append(div)}timeline.append(d)}}}
$('history').addEventListener('change',refreshQuestions);$('question').addEventListener('change',render);$('budget').addEventListener('change',render);$('status').textContent=`Запуск ${data.manifest.run_id}; статус: ${data.manifest.status}. Техническое прохождение не является смысловой оценкой.`;refreshQuestions();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir, args.histories, args.output), ensure_ascii=True))


if __name__ == "__main__":
    main()
