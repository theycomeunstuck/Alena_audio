import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// Run with the bundled Node runtime. All slide evidence remains editable.
const workspaceDir = path.resolve(process.argv[2] ?? process.cwd());
const buildDir = path.join(workspaceDir, '.local', 'report-presentation-build');
const outputDir = path.join(workspaceDir, 'docs', 'research-report-2026-10-20');
const finalizedDir = path.join(workspaceDir, '.local', 'report-presentation-finalized');
const SKILL_DIR = 'C:/Users/name/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const RUNTIME_PYTHON = 'C:/Users/name/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES ??= 'C:/Users/name/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const toolPath = 'C:/Users/name/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const { Presentation, PresentationFile } = await import(pathToFileURL(toolPath).href);
const { finalizePresentation, applyPresentationChartFont, resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(finalizedDir, { recursive: true });
const reportPath = path.join(workspaceDir, 'docs/research-report-2026-10-20/evidence/report.json');
const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
const resourceRows = report.generation_resources.heldout.active_attempts;
const resource = (method, stage) => resourceRows.find(row => row.method === method && row.stage === stage);
const tok = (method, stage, key) => resource(method, stage)[key].known_sum;
const fmt = value => value.toLocaleString('ru-RU');
const FONT = resolvePresentationFont({ fontFamily: 'Arial' });
const C = { ink: '#142B3D', muted: '#536777', teal: '#087F8C', orange: '#AD4A1B', line: '#CAD6DE', pale: '#ECF4F5', bg: '#FFFFFF' };
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const sources = {
  frozen: 'nirs/24_final_memory_results.md. Frozen report: docs/research-report-2026-10-20/evidence/report.json. Protocol memory-final-v4.',
  integration: 'nirs/25_memory_integration_results.md. Load raw: .local/memory-verification/final_20261018_v4/load.json.',
  cases: 'nirs/26_expert_review_and_conversation_memory.md. Supplemental AI analysis only, not a human or calibrated assessment.',
  continuation: 'docs/MEMORY_CONTINUATION_RESULTS_2026_10_20.md. Synthetic API scenario and preserved raw artifacts in docs/memory-continuation-v*/.',
};
function text(slide, name, value, x, y, w, h, size=28, color=C.ink, bold=false, align='left') {
  const obj = slide.shapes.add({ geometry: 'textbox', name, position: { left:x, top:y, width:w, height:h }, fill:'none', line:{fill:'none',width:0} });
  obj.text = value;
  obj.text.style = { typeface:FONT, fontSize:size, bold, color, alignment:align, verticalAlignment:'top', autoFit:'none', wrap:'square', insets:0 };
  return obj;
}
function slide(title, note, number) {
  const s = presentation.slides.add();
  s.background.fill=C.bg;
  text(s, 'title', title, 68, 48, 1144, 108, 44, C.ink, true);
  text(s, 'page-number', String(number).padStart(2,'0'), 1158, 668, 54, 30, 18, C.muted, false, 'right');
  s.speakerNotes.text=note;
  return s;
}
function table(s, name, values, x, y, widths, rowHeight=74, size=25) {
  const t=s.tables.add({ rows:values.length, columns:values[0].length, left:x, top:y, width:widths.reduce((a,b)=>a+b,0), height:rowHeight*values.length, columnWidths:widths, values });
  t.name=name;
  t.borders.assign({fill:C.line,width:1,style:'solid'});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({ fill:'#FFFFFF', textStyle:{ typeface:FONT,fontSize:size,color:C.ink }, margins:{left:14,right:14,top:13,bottom:10},anchor:'middle' });
  t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({ fill:C.ink, textStyle:{typeface:FONT,fontSize:size,bold:true,color:'#FFFFFF'} });
  for(let i=0;i<values.length;i++)t.rows[i].height=rowHeight;
  return t;
}
function node(s,name,label,x,y,w=230,h=102) {
  const n=s.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},fill:C.pale,line:{fill:C.teal,width:1.5}});
  n.text=label;
  n.text.style={typeface:FONT,fontSize:25,bold:false,color:C.ink,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:12};
  return n;
}
function connect(s,a,b,from='right',to='left',color=C.teal) {
  return s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:color,width:2},tail:{type:'arrow',width:'med',length:'med'}});
}
function para(s, name, value, y, size=29, color=C.ink, bold=false, h=86) {return text(s,name,value,68,y,1140,h,size,color,bold);}

// 01. Exact approved research topic, without invented institutional details.
{
  const s=presentation.slides.add(); s.background.fill=C.ink;
  text(s,'product','TeachCopilot',68,52,1120,84,64,'#FFFFFF',true);
  text(s,'exact-approved-topic','Исследование методов отбора и представления информации об ученике из долговременной памяти ИИ-репетитора при планировании персонализированных учебных занятий',68,166,1138,306,42,'#FFFFFF',true);
  text(s,'author','Климов М.Ю., группа 6В41ПИШ\nРуководитель – Зебзеев А.Г.',68,540,1000,90,27,'#DCE9EF');
  text(s,'date','20 октября 2026',68,655,1050,30,20,'#DCE9EF');
  s.speakerNotes.text='Тема и реквизиты взяты из утвержденного индивидуального задания «идз 5 сем.docx». Статус на 20 октября 2026 года. Презентация отделяет технические проверки от содержательной оценки и исследования учебного эффекта.';
}
// 02.
{
  const s=slide('Память между учебными сессиями',sources.continuation,2);
  para(s,'child-message','«хз» становится понятнее, когда сохранён предыдущий вопрос.',184,36,C.teal,true,95);
  para(s,'persistence','После завершения чата система хранит исходные реплики в PostgreSQL. Перезапуск приложения сохраняет эти источники.',314,29,C.ink,false,100);
  para(s,'selection','В новой сессии отбор возвращает подтверждённую память по выбранной теме вместе со ссылками на источники.',439,29,C.ink,false,100);
  para(s,'limit','Короткая реплика и слова ассистента сами по себе не подтверждают освоение темы.',574,27,C.orange,false,68);
}
// 03. Native editable explanatory diagram, as explicitly requested.
{
  const s=slide('Память ученика и поиск учебных материалов',sources.continuation+' '+sources.integration,3);
  const a=node(s,'source-turns','Реплики\nи заметки',68,192,230,106);
  const b=node(s,'postgres','PostgreSQL\nисходные источники',372,192,230,106);
  const c=node(s,'review','Версия памяти\nпросмотр учителем',676,192,230,106);
  const d=node(s,'selection','Отбор по ученику\nи теме',980,192,230,106);
  connect(s,a,b);connect(s,b,c);connect(s,c,d);
  const e=node(s,'materials','Учебные\nматериалы',68,454,230,106);
  const f=node(s,'retrieval','Embeddings\nи векторный поиск',372,454,230,106);
  const g=node(s,'current-dialogue','Контекст ответа\nи текущий диалог',676,454,230,106);
  const h=node(s,'chat','Ответ ученику',980,454,230,106);
  connect(s,e,f);connect(s,f,g);connect(s,g,h);connect(s,d,h,'bottom','top');
  text(s,'memory-lane','Долговременная память',68,146,600,36,24,C.teal,true);
  text(s,'rag-lane','RAG учебных материалов',68,402,600,36,24,C.teal,true);
  text(s,'diagram-caveat','RAG добавляет материалы. Память сохраняет историю ученика и основания выводов.',68,612,1128,48,24,C.muted);
  s.speakerNotes.text += '\nСхема показывает потоки данных для чата ученика. Текущий диалог вместе с найденными материалами формирует контекст ответа. Чат использует подтверждённую память. Рекомендация преподавателю готовится отдельно из доступных источников и не требует предварительного подтверждения памяти. Исходные реплики сохраняются отдельно от модельных выводов. Модельные предположения о компетенциях требуют преподавательского просмотра.';
}
// 04.
{
  const s=slide('Четыре способа отбора и представления памяти',sources.frozen,4);
  table(s,'four-methods',[
    ['Метод','Что получает генератор'],
    ['profile_only','Только карточку ученика'],
    ['tail','Последние исходные реплики'],
    ['summary','Последовательно обновляемое резюме'],
    ['structured','Записи с типами, источниками и исправлениями'],
  ],68,182,[270,870],76,27);
  para(s,'method-confounding','Методы одновременно меняют отбор и представление. Эффект одного формата JSON отдельно не установлен.',599,25,C.muted,false,61);
}
// 05.
{
  const s=slide('Протокол сравнения',sources.frozen,5);
  para(s,'experiment-shape','16 историй × 2 запроса × 4 метода = 128 запланированных ответов',177,35,C.teal,true,104);
  para(s,'situations','8 типов ситуаций, каждый в короткой и длинной истории. Длинные истории превышают бюджет выбранной памяти.',300,28,C.ink,false,98);
  table(s,'protocol-parameters',[
    ['Модель и настройки','Ограничения'],
    ['gemma4:12b, фиксированный digest','Память – до 4000 символов'],
    ['seed 42, temperature 0','Контекст – 8192 токена'],
    ['Код и промпты заморожены','Ответ – до 1536 токенов'],
  ],68,398,[570,570],51,23);
  text(s,'paired-history-unit','Корпус и рубрика не проверены человеком. Единица сопоставления – история.\n128 ответов не являются 128 независимыми наблюдениями.',68,632,1110,62,20,C.muted);
  s.speakerNotes.text += '\nТипы: помощь, самостоятельная попытка, неизвестные условия, исправление записи, реальный прогресс, незавершённая работа, вытеснение важного старого события, неоднозначное авторство. Существенные факты и допустимые действия зафиксированы до генерации, генератор их не получал. Корпус и рубрика содержательно не проверены человеком. Digest 6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b. Восемь dev-историй отделены от 16 отложенных.';
}
// 06. Native chart with literal evidence, materialized workbook in finalization.
{
  const s=slide('Технически приняты 94 из 128 ответов',sources.frozen,6);
  const chart=s.charts.add('bar',{
    name:'technical-completion',position:{left:70,top:173,width:1110,height:353},
    categories:['profile_only','tail','summary','structured'],
    series:[{name:'Технически приняты',values:[32,32,24,6],fill:C.teal}],
    hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:110},
    chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',plotAreaLine:{fill:'none',width:0},
    yAxis:{minimumScale:0,maximumScale:32,majorUnit:8,numberFormatCode:'0',majorGridlines:{fill:C.line,width:1},textStyle:{typeface:FONT,fontSize:22,color:C.muted}},
    xAxis:{textStyle:{typeface:FONT,fontSize:25,color:C.ink}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:27,bold:true,color:C.ink}},
  });
  applyPresentationChartFont(chart,{fontFamily:FONT});
  para(s,'denominator','У каждого метода знаменатель – 32. Все 34 отказа связаны с обновлением памяти.',560,28,C.orange,false,75);
  text(s,'technical-vs-semantic','Принятый контракт ответа не устанавливает фактическую правильность.',68,647,1110,34,23,C.muted);
  s.speakerNotes.text += '\nprofile_only 32/32, tail 32/32, summary 24/32 (8 memory_error), structured 6/32 (26 memory_error). Это покрытие полного конвейера в конкретном протоколе. Не оценка педагогической полезности и не универсальное ранжирование методов.';
}
// 07.
{
  const s=slide('Содержательная оценка остановилась на калибровке',sources.frozen,7);
  text(s,'calibration-number','23 / 24',68,187,500,111,83,C.teal,true);
  text(s,'calibration-caption','калибровочных случаев технически приняты',68,309,500,95,28,C.ink);
  text(s,'coverage-number','0 / 128',690,187,520,111,83,C.orange,true);
  text(s,'coverage-caption','ответов получили формальную содержательную оценку',690,309,520,111,28,C.ink);
  para(s,'failed-case','В C022 судья привёл цитату, которая не является буквальным фрагментом источника. Условие допуска не выполнено.',460,28,C.ink,false,106);
  para(s,'no-winner','Все 128 ответов сохраняют статус not_assessed. Победитель по качеству не установлен.',590,29,C.orange,true,73);
  s.speakerNotes.text += '\nqualified=False, calibration_not_qualified. Ожидаемые метки заданы до вызовов. Приёмка 23 калибровочных ответов не заменяет успешную калибровку всех обязательных случаев. Пустое unknowns само по себе не ошибка, review_required означает преподавательский просмотр, один успех не доказывает устойчивого освоения. Формальные показатели искажений, полезности и пяти критериев для heldout не вычислены.';
}
// 08.
{
  const s=slide('Разбор примеров с технически принятыми ответами',sources.cases,8);
  text(s,'r032-id','R032',68,182,160,46,32,C.teal,true);
  text(s,'r032-case','Учитель отменил ошибочную заметку. В исходной работе ребёнка уже было 303.',256,181,940,86,28,C.ink);
  text(s,'r032-find','Ответ модели ошибочно приписал ребёнку пропуск нуля. Новая самостоятельная проверка полезна, но это не подтверждает ложное прошлое.',256,286,940,121,27,C.orange);
  text(s,'r012-id','R012',68,449,160,46,32,C.teal,true);
  text(s,'r012-case','105 : 5 = 21 и 204 : 6 = 34 не содержат нуля внутри частного, хотя ответ заявляет именно эту цель.',256,448,940,100,28,C.ink);
  text(s,'r012-correct','Подходящий пример для этой цели – 1248 : 12 = 104.',256,566,940,50,28,C.teal,true);
  text(s,'ai-analysis-status','Дополнительный разбор ИИ. Он не заменяет человеческий просмотр и калиброванную оценку корпуса.',68,654,1110,34,20,C.muted);
  s.speakerNotes.text += '\nСлучаи взяты из заранее выбранного слепого пакета: 5 историй × 1 запрос × 4 метода = 20 позиций. Нельзя использовать этот выборочный разбор для частот ошибок по всему корпусу или статистического победителя. R032: t07-l02-t01 исправляет запись учителя, а не действие ребёнка.';
}
// 09.
{
  const s=slide('Токены обновления памяти и итогового ответа',sources.frozen,9);
  const methods=['profile_only','tail','summary','structured'];
  const vals=[['Метод','Обновление\nвход / выход','Ответы\nвход / выход']];
  for(const m of methods)vals.push([m,tok(m,'memory_update','input_tokens')?`${fmt(tok(m,'memory_update','input_tokens'))} / ${fmt(tok(m,'memory_update','output_tokens'))}`:'0 / 0',`${fmt(tok(m,'answer','input_tokens'))} / ${fmt(tok(m,'answer','output_tokens'))}`]);
  table(s,'resource-costs',vals,68,182,[270,435,435],74,26);
  para(s,'actual-calls','Устойчивый журнал: 236 фактических локальных вызовов, 275 825 входных и 75 750 выходных токенов.',582,26,C.ink,false,70);
  text(s,'resource-limit','Полный вход включает схемы. Денежная стоимость локальных вычислений не измерена.',68,661,1110,30,20,C.muted);
  s.speakerNotes.text += '\nДанные generation_resources.heldout.active_attempts и durable_call_ledger. Обновление принадлежит истории и используется её двумя запросами, повторно на каждый ответ не начисляется. Журнал этапов содержит 238 записей. Два вызова обёртки summary для t08-l08 и t10-l08 отклонены локально до записи в устойчивый журнал и до сетевого вызова после truncated_completion на l07. Для этих двух записей performed=true ошибочен. Фактических вызовов 236, cache_hits=0. Источник уточнения: docs/research-report-2026-10-20/metrics.json, call_count_explanation. Суммы секунд журналов этапов: profile_only ответ 37.50, tail ответ 68.18, summary обновление 331.30 и ответ 49.72, structured обновление 143.27 и ответ 9.44. Суммы времени и токенов не являются нормированным качеством: число полученных ответов различается. Нулевые обновления profile_only и tail не означают нулевую стоимость хранения. Энергия и амортизация не измерялись.';
}
// 10.
{
  const s=slide('Работающий прототип и проверка после перезапуска',sources.continuation,10);
  para(s,'restart-result','Две синтетические сессии. Подтверждённая версия памяти сохранила тот же хеш после перезапуска.',178,31,C.teal,true,109);
  para(s,'realistic-child','Ученик выбирает тему и отвечает коротко. Ассистент учитывает предыдущий вопрос и подтверждённую память до поиска материалов.',315,28,C.ink,false,107);
  table(s,'prototype-evidence',[
    ['API-сценарий','Программные проверки'],
    ['13 попыток, 9 ответов, 4 технических отказа','810 тестов приложения + 52 проверки политики'],
  ],68,439,[570,570],68,26);
  text(s,'prototype-limit','Первую ошибку объяснения исправила синтетическая учительская заметка ИИ. Воспоминание частичное. Эффект обучения не измерялся.',68,602,1128,70,23,C.orange);
  s.speakerNotes.text += '\nЭто инженерная проверка, не часть замороженного сравнительного исследования. Все 13 попыток и ошибки сохранены, скрытых повторов не было. Исправление от учителя добавил ИИ в синтетическом сценарии, человек его содержательно не оценивал. После рестарта ответ использовал контекст прошлого круга и интерес к кометам, но не воспроизвёл явно 10 и 4. Причинный вклад памяти и полноценное фактическое воспоминание не установлены. Карточка компетенций автоматически не менялась. В приложении tail с дополненным представлением пар dialogue-pairs-v2, он отличается от исследованного frozen tail.';
}
// 11.
{
  const s=slide('Нагрузка на очередь при 5, 10 и 20 сессиях',sources.integration,11);
  table(s,'simulated-load',[
    ['Сессии','Ошибки','Очередь p95\nlocal, мс','Очередь p95\nexternal, мс'],
    ['5','0','1072,2','201,0'],
    ['10','0','2093,9','527,5'],
    ['20','0','4646,2','1200,9'],
  ],68,183,[205,205,365,365],72,26);
  para(s,'simulation','Реальные PostgreSQL и очередь. Имитация провайдера – 40 мс. Пик вызовов: local 1, external до 4.',507,27,C.ink,false,95);
  text(s,'public-limit','Пропускная способность публичного API не измерена. Внешний проход: 4 попытки, 3 ответа, 1 неизвестный оплачиваемый исход.',68,622,1130,70,23,C.orange);
  s.speakerNotes.text += '\nНоль ошибок относится к обеим линиям на всех трёх уровнях нагрузки. Пики external для 5/10/20 равны 3/4/3, local везде 1. Embeddings и модель в этой нагрузке не вычислялись, использованы синтетические детерминированные векторы. RSS процесса нагрузки 78.5 МиБ. RAM хоста и VRAM нельзя причинно приписывать этой нагрузке, параллельно шёл исследовательский прогон. Внешний проход не признан полностью успешным: четвёртый вызов получил HTTP-ошибку с неизвестным списанием, пятый пример не запускался, резерв сохранён и повтор не выполнялся.';
}
// 12.
{
  const s=slide('Выводы и следующий этап оценки',sources.frozen+' '+sources.continuation,12);
  para(s,'finding','В данном протоколе profile_only и tail завершили все запросы. summary и structured часто останавливались при обновлении памяти.',176,30,C.ink,false,114);
  para(s,'scope','Это вывод о технической надёжности конвейера с одной моделью. Преимущество по содержательному качеству пока не установлено.',318,30,C.orange,true,117);
  para(s,'prototype','Прототип хранит память между сессиями. Рабочий метод по умолчанию – tail. Подтверждение преподавателя остаётся отдельным шагом.',459,28,C.ink,false,107);
  para(s,'next','Далее – человеческий просмотр слепого пакета и спорных случаев, затем содержательная оценка без изменения сохранённых ответов.',591,27,C.teal,true,91);
  s.speakerNotes.text += '\nОграничения: синтетический корпус и рубрика не проверены человеком, одна локальная модель и одна конфигурация, разные способы совместно меняют отбор и представление, общий вход не выровнен по токенам, 16 связанных историй, не 128 независимых наблюдений. Работающий прототип и программные тесты не подтверждают педагогическую эффективность. Требуются корректно допущенный судья либо явно организованная человеческая оценка, без незаметной подмены исходного протокола. Отрицательный результат допустим.';
}

for (const [i,s] of presentation.slides.items.entries()) {
  const stem=`slide-${String(i+1).padStart(2,'0')}`;
  await fs.writeFile(path.join(buildDir,`${stem}.png`),new Uint8Array(await (await presentation.export({slide:s,format:'png',scale:1})).arrayBuffer()));
  await fs.writeFile(path.join(buildDir,`${stem}.layout.json`),await (await s.export({format:'layout'})).text());
}
const montage=await presentation.export({format:'webp',montage:true});
await fs.writeFile(path.join(buildDir,'montage.webp'),new Uint8Array(await montage.arrayBuffer()));
const candidatePath=path.join(buildDir,'candidate-v1.pptx');
await(await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(finalizedDir,`presentation-${Date.now()}.pptx`);
const deliverablePath=path.join(outputDir,'Презентация_память_TeachCopilot.pptx');
const chartDataBefore=new Set((await fs.readdir(workspaceDir)).filter(n=>n.startsWith('.chart-data-')));
let result;
try {
result=await finalizePresentation({
  explicitTotalSlideCount:12,
  requiredNativeTableOwnerSlides:[4,5,9,10,11],
  requiredNativeChartOwnerSlides:[6],
  materializeLiteralChartWorkbooks:true,
  workspaceDir,candidatePath,finalPath,
  pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[4,5,9,10,11].flatMap(n=>['--require-native-table-slide',String(n)])],
  fontPolicy:{basis:'design',families:[FONT]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(buildDir,`${path.basename(finalPath)}.validation.json`),
});
} finally {
  for (const name of (await fs.readdir(workspaceDir)).filter(n=>n.startsWith('.chart-data-')&&!chartDataBefore.has(n))) {
    const source=path.resolve(workspaceDir,name);
    if(path.dirname(source)!==workspaceDir)throw new Error('Unexpected chart temporary path');
    await fs.rename(source,path.join(buildDir,name));
  }
}
await fs.copyFile(finalPath,deliverablePath);
await fs.writeFile(path.join(buildDir,'build-result.json'),JSON.stringify({slides:12,finalPath,deliverablePath,result},null,2));
console.log(JSON.stringify({slides:12,deliverablePath,finalSha256:result.finalSha256,layoutWarnings:result.presentationLayout?.warnings},null,2));
