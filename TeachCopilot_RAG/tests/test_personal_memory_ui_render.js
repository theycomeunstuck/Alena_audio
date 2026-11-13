// Run the production renderers and chat dispatch with a small simulated DOM.
// No browser, model or GPU is started by this test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('web/app.js', 'utf8');
function section(start, end) {
  const from = script.indexOf(start), to = script.indexOf(end, from);
  assert(from >= 0 && to > from, `Missing production function: ${start}`);
  return script.slice(from, to);
}
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.listeners = {}; this.textContent = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute(name, value) { this[name] = value; }
  scrollIntoView() { this.scrolled = true; }
  focus() { this.focused = true; }
}
const descendants = node => node && typeof node === 'object' ? [node, ...(node.children || []).flatMap(descendants)] : [];
const text = node => descendants(node).map(item => item.textContent).join('\n');
const digest = 'b'.repeat(64);
const original = { source_id: 'original', role: 'student', text: 'Original answer', topic_id: 'math', observed_on: '2026-11-01' };
const fix = { source_id: 'fix', role: 'teacher', text: '<script>literal full correction</script>', corrects_source_id: 'original', topic_id: 'math', observed_on: '2026-11-02' };
const other = { source_id: 'other', role: 'student', text: 'Other lesson', topic_id: 'reading' };
const session = {
  learning_session_id: 'lesson-one', completed_at: '2026-11-01T10:00:00Z', topic_id: 'math',
  facts: [{ source_id: 'fix', role: 'teacher', text: 'literal…', kind: 'correction', truncated: true,
    correction_chain: [{ source_id: 'original', role: 'student', status: 'superseded' }, { source_id: 'fix', role: 'teacher', status: 'current' }] }],
  ending: { role: 'assistant', usage: 'context_only', text: 'Let us check it', truncated: false },
};
const summary = {
  available: true, count: 1, limit: 1, sessions: [session], selected_source_count: 1, omitted_source_count: 2,
  date_span: { start: session.completed_at, end: session.completed_at }, algorithm: 'extractive_v1',
  text: '{"this_is_not_user_facing":true}', version_id: 'summary-version',
};
function view({ approved = false, fullArchive = false } = {}) {
  const nodes = Object.fromEntries(['memory-history', 'memory-status', 'memory-summaries'].map(id => [`#${id}`, new Element('div')]));
  nodes['#topic-select'] = { value: 'math' };
  const sources = [original, fix, other];
  const version = { version_id: 'memory-version', method: 'personal_memory', approved, topic_id: 'math',
    state: { source_ids: sources.map(source => source.source_id), archive_snapshot_fingerprint: digest },
    context: { text: '{"internal_graph":true}', sources: [original, fix],
      ...(fullArchive ? { source_access: { mode: 'full_archive', source_ids: sources.map(source => source.source_id), snapshot_fingerprint: digest } } : {}) } };
  const data = { revision: 2, source_revision: 3, sources: [original, fix], personal_graph_sources: sources,
    personal_graph_snapshot_fingerprint: digest, versions: [version],
    session_summaries: { last_session: summary, last_seven_sessions: { ...summary, limit: 7 } } };
  const posts = [], gets = [];
  const state = { learnerId: 'child', viewRevision: 1, memoryViewRevision: 0, token: 'token',
    account: { role: 'teacher' }, features: { personal_memory: true, personal_graph: false }, memoryCorrectionJobs: new Map() };
  const context = {
    document: { createElement: tag => new Element(tag), getElementById: id =>
      Object.values(nodes).flatMap(descendants).find(node => node.id === id) },
    state, URLSearchParams, $: selector => nodes[selector],
    topicTitle: row => row.topic_id || '', safeDate: value => value || '', memorySourceDomId: id => `memory-source-${id}`,
    request: async (path, options = {}) => {
      if (path.endsWith('/confirm')) { posts.push(options.body); return {}; }
      if (path.endsWith('/dashboard')) return { revision: 2 };
      gets.push(path); return data;
    },
    appendMemoryRecords: () => {}, appendSelectionWarnings: () => {}, renderDashboard: () => {}, renderNextStep: () => {},
    errorText: error => error.message, busy: (_button, callback) => callback(), toast: () => {},
  };
  vm.runInNewContext([
    section('  const el = ', '\n  const cleanFictionalPrefix'),
    section('  function sourceDetails(', '\n  const memoryTypeLabels'),
    section('  function displayTimezone()', '\n  function renderTimezonePreference'),
    section('  function renderSessionSummaries(', '\n  function appendSelectionWarnings'),
    'globalThis.loadMemory = loadMemory; globalThis.renderSessionSummaries = renderSessionSummaries;',
  ].join('\n'), context);
  return { context, nodes, state, data, posts, gets };
}

async function verifyChatMethod(method, learningAction = undefined, checkinId = undefined) {
  const pending = { memory_method: method, content: 'Remember our last lesson?', client_message_id: 'message', learning_action: learningAction, checkin_id: checkinId };
  const state = { learnerId: 'child', token: 'token', pendingMessages: new Map([['child', pending]]),
    learningSessionsByLearner: new Map([['child', { status: 'active', learning_session_id: 'active', subject_id: 'math', topic_id: 'math' }]]) };
  let body;
  const context = { state, $: () => ({}), autonomousAvailable: () => false, renderLearningSession: () => {},
    autonomousProtocol: 'autonomous', guidedViewRevision: 0, chatReturnAttempt: null, guidedContexts: new Map(),
    request: async (_path, options) => { body = options.body; return { choices: [{ message: { content: 'Reply' } }] }; },
    updateChat: () => {}, refreshGuided: () => {}, toast: () => {}, errorText: error => error.message };
  vm.runInNewContext(section('  async function sendPendingMessage()', '\n  async function loginWith(') + '\nglobalThis.send = sendPendingMessage;', context);
  await context.send();
  assert.equal(body.metadata.memory_method, learningAction ? undefined : method);
  assert.equal(body.metadata.checkin_id, learningAction ? undefined : checkinId);
  assert.equal(body.metadata.approved_source_ids, undefined, 'UI must not invent a source grant');
  assert.equal(body.metadata.session_summaries, undefined, 'UI must not inject teacher summaries into chat');
}

(async () => {
  const draft = view();
  await draft.context.loadMemory();
  assert.equal(draft.posts.length, 0, 'Viewing memory cannot confirm it');
  assert(draft.gets[0].includes('include_personal_graph_snapshot=true'), 'Practical memory reuses the source snapshot provider');
  const summaries = draft.nodes['#memory-summaries'];
  assert.equal(summaries.children.length, 2);
  const content = text(summaries);
  for (const value of ['Последнее занятие', 'Последние семь занятий', 'Занятий: 1', 'Источников: 1', '2026', 'Учитель', 'исправление записи', 'original (заменена)', 'fix (текущая)', 'Слова помощника не подтверждают', '<script>literal full correction</script>', 'не вошло реплик: 2']) assert(content.includes(value), value);
  assert(!content.includes('this_is_not_user_facing'));
  assert(!text(draft.nodes['#memory-history']).includes('internal_graph'), 'Internal serialized memory is not user-facing');
  const open = descendants(summaries).find(node => node.tag === 'button' && node.textContent === 'Открыть исходную реплику');
  open.listeners.click();
  const openedSource = descendants(summaries).find(node => node.id === 'summary-source-last_session-fix');
  assert(openedSource.scrolled && openedSource.focused, 'Source navigation must open and focus its original');
  const sourcePanel = descendants(summaries).find(node => node.tag === 'details');
  assert.equal(sourcePanel.open, true);

  let all = descendants(draft.nodes['#memory-history']);
  const preview = all.find(node => node.textContent === 'Показать весь доступный снимок исходников памяти');
  assert(preview, 'The practical mode must provide the same explicit archive review as the experiment');
  await preview.listeners.click();
  all = descendants(draft.nodes['#memory-history']);
  const checks = all.filter(node => node.tag === 'input' && node.type === 'checkbox');
  assert.equal(checks.length, 2);
  checks[0].checked = true; checks[0].listeners.change(); checks[1].checked = true;
  await all.find(node => node.textContent === 'Подтвердить память для помощника').listeners.click();
  assert.equal(draft.posts[0].allow_full_archive, true);
  assert.equal(draft.posts[0].reviewed_full_archive, true);
  assert.equal(draft.posts[0].archive_snapshot_fingerprint, digest);

  const selected = view(); await selected.context.loadMemory();
  await descendants(selected.nodes['#memory-history']).find(node => node.textContent === 'Подтвердить память для помощника').listeners.click();
  assert.equal(selected.posts[0].allow_full_archive, false, 'A normal confirmation only grants selected sources');
  assert.equal(selected.posts[0].archive_snapshot_fingerprint, undefined);

  const approved = view({ approved: true, fullArchive: true }); await approved.context.loadMemory();
  all = descendants(approved.nodes['#memory-history']);
  assert(!all.some(node => node.textContent === 'Подтвердить память для помощника'));
  await all.find(node => node.textContent === 'Показать разрешённый снимок исходников памяти').listeners.click();
  assert(text(approved.nodes['#memory-history']).includes('Other lesson'), 'A saved full grant remains inspectable');
  assert(text(approved.nodes['#memory-history']).includes('сводки в пределах просмотренного снимка памяти'));

  const empty = view(); empty.context.renderSessionSummaries({ session_summaries: {} });
  assert.equal(empty.nodes['#memory-summaries'].children.length, 2);
  assert.equal((text(empty.nodes['#memory-summaries']).match(/Завершённых занятий пока нет/g) || []).length, 2);

  const heuristic = view();
  heuristic.context.renderSessionSummaries({ session_summaries: { last_session: { ...summary, sessions: [{ ...session, ending: null,
    facts: [
      { source_id: 'interest', role: 'student', kind: 'interest', kind_origin: 'lexical_heuristic', text: 'Я не люблю дроби.' },
      { source_id: 'uncertainty', role: 'student', kind: 'uncertainty', kind_origin: 'lexical_heuristic', text: 'Я не знаю, что значит слово «сомневаюсь».' },
    ] }] } } });
  const heuristicContent = text(heuristic.nodes['#memory-summaries']);
  for (const value of ['упоминание интереса', 'упоминание неуверенности', 'Я не люблю дроби.', 'Я не знаю, что значит слово «сомневаюсь».']) assert(heuristicContent.includes(value), value);
  assert.equal((heuristicContent.match(/Автоматическая метка по словам/g) || []).length, 2, 'Lexical categories must remain explicit automatic labels beside literal quotations');
  assert(!heuristicContent.includes('неуверенность в реплике'));
  for (const method of ['personal_memory', 'personal_graph', 'tail']) await verifyChatMethod(method);
  await verifyChatMethod('personal_memory', { operation: 'division_check' });
  await verifyChatMethod('tail', undefined, 'daily-question');
  console.log('Personal memory UI checks passed (production render/dispatch, simulated DOM).');
})().catch(error => { console.error(error); process.exitCode = 1; });
