// Calendar memory, student defaults and optional check-in using production UI code.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('web/app.js', 'utf8');
function section(start, end) {
  const from = script.indexOf(start), to = script.indexOf(end, from);
  assert(from >= 0 && to > from, start); return script.slice(from, to);
}
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.listeners = {}; this.textContent = ''; this.dataset = {}; this.value = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute(name, value) { this[name] = value; }
  focus() { this.focused = true; }
  scrollIntoView() {}
}
const descendants = node => node && typeof node === 'object' ? [node, ...(node.children || []).flatMap(descendants)] : [];
const text = node => descendants(node).map(item => item.textContent).join('\n');
async function verifyDefault(role, enabled = true) {
  const nodes = {}, state = { token: 'auth' };
  const $ = (selector, root) => {
    if (root) return root[selector] ||= new Element('option');
    return nodes[selector] ||= new Element();
  };
  const context = { state, $, request: async () => ({ account: { role }, features: { personal_memory: enabled } }),
    renderTimezonePreference: () => {}, displayName: () => 'Name', loadLearners: async () => {}, loadCatalog: async () => {},
    showLogin: error => { throw new Error(error); }, errorText: error => error.message };
  vm.runInNewContext(section('  async function boot()', '\n  async function loadLearners()') + '\nglobalThis.boot = boot;', context);
  await context.boot();
  assert.equal(nodes['#memory-method'].value, enabled ? 'personal_memory' : 'tail');
  assert.equal(nodes['#chat-memory-method'].value, 'tail', `${role}: an absent approval must not block the first normal question`);
}
function setup() {
  const nodes = {}, requests = [], dailyCheckins = new Map(), ownArchiveViews = new Map();
  const $ = selector => nodes[selector] ||= new Element();
  const state = { learnerId: 'child', token: 'auth', viewRevision: 1, account: { role: 'student' },
    profile: { display_timezone: 'Asia/Tomsk' }, features: { conversation_memory: true, daily_checkin: true },
    learningSessionsByLearner: new Map(), pendingMessages: new Map() };
  const context = { state, $, dailyCheckins, ownArchiveViews, checkinObserver: null,
    document: { createElement: tag => new Element(tag), visibilityState: 'visible', addEventListener: () => {},
      getElementById: id => Object.values(nodes).flatMap(descendants).find(node => node.id === id) },
    IntersectionObserver: class {
      constructor(callback) { this.callback = callback; }
      observe() {} disconnect() {}
    },
    request: async (path, options) => {
      requests.push({ path, body: options?.body });
      return { checkin_id: 'daily', status: path.endsWith('/skip') ? 'skipped' : 'shown', available: !path.endsWith('/skip'), text: 'Что тебя сегодня порадовало?', shown_at: '2026-11-01T20:05:01Z', assistant_source_id: 'server-question-source' };
    },
    URLSearchParams, errorText: error => error.message, recoveryBlocksMessages: () => false,
    renderLearningSession: () => {}, updateChat: () => {}, sendPendingMessage: () => {},
    busy: (_button, callback) => callback(), loadConversationMemory: () => {},
    // The extracted wiring also binds the diagnostics refresh button; the
    // diagnostics panel itself is out of scope for this harness.
    loadMemoryDiagnostics: () => {},
    loadLearningArchive: () => {}, crypto: { randomUUID: () => 'client-id' },
  };
  vm.runInNewContext([
    section('  const el = ', '\n  const cleanFictionalPrefix'),
    section('  function displayTimezone()', '\n  function renderTimezonePreference'),
    section('  function renderConversationMemory(', '\n  async function loadConversationMemory'),
    section('  function currentDailyCheckin(', '\n  function renderSessionSummaries'),
    section("  $('#refresh-conversation-memory')", "\n  $('#learner-filter')"),
    'globalThis.renderConversationMemory = renderConversationMemory; globalThis.formatDisplayDate = formatDisplayDate; globalThis.renderDailyCheckin = renderDailyCheckin; globalThis.markDailyCheckinShown = markDailyCheckinShown; globalThis.openOwnConversationSource = openOwnConversationSource;',
  ].join('\n'), context);
  return { context, nodes, $, state, requests, dailyCheckins };
}
(async () => {
  await verifyDefault('student'); await verifyDefault('teacher'); await verifyDefault('teacher', false);
  const v = setup(), instant = '2026-11-01T20:05:01.123Z';
  assert(v.context.formatDisplayDate(instant, { hour:'2-digit', minute:'2-digit' }).includes('03:05'));
  assert(v.context.formatDisplayDate(instant).startsWith('2 '));
  v.state.profile.display_timezone = 'Europe/Moscow';
  assert(v.context.formatDisplayDate(instant, { hour:'2-digit', minute:'2-digit' }).includes('23:05'));
  assert(v.context.formatDisplayDate(instant).startsWith('1 '));
  v.state.profile.display_timezone = 'America/Los_Angeles';
  assert(v.context.formatDisplayDate('2026-11-02').startsWith('2 '), 'An immutable activity_day must not be regrouped by display timezone');
  const days = ['2026-09-01', '2026-09-04', '2026-09-20', '2026-10-15', '2026-10-30', '2026-11-01', '2026-11-02'].map(day => ({
    available:true, day, timezone:'Asia/Tomsk', timezones:['Asia/Tomsk'], source_count:2, facts:[{ source_id:day, role:'user', text:`Literal ${day}`, created_at:instant, truncated:true }],
  }));
  v.context.renderConversationMemory({ learning_timezone:'Asia/Tomsk',
    recent_messages: [{ source_id:'message', role:'assistant', content:'<img literal>', created_at:instant }],
    last_day:days[6], last_seven_active_days:{ available:true, count:7, days },
    system_text:'SECRET SYSTEM', provenance:{ private:'PRIVATE JSON' }, budget:{ omitted_recent_message_count:2, omitted_day_count:1 } });
  const cards = v.$('#conversation-memory-cards'), content = text(cards);
  assert.equal(cards.children.length, 3);
  for (const value of ['Последний активный день', 'Последние семь активных дней', 'Дней: 7', 'Literal 2026-09-01', 'Literal 2026-11-02', 'Полный архив сохраняется без этого ограничения', 'Помощник – только контекст', '<img literal>', 'не вошло последних сообщений: 2', 'не поместилось сводок дней: 1']) assert(content.includes(value), value);
  assert(!content.includes('SECRET SYSTEM') && !content.includes('PRIVATE JSON'));
  const time = descendants(cards).find(node => node.tag === 'time');
  assert.equal(time.dateTime, instant, 'Exact server timestamp remains available beside viewer-local formatting');

  const requestCheckin = v.context.request;
  const historyPaths = [];
  v.context.request = async path => {
    historyPaths.push(path);
    const older = path.includes('before_source_order');
    return { learner_id:'child', messages:older ? [{ source_id:'older', role:'assistant', content:'Earlier standalone daily question', created_at:instant, source_order:1 }]
      : Array.from({ length:50 }, (_, index) => ({ source_id:`message-${index}`, role:'user', content:`Full message ${index}`, created_at:instant, source_order:index+51 })),
      next_before_source_order:older ? null : 51 };
  };
  await v.context.openOwnConversationSource('older');
  assert.equal(historyPaths.length, 2);
  assert(historyPaths[1].includes('before_source_order=51'));
  assert.equal(descendants(v.$('#own-conversation-history-log')).filter(node => node.tag === 'article').length, 51, 'Raw history pagination must preserve more than the model twenty-message window');
  assert(text(v.$('#own-conversation-history-log')).includes('Earlier standalone daily question'));
  assert(descendants(v.$('#own-conversation-history-log')).find(node => node.id === 'own-source-older').focused);
  v.context.request = requestCheckin;

  const offered = () => ({ learnerId:'child', token:'auth', revision:1, data:{ checkin_id:'daily', available:true, status:'offered', text:'Что тебя сегодня порадовало?' } });
  let view = offered(); v.dailyCheckins.set('child', view); v.context.renderDailyCheckin();
  assert.equal(v.$('#daily-checkin').hidden, false);
  assert.equal(v.requests.length, 0, 'Rendering an offscreen prompt must not fabricate a shown archive record');
  v.context.checkinObserver.callback([{ isIntersecting:false }]); assert.equal(v.requests.length, 0);
  v.context.checkinObserver.callback([{ isIntersecting:true }]); await view.showing;
  assert.equal(view.data.status, 'shown');
  assert.deepEqual(v.requests.map(item => item.path), ['/learners/child/checkin/daily/shown']);
  assert.equal(Object.keys(v.requests[0].body).length, 0, 'A page view must not create a technical lesson');
  v.$('#daily-checkin-answer').value = 'Построил башню.';
  await v.$('#daily-checkin-form').listeners.submit({ preventDefault() {} });
  assert.equal(v.state.pendingMessages.get('child').checkin_id, 'daily');
  assert.equal(v.state.pendingMessages.get('child').memory_method, 'tail', 'Optional own answer requires no teacher grant');
  assert.equal(v.state.pendingMessages.get('child').content, 'Построил башню.');

  v.state.pendingMessages.clear();
  view = offered(); v.dailyCheckins.set('child', view);
  let resolveShown;
  v.context.request = async path => path.endsWith('/shown') ? await new Promise(resolve => { resolveShown = resolve; }) : { ...view.data, status:'skipped', available:false };
  const showing = v.context.markDailyCheckinShown(view, true);
  await v.$('#skip-daily-checkin').listeners.click();
  resolveShown({ ...view.data, status:'shown', available:true }); await showing;
  assert.equal(view.data.status, 'skipped', 'An older shown response cannot reopen a concurrently skipped question');
  assert.equal(v.$('#daily-checkin').hidden, true);
  console.log('Alena calendar memory/timezone/check-in UI checks passed (simulated DOM).');
})().catch(error => { console.error(error); process.exitCode = 1; });
