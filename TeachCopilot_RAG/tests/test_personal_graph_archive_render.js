// Exercise the personal graph archive review and saved permission in the actual UI code.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const script = fs.readFileSync('web/app.js', 'utf8');
function section(start, end) {
  const from = script.indexOf(start), to = script.indexOf(end, from);
  assert(from >= 0 && to > from, `Cannot locate ${start}`);
  return script.slice(from, to);
}

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.textContent = '';
    this.className = '';
    this.disabled = false;
    this.checked = false;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  setAttribute() {}
}

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
function labelText(node) { return descendants(node).map(item => item.textContent).join('\n'); }

const digest = 'a'.repeat(64);
const archiveSources = [
  {source_id: 'selected', author_role: 'student', text: 'Current answer', topic_id: 'math', observed_on: '2026-01-01',
    dialogue_context: {role: 'assistant', usage: 'context_only', text: 'What did you try?', excerpt_mode: 'tail', truncated: false}},
  {source_id: 'archive-old', author_role: 'teacher', approved: true, text: 'Older approved note', topic_id: 'reading', observed_on: '2025-12-01'},
];
const excludedSource = {source_id: 'failed-source', author_role: 'student', text: 'Failed source omitted from graph', topic_id: 'math', observed_on: '2026-01-02'};
const allIds = archiveSources.map(source => source.source_id).sort();

function runMemoryView({approved = false, fullArchive = false} = {}) {
  const topicRoot = new Element('div');
  const history = new Element('div');
  const status = new Element('p');
  const nodes = {'#topic-select': {value: 'math'}, '#memory-history': history, '#memory-status': status};
  const contextData = {topic_id: 'math', text: JSON.stringify({lessons: []}), evidence_ids: ['selected'],
    sources: [archiveSources[0]], ...(fullArchive ? {source_access: {mode: 'full_archive', source_ids: allIds,
      snapshot_fingerprint: digest}} : {})};
  const version = {version_id: 'version-1', method: 'personal_graph', approved, stale: false,
    topic_id: 'math', state: {method: 'personal_graph', source_ids: allIds,
      archive_snapshot_fingerprint: digest}, context: contextData};
  const topicData = {revision: 3, source_revision: 8, sources: [archiveSources[0]],
    personal_graph_sources: archiveSources, personal_graph_snapshot_fingerprint: digest,
    personal_graph_hidden_correction_count: 1, versions: [version]};
  const allData = {revision: 3, source_revision: 8, source_snapshot_fingerprint: digest,
    sources: [...archiveSources, excludedSource], personal_graph_sources: archiveSources, personal_graph_snapshot_fingerprint: digest,
    personal_graph_hidden_correction_count: 1, versions: [version]};
  const posted = [];
  const state = {learnerId: 'child', viewRevision: 1, token: 'session', memoryViewRevision: 0,
    account: {role: 'teacher'}, features: {personal_graph: true}, memoryCorrectionJobs: new Map(),
    nextStepsByLearner: new Map(), revision: 3};
  const context = {
    document: {createElement: tag => new Element(tag)}, state, URLSearchParams,
    $: selector => nodes[selector],
    request: async (path, options = {}) => {
      if (path.includes('/confirm')) { posted.push(options.body); return {confirmed: true}; }
      if (path.endsWith('/dashboard')) return {revision: 3};
      if (path.startsWith('/learners/child/memory?topic_id=')) return topicData;
      if (path.includes('include_personal_graph_snapshot=true')) return allData;
      throw new Error(`Unexpected request ${path}`);
    },
    topicTitle: row => row.topic_id || '', safeDate: value => value || '',
    memorySourceDomId: id => `memory-source-${id}`,
    appendMemoryRecords: () => {}, appendSelectionWarnings: () => {},
    busy: (_button, callback) => callback(), toast: () => {}, renderDashboard: () => {}, renderNextStep: () => {},
    errorText: error => error.message, Object, Array, String, Map, Set, Boolean, Number,
  };
  const code = [
    script.slice(script.indexOf('  const el = '), script.indexOf('\n', script.indexOf('  const el = '))),
    section('  function sourceDetails(', '\n  const memoryTypeLabels'),
    section('  async function loadMemory()', '\n  function appendSelectionWarnings'),
    'globalThis.loadMemory = loadMemory;',
  ].join('\n');
  vm.runInNewContext(code, context);
  return {context, state, nodes, posted, loadMemory: context.loadMemory};
}

(async () => {
  const draft = runMemoryView();
  await draft.loadMemory();
  assert(labelText(draft.nodes['#memory-history']).includes('Снимок графа не содержит исправления без доступных исходных реплик: 1'),
    'The graph draft must disclose hidden correction notes before the teacher chooses access');
  let all = descendants(draft.nodes['#memory-history']);
  const previewButton = all.find(node => node.tag === 'button' && node.textContent === 'Показать весь доступный снимок исходников графа');
  assert(previewButton, `The teacher must be able to open the available graph source snapshot (${draft.nodes['#memory-status'].textContent})`);
  await previewButton.listeners.click();
  all = descendants(draft.nodes['#memory-history']);
  const previewRoot = all.find(node => node.className === 'memory-archive-preview');
  assert(labelText(previewRoot).includes('Older approved note'));
  assert(!labelText(previewRoot).includes('Failed source omitted from graph'),
    'The graph preview must use its filtered source list rather than the general memory source list');
  const checkboxes = all.filter(node => node.tag === 'input' && node.type === 'checkbox');
  assert.equal(checkboxes.length, 2);
  const [reviewed, allow] = checkboxes;
  assert.equal(reviewed.disabled, false, 'Review acknowledgment unlocks after a matching preview loads');
  reviewed.checked = true; reviewed.listeners.change();
  assert.equal(allow.disabled, false);
  allow.checked = true;
  reviewed.checked = false; reviewed.listeners.change();
  assert.equal(allow.checked, false, 'Removing the review acknowledgment clears full archive consent');
  assert.equal(allow.disabled, true);
  const confirm = all.find(node => node.tag === 'button' && node.textContent === 'Подтвердить память для помощника');
  await confirm.listeners.click();
  assert.equal(draft.posted[0].allow_full_archive, false,
    'A grant cannot be submitted after the teacher clears the review acknowledgment');
  assert.equal(draft.posted[0].reviewed_full_archive, false);
  assert.equal(draft.posted[0].archive_snapshot_fingerprint, undefined);

  const granted = runMemoryView();
  await granted.loadMemory();
  all = descendants(granted.nodes['#memory-history']);
  const grantedPreview = all.find(node => node.tag === 'button' && node.textContent === 'Показать весь доступный снимок исходников графа');
  await grantedPreview.listeners.click();
  assert(labelText(granted.nodes['#memory-history']).includes('Скрыто исправлений без доступной цепочки исходников: 1'),
    'The teacher must be told when correction notes are excluded from the graph snapshot');
  all = descendants(granted.nodes['#memory-history']);
  const grantChecks = all.filter(node => node.tag === 'input' && node.type === 'checkbox');
  grantChecks[0].checked = true; grantChecks[0].listeners.change();
  grantChecks[1].checked = true;
  const grantConfirm = all.find(node => node.tag === 'button' && node.textContent === 'Подтвердить память для помощника');
  await grantConfirm.listeners.click();
  assert.equal(granted.posted[0].allow_full_archive, true,
    'A reviewed and explicitly granted snapshot must be submitted as a full archive grant');
  assert.equal(granted.posted[0].reviewed_full_archive, true,
    'The API request must explicitly acknowledge that the teacher reviewed the snapshot');
  assert.equal(granted.posted[0].archive_snapshot_fingerprint, digest,
    'The grant must submit the fingerprint returned with its reviewed preview');

  const approved = runMemoryView({approved: true, fullArchive: true});
  await approved.loadMemory();
  all = descendants(approved.nodes['#memory-history']);
  assert(labelText(approved.nodes['#memory-history']).includes('Подтверждение разрешило чату искать исходные реплики'),
    'A confirmed full grant must be described accurately');
  const viewButton = all.find(node => node.tag === 'button' && node.textContent === 'Показать разрешённый снимок исходников графа');
  assert(viewButton, 'The teacher must be able to inspect a previously granted snapshot');
  await viewButton.listeners.click();
  assert(labelText(approved.nodes['#memory-history']).includes('Скрыто исправлений без доступной цепочки исходников: 1'),
    'The saved snapshot viewer must disclose hidden correction notes');
  assert(labelText(approved.nodes['#memory-history']).includes('Older approved note'),
    'The saved grant viewer must show sources outside the original selected context');
  assert(labelText(approved.nodes['#memory-history']).includes('What did you try?'),
    'The saved grant viewer must show attached assistant context excerpts');
  console.log('Personal graph archive review checks passed (simulated DOM, not a visual browser test).');
})().catch(error => { console.error(error); process.exitCode = 1; });
