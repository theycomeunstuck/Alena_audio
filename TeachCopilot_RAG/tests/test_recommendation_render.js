// Exercise the actual rendering functions with a small DOM, without a browser.
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
    this.textContent = '';
    this.className = '';
    this.classList = { add: value => { this.className += ` ${value}`; } };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener() {}
}

const ids = ['#next-step-result', '#next-step-status', '#request-next-step'];
const nodes = Object.fromEntries(ids.map(id => [id, new Element('div')]));
const reason = 'Исправил после вопроса учителя.';
const data = {
  action: 'clarify_missing_observation', reason, unknowns: [], topic_id: 'math',
  review_required: true,
  memory_selection_warnings: [{code: 'withheld_without_exact_source_chunk'}],
  evidence_review: {
    version: 'recommendation-source-review-v1', attribution_checks: [{quote: 'вопроса учителя'}],
    attribution_contexts: [{source_id: 'other', role: 'student', text: '<img src=x>', cited: false, corrected_by: ['fix'],
      dialogue_context: {role: 'assistant', usage: 'context_only', text: 'Сколько частей?'},
      session_end_context: {role: 'assistant', usage: 'context_only', text: 'Продолжим?'}},
      {source_id: 'fix', role: 'teacher', text: 'Была подсказка.', cited: false, corrects_source_id: 'other'}],
    sources: [{source_id: 'cited', role: 'student', text: 'Хз.', cited: true,
      session_end_context: {role: 'assistant', usage: 'context_only', text: 'Почему так?'}}]
  }
};
const state = {learnerId: 'child', revision: 1,
  nextStepsByLearner: new Map([['child', {result: data, finished: true}]])};
const context = {
  document: {createElement: tag => new Element(tag)}, state,
  $: selector => nodes[selector], topicTitle: row => row.topic_id || '', topicById: () => null,
  safeDate: value => value || '', memorySourceDomId: id => `memory-source-${id}`,
  Array, Object, String
};
const el = section('  const el = ', '\n  const cleanFictionalPrefix');
const sources = section('  function sourceDetails(', '\n  const memoryTypeLabels');
const selectionWarnings = section('  function appendSelectionWarnings(', '\n  function renderNextStep()');
const render = section('  function renderNextStep()', '\n  async function pollNextStep');
vm.runInNewContext(`${el}\n${sources}\n${selectionWarnings}\n${render}\nrenderNextStep();`, context);

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
const all = descendants(nodes['#next-step-result']);
const text = all.map(node => node.textContent).join('\n');
assert(text.includes(reason));
assert(text.includes('Это предупреждение не определяет, есть ли ошибка.'));
assert(text.includes('Другие выбранные диалоги могут относиться к другим событиям.'));
assert(text.includes('В ссылках черновика не указана.'));
assert(text.includes('Сколько частей?') && text.includes('Продолжим?') && text.includes('Почему так?'));
assert(text.includes('Запись исправлена: fix. Сверьте её с исправлениями.'));
assert(text.includes('Исправляет источник other. Исходная запись сохранена.') && text.includes('Была подсказка.'));
assert(text.includes('их соответствие запросу не установлено'));
assert(!text.includes('из-за ограничения объёма'), 'A graph-only warning must not be described as a budget failure');
assert(all.some(node => node.textContent === '<img src=x>'));
assert(!all.some(node => node.tag === 'img'));
const extra = all.find(node => node.tag === 'article' && node.children.some(child => child.textContent === 'other'));
assert(extra && !extra.id, 'Additional context must not create duplicate source IDs');
vm.runInNewContext(`globalThis.archivePreview = sourceDetails([{
  source_id: 'archive-answer', author_role: 'student', topic_id: 'math', observed_on: '2026-01-01',
  text: 'Student answer',
  dialogue_context: {role: 'assistant', usage: 'context_only', text: 'What did you try?', excerpt_mode: 'tail', truncated: false}
}], 'Full archive', {showDialogueContext: true, suppressId: true});`, context);
assert(descendants(context.archivePreview).map(node => node.textContent).join('\n').includes('What did you try?'),
  'A full archive preview must render its attached assistant question');
data.memory_selection_warnings = [
  {code: 'memory_budget_exceeded'}, {code: 'assistant_context_not_source_evidence'},
  {code: 'duplicate_source_turns'}, {code: 'retrieved_source_fragment_not_hydrated'}, {code: 'unknown'}
];
vm.runInNewContext('renderNextStep();', context);
const warningText = descendants(nodes['#next-step-result']).map(node => node.textContent).join('\n');
assert(warningText.includes('из-за ограничения объёма'));
assert(warningText.includes('не используются как самостоятельное свидетельство'));
assert(warningText.includes('пересекаются с уже выбранными'));
assert(warningText.includes('не были подгружены для проверки'));
assert(warningText.includes('Часть найденных фрагментов не вошла в память.'));
console.log('Recommendation rendering checks passed (simulated DOM, not a visual browser test).');
