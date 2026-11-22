const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const script = fs.readFileSync('web/app.js', 'utf8');
function section(start, end) {
  const from = script.indexOf(start), to = script.indexOf(end, from);
  assert(from >= 0 && to > from, start);
  return script.slice(from, to);
}

class Element {
  constructor(tag) { this.localName = tag; this.children = []; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  set textContent(text) { this.replaceChildren(text); }
  get textContent() {
    return this.children.map(child => typeof child === 'string' ? child : child.textContent).join('');
  }
  set innerHTML(_html) { assert.fail('Diagnostics must render excerpts as text'); }
}

const results = new Element('div');
const context = { document: { createElement: tag => new Element(tag) }, $: () => results };
vm.runInNewContext([
  section('  const el = ', '\n  const cleanFictionalPrefix'),
  section('  function appendSelectionWarnings(', '\n  async function loadMemoryDiagnostics('),
  section('  function renderDiagnosticsQuoteCheck(', '\n  function renderNextStep('),
  'globalThis.render = renderDiagnosticsRetrieval;',
].join('\n'), context);

const candidates = [
  { source_id: 'student-1', text: 'Про круг на 8 частей.' },
  { source_id: 'student-2', text: '<img src=x onerror=alert(1)>' },
];
context.render({ quote_check: { status: 'unique', author: 'user', selected_source_id: 'student-1', candidates } });
let card = results.children[0];
assert.equal(card.children[0].textContent, 'Проверка дословной цитаты');
assert.match(card.textContent, /Автор цитаты – ученик/);
assert.match(card.textContent, /Одна реплика лучше всего/);
assert.equal(card.children.find(node => node.localName === 'blockquote').textContent, 'student-1 · Про круг на 8 частей.');
assert.equal(card.children.find(node => node.localName === 'ul').children[1].textContent, 'student-2 · <img src=x onerror=alert(1)>');
assert.match(results.children[1].textContent, /Каналы поиска/);

context.render({ quote_check: { status: 'ambiguous', author: 'assistant', selected_source_id: null, candidates } });
card = results.children[0];
assert.match(card.textContent, /Автор цитаты – помощник/);
assert.match(card.textContent, /Несколько похожих реплик/);
assert.doesNotMatch(card.textContent, /Выбранная реплика/);
assert.equal(card.children.find(node => node.localName === 'ul').children.length, 2);

context.render({ status: 'not_approved', hint: 'Подтвердите память.',
  quote_check: { status: 'unavailable', author: 'user', selected_source_id: null, candidates: [] } });
assert.match(results.children[0].textContent, /Нет подходящей реплики/);
assert.equal(results.children[1].textContent, 'Подтвердите память.');

context.render({ quote_check: { status: 'not_requested' } });
assert.equal(results.children[0].children[1].textContent, 'Это не просьба о дословной цитате');
assert.equal(results.children[0].children.length, 2);
