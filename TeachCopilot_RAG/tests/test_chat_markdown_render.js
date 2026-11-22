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
  set innerHTML(_html) { assert.fail('Assistant replies must use text and DOM nodes'); }
  scrollIntoView() {}
}

const chatLog = new Element('div');
const state = { learnerId: 'learner', chatByLearner: new Map() };
const context = { document: { createElement: tag => new Element(tag) }, state, $: () => chatLog };
vm.runInNewContext([
  section('  const el = ', '\n  const cleanFictionalPrefix'),
  section('  function maskAssistantQuotes(', '\n  function ownArchiveCurrent('),
  section('  function chatBubble(', '\n  function citationLabel('),
  'globalThis.ui = { chatBubble, updateChat };',
].join('\n'), context);

const { chatBubble, updateChat } = context.ui;
const reply = text => chatBubble('assistant', text).children[0];
const elements = node => node.children.filter(child => typeof child !== 'string');
const tags = node => elements(node).map(child => child.localName);
const descendants = node => [node, ...elements(node).flatMap(descendants)];

const sample = [
  '# Первый шаг', '## Подсказка', '### Проверка',
  'Возьми **3**, *умножь* и _проверь_ через `print(3 * 2)`.', 'Вторая строка.',
  '', 'Следующий абзац.', '',
  '- Первый пункт', '* Второй пункт', '• Третий пункт', '1. Сначала', '2) Потом',
].join('\r\n');
const formatted = reply(sample);
assert.deepEqual(tags(formatted), ['p', 'p', 'p', 'p', 'p', 'ul', 'ol']);
const [firstHeading, secondHeading, thirdHeading, paragraph, secondParagraph, unordered, ordered] = formatted.children;
for (const heading of [firstHeading, secondHeading, thirdHeading]) assert.deepEqual(tags(heading), ['strong']);
assert.equal(firstHeading.textContent, 'Первый шаг');
assert.deepEqual(tags(paragraph), ['strong', 'em', 'em', 'code']);
assert.equal(paragraph.textContent, 'Возьми 3, умножь и проверь через print(3 * 2).\nВторая строка.');
assert.equal(secondParagraph.textContent, 'Следующий абзац.');
assert.deepEqual(tags(unordered), ['li', 'li', 'li']);
assert.deepEqual(unordered.children.map(item => item.textContent), ['Первый пункт', 'Второй пункт', 'Третий пункт']);
assert.deepEqual(tags(ordered), ['li', 'li']);
assert.deepEqual(ordered.children.map(item => item.textContent), ['Сначала', 'Потом']);
assert.equal(reply('3) Начинаем с трёх').children[0].start, 3);

const literal = '<img src=x onerror=alert(1)> [**ссылка**](javascript:alert(1))';
const untrusted = reply(literal);
assert.equal(untrusted.textContent, literal);
assert.deepEqual(descendants(untrusted).map(node => node.localName), ['div', 'p']);
const unclosed = '**незакрытый маркер и `код без конца; имя some_python_var';
assert.equal(reply(unclosed).textContent, unclosed);
assert.deepEqual(tags(reply(unclosed).children[0]), []);

const quote = '«мои *слова* «**буквально**»\n\n- так и есть\n# _точно_ `да`»';
const quoted = reply(`До **${quote}** после *курсив*.`);
assert.deepEqual(tags(quoted), ['p']);
assert.deepEqual(tags(quoted.children[0]), ['strong', 'em']);
assert.equal(quoted.children[0].children[1].textContent, quote);
const incompleteQuote = '«моя *фраза*\n- продолжение';
assert.equal(reply(`**Вступление** ${incompleteQuote}`).textContent, `Вступление ${incompleteQuote}`);
for (const guardReply of [
  'Вот твоя фраза дословно: «кавычка » и **текст**\r\n- продолжение»',
  'Вот мой ответ дословно (запись от 2026-11-22): «`код` и *курсив*»',
  'Нашлось несколько похожих фраз:\n1. «текст » и *фраза*»\n2. «ещё **одна**»\nКакую из них ты имеешь в виду?',
]) {
  const rendered = reply(guardReply);
  assert.equal(rendered.textContent, guardReply);
  assert.deepEqual(tags(rendered), []);
}

const user = chatBubble('user', sample);
assert.equal(user.textContent, sample);
assert.deepEqual(tags(user), []);
updateChat('assistant', sample);
assert.equal(state.chatByLearner.get('learner')[0].content, sample);
assert.equal(chatLog.children[0].textContent, formatted.textContent);

const verified = chatBubble('assistant', '**Ответ**', {
  operation: 'print_output', status: 'verified', facts: { stdout: '<img>\n**вывод**\n' },
});
assert.deepEqual(tags(verified), ['p', 'pre', 'p']);
assert.equal(verified.children[1].textContent, '<img>\n**вывод**\n');
