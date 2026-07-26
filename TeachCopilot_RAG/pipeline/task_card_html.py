"""Печатная форма карточек-заданий: лист ребёнку, лист взрослому.

Четвёртый этап урока раздаётся всему классу сразу, поэтому карточки нужны на
бумаге. Раскладка простая и намеренно скучная: A4, крупный шрифт, разрыв
страницы после каждой карточки — чтобы можно было разрезать и раздать.

Порядок листов: сначала все детские карточки (их печатают и отдают), потом все
листы для взрослого (ответы, лестница подсказок, зона по карте компетенций и
замечания к сгенерированному тексту). Так стопку не нужно разбирать вручную.

Файл самодостаточный: стили внутри, картинок и внешних ссылок нет.
"""
from __future__ import annotations

from html import escape
from typing import Sequence

from pipeline.task_designer import TaskCard

_STYLES = """
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 0;
    background: #f4f4f5;
    color: #18181b;
    font-family: "Segoe UI", "PT Sans", system-ui, sans-serif;
    font-size: 15px;
    line-height: 1.5;
  }
  .sheet {
    width: 210mm;
    min-height: 297mm;
    margin: 0 auto 12mm;
    padding: 18mm 16mm;
    background: #fff;
    box-shadow: 0 1px 6px rgba(0,0,0,.12);
  }
  .sheet + .sheet { page-break-before: always; }
  .kicker { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: #71717a; margin: 0 0 6px; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  h2 { font-size: 19px; margin: 0 0 10px; }
  .for-whom { font-size: 17px; font-weight: 600; margin: 0 0 14px; }
  .story { font-size: 17px; margin: 0 0 20px; padding: 12px 14px; border-left: 4px solid #a1a1aa; background: #fafafa; }
  ol.tasks { margin: 0; padding-left: 22px; }
  ol.tasks > li { margin-bottom: 20px; font-size: 17px; }
  .needs { font-size: 13px; color: #52525b; margin-top: 4px; }
  .worksheet { margin-top: 8px; border-bottom: 1px dashed #a1a1aa; height: 26mm; }
  .reflection { margin-top: 24px; font-size: 16px; font-style: italic; }
  .meta { margin: 0 0 16px; font-size: 13px; color: #52525b; }
  .answer { margin: 4px 0 0; font-size: 14px; }
  .answer b { font-weight: 600; }
  .hints { margin: 6px 0 0; padding-left: 18px; font-size: 14px; color: #3f3f46; }
  .hints li { margin-bottom: 3px; }
  .checks { font-size: 13px; color: #7c2d12; margin-top: 4px; }
  .warnings { margin-top: 18px; padding: 10px 12px; border: 1px solid #d4d4d8; font-size: 13px; }
  .warnings ul { margin: 6px 0 0; padding-left: 18px; }
  .empty { color: #71717a; font-style: italic; }
  @media print {
    body { background: #fff; }
    .sheet { margin: 0; box-shadow: none; width: auto; min-height: auto; padding: 12mm 10mm; }
  }
"""


def _tasks_for_child(card: TaskCard) -> str:
    items = []
    for task in card.tasks:
        parts = [escape(task.statement)]
        if task.materials:
            parts.append(f'<div class="needs">Понадобится: {escape(", ".join(task.materials))}</div>')
        # Пустая линейка под ответ: карточку заполняют прямо на листе.
        parts.append('<div class="worksheet"></div>')
        items.append("<li>" + "".join(parts) + "</li>")
    return "<ol class=\"tasks\">" + "".join(items) + "</ol>"


def _child_sheet(card: TaskCard) -> str:
    story = f'<p class="story">{escape(card.story_intro)}</p>' if card.story_intro else ""
    reflection = (
        f'<p class="reflection">{escape(card.reflection_question)}</p>' if card.reflection_question else ""
    )
    return (
        '<section class="sheet">'
        '<p class="kicker">Индивидуальное задание</p>'
        f"<h1>{escape(card.story_title)}</h1>"
        f'<p class="for-whom">Для: {escape(card.pseudonym)}</p>'
        f"{story}{_tasks_for_child(card)}{reflection}"
        "</section>"
    )


def _tutor_sheet(card: TaskCard) -> str:
    items = []
    for task in card.tasks:
        parts = [f"<b>{escape(task.statement)}</b>"]
        if task.expected_answer:
            parts.append(f'<p class="answer">Ответ: {escape(task.expected_answer)}</p>')
        if task.checks_error:
            parts.append(f'<p class="checks">Проверяет ошибку: {escape(task.checks_error)}</p>')
        if task.hints:
            hints = "".join(
                f"<li>Ступень {hint.level} ({escape(hint.type)}): {escape(hint.text)}</li>"
                for hint in task.hints
            )
            parts.append(f'<ol class="hints">{hints}</ol>')
        items.append("<li>" + "".join(parts) + "</li>")

    warnings = ""
    if card.warnings:
        warning_items = "".join(f"<li>{escape(item)}</li>" for item in card.warnings)
        warnings = f'<div class="warnings"><b>Замечания к сгенерированному тексту:</b><ul>{warning_items}</ul></div>'

    notes = f"<p>{escape(card.tutor_notes)}</p>" if card.tutor_notes else ""
    sources = f"<p class=\"meta\">Материал: {escape(', '.join(card.sources))}</p>" if card.sources else ""

    return (
        '<section class="sheet">'
        '<p class="kicker">Лист для взрослого — не выдавать ребёнку</p>'
        f"<h1>{escape(card.pseudonym)}: {escape(card.story_title)}</h1>"
        f'<p class="meta">Тема: {escape(card.topic_title)} [{escape(card.topic_id)}]<br>'
        f"Зона: {escape(card.zone_ru)} · сложность: {escape(card.difficulty)} · "
        f"ступеней подсказок: {card.hint_depth}</p>"
        "<h2>Как помогать</h2>"
        "<p>Подсказки открываются по одной и только если ребёнок застрял. "
        "Как только он справился — опоры убираем.</p>"
        f"<ol class=\"tasks\">{''.join(items)}</ol>"
        f"{notes}{sources}{warnings}"
        "</section>"
    )


def render_cards_html(cards: Sequence[TaskCard], title: str = "Индивидуальные задания") -> str:
    """Собрать печатную страницу: детские карточки, затем листы для взрослого."""
    if not cards:
        body = '<section class="sheet"><p class="empty">Карточек нет.</p></section>'
    else:
        body = "".join(_child_sheet(card) for card in cards) + "".join(_tutor_sheet(card) for card in cards)

    return (
        "<!doctype html>\n"
        '<html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{_STYLES}</style></head><body>{body}</body></html>\n"
    )
