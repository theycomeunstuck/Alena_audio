"""Печатная форма: что видит ребёнок и что остаётся у взрослого."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import make_task_card  # noqa: E402
from pipeline.learner_context import list_learner_ids  # noqa: E402
from pipeline.task_card_html import render_cards_html  # noqa: E402
from pipeline.task_designer import Hint, Task, TaskCard  # noqa: E402


def _card(pseudonym="Тёма", story_intro="Тёма, проверь чертежи ракеты."):
    return TaskCard(
        learner_id="zvezdin-artem",
        pseudonym=pseudonym,
        topic_id="math.g3.geometry.point_line_ray_segment",
        topic_title="Точка, прямая, луч, отрезок",
        zone="zpd",
        zone_ru="в ЗБР (делает с опорой)",
        difficulty="medium",
        hint_depth=2,
        story_title="Чертежи ракеты",
        story_intro=story_intro,
        tasks=(
            Task(
                statement="Начерти луч от точки старта.",
                expected_answer="Луч с одним началом, второй конец не ставим.",
                checks_error="рисует у луча два конца",
                hints=(
                    Hint(1, "hint_question", "А конец у полёта есть?"),
                    Hint(2, "visual", "Вспомни фонарик."),
                ),
                materials=("линейка",),
            ),
        ),
        reflection_question="Что стало понятно?",
        tutor_notes="Смотреть на второй конец.",
        sources=("math_g3_geometry_lines_demo.json",),
        warnings=("подсказка 1 содержит готовый ответ",),
    )


def test_child_sheet_hides_answers_hints_and_topic_id():
    html = render_cards_html([_card()])
    child_part = html.split("Лист для взрослого")[0]

    assert "Начерти луч от точки старта." in child_part
    assert "Луч с одним началом" not in child_part
    assert "Вспомни фонарик" not in child_part
    assert "math.g3.geometry.point_line_ray_segment" not in child_part


def test_tutor_sheet_carries_answers_ladder_zone_and_warnings():
    html = render_cards_html([_card()])
    tutor_part = html.split("Лист для взрослого")[1]

    assert "Луч с одним началом" in tutor_part
    assert "Ступень 1" in tutor_part and "Ступень 2" in tutor_part
    assert "в ЗБР (делает с опорой)" in tutor_part
    assert "подсказка 1 содержит готовый ответ" in tutor_part


def test_group_gets_child_sheets_first_then_tutor_sheets():
    html = render_cards_html([_card(pseudonym="Тёма"), _card(pseudonym="Белка")])

    assert html.count('class="sheet"') == 4
    first_tutor = html.index("Лист для взрослого")
    assert html.count("Индивидуальное задание", 0, first_tutor) == 2


def test_model_text_is_escaped_not_injected():
    html = render_cards_html([_card(story_intro="<script>alert(1)</script>")])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_empty_input_renders_a_valid_page():
    html = render_cards_html([])
    assert html.startswith("<!doctype html>") and "Карточек нет" in html


def test_all_flag_resolves_to_every_learner():
    args = make_task_card.argparse.Namespace(all=True, learners=None, learner_id=None)
    assert make_task_card._resolve_learner_ids(args) == list_learner_ids()


def test_learners_flag_is_split_and_trimmed():
    args = make_task_card.argparse.Namespace(all=False, learners=" a , b ,, c ", learner_id=None)
    assert make_task_card._resolve_learner_ids(args) == ["a", "b", "c"]


def test_html_to_stdout_is_refused_because_powershell_breaks_the_encoding(capsys):
    code = make_task_card.main(["zvezdin-artem", "--format", "html"])

    assert code == 2
    assert "--out" in capsys.readouterr().err


def test_html_with_out_passes_the_guard(tmp_path, monkeypatch):
    """Проверяем только, что защита пропускает корректный вызов дальше."""
    def explode(*_args, **_kwargs):
        raise make_task_card.TaskDesignError("до модели дошли")

    monkeypatch.setattr(make_task_card, "design_task_card", explode)
    code = make_task_card.main(["zvezdin-artem", "--format", "html", "--out", str(tmp_path / "c.html")])

    assert code == 1  # дошли до генерации и упали там, а не на проверке аргументов
