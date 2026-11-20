"""Generate the three frozen six-session synthetic histories (client A).

Run once from the repository root:

    uv run --locked python -m tests.r2_memory_quality.generate_histories

The output files are committed and frozen; regenerating with the same seed
produces byte-identical files. Every generated dialogue turn is deterministic
seeded variation (seeded RNG, computed arithmetic answers with worked
solutions), never repeated filler. Key facts that the gold manifest references
are authored explicitly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "histories"

NAMES = ["Катя", "Мила", "Гриша", "Артём", "Соня", "Дима", "Лена", "Петя"]
# Quantities 5..20 always take the genitive plural, so one form per object is
# grammatically safe in every generated sentence.
OBJECTS = [
    {"gen": "наклеек", "name": "наклейки"},
    {"gen": "карандашей", "name": "карандаши"},
    {"gen": "марок", "name": "марки"},
    {"gen": "мячей", "name": "мячи"},
    {"gen": "конфет", "name": "конфеты"},
    {"gen": "тетрадей", "name": "тетради"},
    {"gen": "кубиков", "name": "кубики"},
    {"gen": "пуговиц", "name": "пуговицы"},
]
PRAISE = ["Верно!", "Правильно, продолжаем.", "Отлично, ещё одно.", "Хорошо. Дальше:",
          "Да, верно. Следующий пример:", "Молодец! Дальше – сложнее:", "Точно. Продолжаем:"]
RETRY = ["Почти! Смотри ещё раз внимательно:", "Ошибка есть, разберём по шагам:",
         "Проверим вместе, где съехало:", "Давай пересчитаем спокойно:"]
REMARKS = ["А можно пример посложнее?", "Теперь понятно, так и буду решать.", "Давай ещё.",
           "Ого, быстрее получилось, чем вчера.", "Так тоже можно решать? Запишу в тетрадь.",
           "Я так и думал, просто сомневался.", "Понял, спасибо. Что дальше?"]
EXPLAIN = ["Смотри: складываем десятки, потом единицы, это держит порядок.",
           "Обратная проверка умножением показывает, что решение сходится.",
           "Разбиваем число на удобные части – так меньше шансов на ошибку.",
           "Считаем по частям: сначала целое, потом остаток."]

# Semantic labels of the authored key blocks per session, in order. The frozen
# history files carry the resolved concrete turn keys ("key_block_keys"), the
# gold manifest references the semantic keys below.
KEY_BLOCK_LABELS = {
    "r2-mem-a": [["comics_interest", "aquarium"], ["dog_baitik", "times_error_help"],
                 ["exact_hours_minutes"], ["space_interest_change"], ["unfinished_rockets"],
                 ["dog_progress"]],
    "r2-mem-b": [["swim_days", "story_fox_start"], ["dinosaurs_interest", "pirates_avoidance"],
                 ["hzy_fractions"], ["story_fox_continue"], ["easy_tasks_today"], ["closing"]],
    "r2-mem-final": [["perimeter_area_intro"], ["brother_age"], ["lego_to_programming"],
                     ["brother_age_check", "perimeter_area_repeat"], ["unfinished_aquarium"],
                     ["exact_candies"]],
}


def _two_step_story(rng):
    obj = rng.choice(OBJECTS)
    total = rng.randint(20, 40)
    give = rng.randint(5, 9)
    lose = rng.randint(2, 4)
    name = rng.choice(NAMES)
    answer = total - give - lose
    question = (f"Задача в два шага: у {name} было {total} {obj['gen']}. Она отдала {give} "
                f"{obj['gen']} подруге и потеряла ещё {lose} по дороге домой. Сколько "
                f"{obj['name']} осталось у {name}? Реши и объясни, как считал.")
    solution = f"Сначала {total} – {give} = {total - give}, потом {total - give} – {lose} = {answer}."
    check = (f"Проверяем обратным действием: {answer} + {lose} + {give} = {total} – сходится. "
             f"Ответ {answer}, {obj['name']} осталось.")
    return question, solution, answer, check


def _two_operations(rng):
    a = rng.randint(6, 12)
    b = rng.randint(3, 9)
    c = rng.randint(5, 30)
    answer = a * b - c
    question = (f"Комбинированный пример: сначала вычисли {a} × {b}, потом из результата "
                f"вычти {c}. Запиши промежуточный результат и финальный ответ.")
    solution = f"{a} × {b} = {a * b}, затем {a * b} – {c} = {answer}."
    check = (f"Верно: промежуточный {a * b}, финальный {answer}. Если промежуточный "
             f"совпал, значит порядок действий соблюдён.")
    return question, solution, answer, check


def _money(rng):
    price = rng.randint(6, 15)
    count = rng.randint(3, 5)
    total = price * count + rng.randint(10, 40)
    answer = total - price * count
    question = (f"Задача про деньги: у ученика {total} рублей. Он купил {count} булочки "
                f"по {price} рублей за каждую. Сколько рублей осталось на проезд домой? "
                f"Объясни решение.")
    solution = f"{count} × {price} = {price * count} рублей за булочки, {total} – {price * count} = {answer}."
    check = (f"Правильно: на булочки ушло {price * count} рублей, осталось {answer}. "
             f"Можно проверить: {answer} + {price * count} = {total}.")
    return question, solution, answer, check


def _minutes(rng):
    hours = rng.randint(2, 4)
    minutes = rng.choice([10, 15, 20, 30])
    answer = hours * 60 + minutes
    question = (f"Задача про время: тренировка длилась {hours} часа {minutes} минут. "
                f"Сколько это всего минут? Переведи часы в минуты и сложи.")
    solution = f"{hours} часа – это {hours * 60} минут, плюс {minutes} минут = {answer} минут."
    check = (f"Да: {hours * 60} + {minutes} = {answer}. Запомни перевод: 1 час = 60 минут, "
             f"поэтому часы всегда умножаем на 60.")
    return question, solution, answer, check


def _perimeter_area(rng):
    side = rng.randint(4, 12)
    perimeter, area = side * 4, side * side
    question = (f"Геометрия: квадрат со стороной {side} сантиметров. Найди его периметр и "
                f"площадь, и напиши, чем они отличаются.")
    solution = (f"Периметр: ({side} + {side}) × 2 = {perimeter} сантиметров, это по контуру. "
                f"Площадь: {side} × {side} = {area} квадратных сантиметров, это внутри.")
    check = (f"Всё верно: периметр {perimeter} см по контуру, площадь {area} кв. см внутри. "
             f"Это разные величины, и единицы у них разные.")
    return question, solution, f"{perimeter} и {area}", check


def _filler_exchange(rng, session_index, index):
    """One generated tutoring exchange: question, worked solution, verification."""
    kind = index % 5
    if kind == 0:
        question, solution, answer, check = _two_step_story(rng)
    elif kind == 1:
        question, solution, answer, check = _two_operations(rng)
    elif kind == 2:
        question, solution, answer, check = _money(rng)
    elif kind == 3:
        question, solution, answer, check = _minutes(rng)
    else:
        question, solution, answer, check = _perimeter_area(rng)
    turns = [{"role": "assistant", "text": question}]
    if index % 6 == 2:
        turns.append({"role": "student", "text": rng.choice(REMARKS)})
        turns.append({"role": "assistant",
                      "text": "Хороший настрой. Сначала решим этот пример, потом продолжим."})
    turns.append({"role": "student", "text": solution})
    if index % 8 == 5:
        turns.append({"role": "assistant",
                      "text": f"{rng.choice(RETRY)} Посмотри на свои вычисления ещё раз: "
                              f"{rng.choice(EXPLAIN)}"})
        turns.append({"role": "student", "text": solution})
    turns.append({"role": "assistant", "text": check})
    if index % 4 == 1:
        turns.append({"role": "assistant", "text": rng.choice(PRAISE)})
    return turns


def _turns_to_blocks(turns, key_prefix, markers=None):
    """Convert role turns into loader turn records with stable keys.

    Every block starts with a student turn; an assistant turn without a
    preceding student turn gets a minimal consent stub so the chat-pair
    contract of the storage layer holds. ``markers`` maps turn positions to
    semantic labels; the block that carries the marked turn records its
    concrete key, so the frozen history files and the loader share one
    numbering.
    """
    blocks = []
    marker_keys = {}
    index = 1
    pending_user = None
    for position, turn in enumerate(turns):
        current_key = None
        if turn["role"] == "student":
            if pending_user is not None:
                blocks.append(pending_user)
                index += 1
                pending_user = None
            pending_user = {"key": f"{key_prefix}-t{index}", "text": turn["text"]}
            current_key = pending_user["key"]
        else:
            if pending_user is None:
                pending_user = {"key": f"{key_prefix}-t{index}",
                                "text": "Хорошо, давай продолжать."}
            pending_user["assistant"] = turn["text"]
            current_key = pending_user["key"]
            blocks.append(pending_user)
            index += 1
            pending_user = None
        for label, marker in (markers or {}).items():
            if marker == position:
                marker_keys[label] = current_key
    if pending_user is not None:
        blocks.append(pending_user)
    if markers:
        return blocks, marker_keys
    return blocks


def _build_session(rng, session_spec, session_index, history_id):
    """One session: filler exchanges plus authored key blocks at fixed spots."""
    key_prefix = f"{history_id}-s{session_index + 1}"
    turns = []
    authored = [dict(block) for block in session_spec["key_blocks"]]
    labels = session_spec["key_block_labels"]
    markers = {}
    filler_total = session_spec["filler_exchanges"]
    chunk = max(1, filler_total // (len(authored) + 1))
    filler_used = 0
    for block_index in range(len(authored) + 1):
        filler = chunk if block_index < len(authored) else filler_total - filler_used
        for exchange_index in range(filler):
            turns.extend(_filler_exchange(rng, session_index,
                                          session_index * 2 + block_index * 5 + exchange_index))
        filler_used += filler
        if block_index < len(authored):
            markers[labels[block_index]] = len(turns)
            turns.extend(authored[block_index]["turns"])
    blocks, marker_keys = _turns_to_blocks(turns, key_prefix, markers)
    resolved = {f"{history_id.split('r2-mem-')[1]}.{label}": key
                for label, key in marker_keys.items()}
    return {"session_key": key_prefix, "day_offset": session_spec["day_offset"],
            "start_hour": session_spec["start_hour"], "topic_id": session_spec["topic_id"],
            "key_block_keys": resolved, "turns": blocks}


def _history_specs():
    return [
        {
            "history_id": "r2-mem-a",
            "description": "Development history A – the oversized archive. Must exceed the "
                           "32768-token model window in total transcript size. Carries the "
                           "changed-interest arc (comics -> space), the 'во сколько раз' error "
                           "with the assistant's help, the unfinished rocket exercise, the dog "
                           "Байтик incidental fact, the exact 'часы и минуты' wording and the "
                           "аквариум fact that sits outside the seven-active-day view.",
            "exceeds_32768_token_window": True,
            "sessions": [
                {"day_offset": -47, "start_hour": 15, "topic_id": "math.g4.word_problems",
                 "filler_exchanges": 58,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Здравствуйте! Я сейчас больше всего люблю комиксы про "
                               "супергероев, читаю их каждый вечер и хочу задачу в тему."},
                      {"role": "assistant",
                       "text": "Отличная тема! Сегодня потренируем задачи, а про супергеройские "
                               "комиксы обязательно придумаем пример."}],
                     [{"role": "student",
                       "text": "Кстати, у меня дома есть аквариум с тремя гуппи, я их каждое "
                               "утро кормлю перед школой."},
                      {"role": "assistant",
                       "text": "Здорово! Гуппи неприхотливые. Запомнил: аквариум с тремя гуппи "
                               "кормится по утрам."}],
                 ]},
                {"day_offset": -40, "start_hour": 16, "topic_id": "math.g4.multiplication",
                 "filler_exchanges": 58,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "А ещё у меня появилась собака Байтик, австралийская овчарка. "
                               "Мы с ней гуляем после школы и она любит бегать за мячом."},
                      {"role": "assistant",
                       "text": "Байтик – звучит как маленький байт, отличное имя для собаки "
                               "будущего программиста!"}],
                     [{"role": "student",
                       "text": "Я часто путаюсь: «во сколько раз больше» и «на сколько "
                               "больше» – это же одно и то же? Всегда путаюсь в задачах."},
                      {"role": "assistant",
                       "text": "Нет, это разные сравнения. Возьмём числа 6 и 2: «во сколько "
                               "раз больше» – это деление, 6 : 2 = 3, то есть в 3 раза; а "
                               "«на сколько больше» – это вычитание, 6 – 2 = 4, то есть на 4. "
                               "Запомни правило: «во сколько» – умножаем или делим, "
                               "«на сколько» – складываем или вычитаем."}],
                 ]},
                {"day_offset": -33, "start_hour": 15, "topic_id": "math.g4.time_units",
                 "filler_exchanges": 58,
                 "key_blocks": [
                     [{"role": "assistant",
                       "text": "Итак, вопрос про время, слушай внимательно: сколько минут в "
                               "двух с половиной часах? Подумай и ответь полным решением."},
                      {"role": "student",
                       "text": "Два часа это 120 минут, половина это 30 минут, всего 150 минут."},
                      {"role": "assistant",
                       "text": "Верно, 150 минут! Ты разложил задачу на части и ничего не "
                               "потерял."}],
                 ]},
                {"day_offset": -26, "start_hour": 17, "topic_id": "programming.scratch.conditions",
                 "filler_exchanges": 58,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Знаешь, комиксы про супергероев мне теперь скучны, дочитал и "
                               "больше не тянет. Мне теперь нравится космос и ракеты, хочу "
                               "задачи про ракеты и планеты."},
                      {"role": "assistant",
                       "text": "Понял тебя: интерес сместился от супергеройских комиксов к "
                               "космосу и ракетам. Будем строить задачи вокруг космоса."}],
                 ]},
                {"day_offset": -12, "start_hour": 16, "topic_id": "math.g4.division",
                 "filler_exchanges": 56,
                 "key_blocks": [
                     [{"role": "student", "text": "Давай упражнение, я готов."},
                      {"role": "assistant",
                       "text": "Держи упражнение из трёх пунктов: 1) реши 84 : 12; 2) придумай "
                               "обратную задачу на умножение к этому же числу; 3) составь свою "
                               "задачу про ракеты с делением и запиши её решение полностью."},
                      {"role": "student", "text": "1) 7. 2) Обратная: 7 × 12 = 84, значит 84 : 7 = 12."},
                      {"role": "assistant",
                       "text": "Пункты 1 и 2 верные. Пункт 3 – своя задача про ракеты с "
                               "делением – осталось составить и решить."},
                      {"role": "student",
                       "text": "Идею придумал: ракета летит к Марсу и делит топливные баки "
                               "между отсеками. Но допишу завтра, уже поздно."},
                      {"role": "assistant",
                       "text": "Хорошо, продолжим с задачей про ракеты в следующий раз – "
                               "пункт 3 остаётся незавершённым, я его запомнил."}],
                 ]},
                {"day_offset": -5, "start_hour": 15, "topic_id": "math.g4.word_problems",
                 "filler_exchanges": 56,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Мы с Байтиком сегодня быстро прошли маршрут, он стал "
                               "послушнее и бегает за мячом без остановки."},
                      {"role": "assistant",
                       "text": "Отличный прогресс у Байтика! Тогда и мы продолжим без "
                               "отвлечений, тем более аквариум уже покормлен."}],
                 ]},
            ],
        },
        {
            "history_id": "r2-mem-b",
            "description": "Development history B – dinosaurs interest with a separate pirate "
                           "story avoidance, the one-off 'easy tasks today' request, the «хз» "
                           "answer tied to its question, the fox story continuation, the "
                           "swimming-days fact outside the seven-active-day view and the "
                           "genuinely missing favorite-color fact.",
            "exceeds_32768_token_window": False,
            "sessions": [
                {"day_offset": -45, "start_hour": 16, "topic_id": "math.g4.arithmetic",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Запиши себе расписание: я хожу на плавание по вторникам и "
                               "четвергам, поэтому уроки в эти дни короткие и после обеда."},
                      {"role": "assistant",
                       "text": "Записал: плавание по вторникам и четвергам, в эти дни работаем "
                               "короткими сессиями после обеда."}],
                     [{"role": "assistant",
                       "text": "И начнём сегодня маленькую историю про лисёнка Рыжика: он нашёл "
                               "в лесу старую карту с тремя загадками. Первая загадка: сколько "
                               "лапок у трёх лисят, если у каждого по четыре лапки?"},
                      {"role": "student", "text": "12, потому что 3 × 4 = 12."},
                      {"role": "assistant",
                       "text": "Верно! Рыжик записал ответ на карту и побежал дальше по "
                               "тропинке. Историю продолжим в следующий раз."}],
                 ]},
                {"day_offset": -38, "start_hour": 15, "topic_id": "math.g4.arithmetic",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Я обожаю истории про динозавров, особенно про тираннозавра – "
                               "хочу задачу про него."},
                      {"role": "assistant",
                       "text": "Принято: ты любишь истории про динозавров. Добавим "
                               "тираннозавра в наши примеры."}],
                     [{"role": "student",
                       "text": "И ещё одна просьба: не рассказывай мне страшные истории про "
                               "пиратов, я их боюсь. Только не про пиратов, ладно?"},
                      {"role": "assistant",
                       "text": "Хорошо, страшных историй про пиратов не будет – это только про "
                               "истории в занятиях, а твой интерес к динозаврам остаётся."}],
                 ]},
                {"day_offset": -31, "start_hour": 16, "topic_id": "math.g4.fractions",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "assistant",
                       "text": "Вопрос по дробям: какая дробь больше – 2/5 или 3/5? Ответь "
                               "сразу, как думаешь."},
                      {"role": "student", "text": "хз"},
                      {"role": "assistant",
                       "text": "Давай разберём: знаменатель одинаковый, значит сравниваем "
                               "числители: 3 больше 2, поэтому 3/5 больше. Правило: при "
                               "одинаковых знаменателях больше та дробь, у которой числитель "
                               "больше."}],
                 ]},
                {"day_offset": -20, "start_hour": 15, "topic_id": "math.g4.arithmetic",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "assistant",
                       "text": "Продолжаем историю про лисёнка Рыжика и его карту с тремя "
                               "загадками. Он дошёл до развилки."},
                      {"role": "student",
                       "text": "Он встретил ежа, который знал дорогу к третьей загадке, и "
                               "поделился с ним черникой!"},
                      {"role": "assistant",
                       "text": "Отличное продолжение: ёж-проводник ведёт Рыжика к третьей "
                               "загадке, а черника стала благодарностью. Сюжет сохранён."}],
                 ]},
                {"day_offset": -10, "start_hour": 16, "topic_id": "math.g4.arithmetic",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Можно сегодня только лёгкие задачки, у меня устали глаза после "
                               "плавания? Только на сегодня, не на всегда."},
                      {"role": "assistant",
                       "text": "Конечно, сегодня – только лёгкие задачки. Это разовая "
                               "договорённость именно на сегодня, не постоянное правило."}],
                 ]},
                {"day_offset": -3, "start_hour": 15, "topic_id": "math.g4.arithmetic",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "assistant",
                       "text": "Сегодня без новой истории, закрепим счёт и вернёмся к Рыжику в "
                               "следующий раз."},
                      {"role": "student", "text": "Хорошо, давай считать."}],
                 ]},
            ],
        },
        {
            "history_id": "r2-mem-final",
            "description": "FINAL CHECK history (held out) – LEGO->programming changed interest, "
                           "the corrected brother-age fact, the perimeter-vs-area error with "
                           "help, the unfinished aquarium-fractions task, the exact candy "
                           "division wording and the genuinely missing future-job fact.",
            "exceeds_32768_token_window": False,
            "sessions": [
                {"day_offset": -44, "start_hour": 15, "topic_id": "math.g4.geometry",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "assistant",
                       "text": "Напомни на всякий случай: периметр – это сумма длин всех сторон "
                               "по контуру, сколько идём вокруг фигуры; площадь – сколько "
                               "места внутри. Будем различать их на задачах."},
                      {"role": "student", "text": "Ага, по контуру – периметр, внутри – площадь. Запомню."}],
                 ]},
                {"day_offset": -36, "start_hour": 16, "topic_id": "math.g4.word_problems",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Мой брат старше меня на пять лет, мы вместе решаем примеры по "
                               "вечерам."},
                      {"role": "assistant",
                       "text": "Хорошо, записал: занятия по вечерам вместе с братом."}],
                 ]},
                {"day_offset": -28, "start_hour": 15, "topic_id": "programming.scratch.basics",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "student",
                       "text": "Раньше я любил LEGO и собирал из него всё подряд, а теперь мне "
                               "интереснее программирование и Scratch – хочу задачи в эту "
                               "сторону."},
                      {"role": "assistant",
                       "text": "Понял: интерес сместился от LEGO к программированию и Scratch. "
                               "Будем двигать задачи в эту сторону."}],
                 ]},
                {"day_offset": -18, "start_hour": 16, "topic_id": "math.g4.geometry",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "student", "text": "Хорошо, давай проверять."},
                      {"role": "assistant",
                       "text": "Сверим одну запись из твоей карточки: у меня записано «брат "
                               "старше ученика на пять лет». Проверим это?"},
                      {"role": "student", "text": "Точно пять? Кажется, уже семь – у него в "
                                                  "этом году был день рождения."},
                      {"role": "assistant",
                       "text": "Спасибо за проверку, зафиксируем исправление: брат старше на "
                               "семь лет."}],
                     [{"role": "assistant",
                       "text": "Теперь разберём периметр против площади на примере квадрата "
                               "3×3: периметр 12 сантиметров, площадь 9 квадратных "
                               "сантиметров. Числа разные, потому что это разные величины."},
                      {"role": "student", "text": "Понял, периметр по контуру, площадь внутри."}],
                 ]},
                {"day_offset": -9, "start_hour": 15, "topic_id": "math.g4.fractions",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "student", "text": "Готов, давай упражнение."},
                      {"role": "assistant",
                       "text": "Упражнение из двух пунктов: 1) реши: из аквариума, заполненного "
                               "на 3/4, вылили 1/4 – какая часть осталась? 2) составь свою "
                               "задачу про аквариум и дроби с полным решением."},
                      {"role": "student", "text": "1) 2/4, то есть половина аквариума."},
                      {"role": "assistant",
                       "text": "Пункт 1 верный. Пункт 2 – своя задача про аквариум и дроби – "
                               "остаётся незавершённым, вернёмся к нему в следующий раз."},
                      {"role": "student", "text": "Ладно, в следующий раз додумаю."}],
                 ]},
                {"day_offset": -2, "start_hour": 16, "topic_id": "math.g4.division",
                 "filler_exchanges": 14,
                 "key_blocks": [
                     [{"role": "assistant",
                       "text": "Вопрос ровно в такой формулировке: раздели 96 конфет между "
                               "четырьмя друзьями поровну. Сколько получит каждый?"},
                      {"role": "student", "text": "24 конфеты, потому что 96 : 4 = 24."},
                      {"role": "assistant",
                       "text": "Верно, по 24. Проверка умножением: 24 × 4 = 96 – сходится."}],
                 ]},
            ],
        },
    ]


CORRECTIONS = [
    {"history_id": "r2-mem-final",
     "corrects_key": "final.brother_age",
     "key": "final.brother_age_correction",
     "day_offset": -18,
     "author_role": "teacher",
     "text": "Правка преподавателя: брат старше ученика на семь лет, а не на пять."},
]


def build_histories(seed=20261117):
    rng = random.Random(seed)
    histories = []
    for spec in _history_specs():
        history = {"history_id": spec["history_id"], "description": spec["description"],
                   "exceeds_32768_token_window": spec["exceeds_32768_token_window"],
                   "sessions": []}
        for index, session_spec in enumerate(spec["sessions"]):
            block_labels = KEY_BLOCK_LABELS[spec["history_id"]][index]
            session_spec["key_block_labels"] = block_labels
            session_spec["key_blocks"] = [{"label": label, "turns": turns}
                                          for label, turns in zip(block_labels,
                                                                  session_spec["key_blocks"])]
            history["sessions"].append(_build_session(rng, session_spec, index,
                                                      spec["history_id"]))
        histories.append(history)
    corrections = [dict(item) for item in CORRECTIONS]
    return histories, corrections


def main():
    histories, corrections = build_histories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for history in histories:
        path = OUT_DIR / f"{history['history_id']}.json"
        path.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
        total = sum(len(turn["text"]) + len(turn.get("assistant") or "")
                    for session in history["sessions"] for turn in session["turns"])
        print(f"{path.name}: sessions={len(history['sessions'])} "
              f"turns={sum(len(s['turns']) for s in history['sessions'])} chars={total}")
    (OUT_DIR / "corrections.json").write_text(
        json.dumps(corrections, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
