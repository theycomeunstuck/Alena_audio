# Ограниченные механические наблюдения v2

Это наблюдения над двумя готовыми пакетами. Общего pass, оценки учебной правильности и выбора предложенного примера здесь нет. Человеческий просмотр не выполнен.

Символические деления ищутся только как целые числа с `:`, `÷` или `/`; словесное деление и оформление операндов Markdown могут остаться нераспознанными. Все совпадения перечислены, включая цитаты старых примеров. Неверное числовое свойство относится только к найденной паре, а не автоматически к ответу.

В кодовых блоках допускаются только прямые вызовы `print` с константными аргументами и литералами `sep`/`end`. Ничего из ответа не исполняется через eval/exec. Отсутствие блока – not_found/not_observed; проза, inline-код и неподдерживаемый код не считаются ошибкой. Несовпадение вывода блока с фиксированной целью не определяет роль этого блока в объяснении.

Для T09/T10 отмечаются лишь буквальные появления числа или полной исправленной строки. Контекст отрицания и смысл подсказки здесь не оцениваются. not_observed никогда не означает правильный ответ.

| Пакет | Случай | Наблюдение |
| --- | --- | --- |
| baseline01 | T01 | not_observed; outside_narrow_mechanical_scope |
| baseline01 | T02 | not_observed; outside_narrow_mechanical_scope |
| baseline01 | T03 | not_observed;  |
| baseline01 | T04 | not_observed;  |
| baseline01 | T05 | not_observed; not_found |
| baseline01 | T06 | not_observed; not_found |
| baseline01 | T07 | not_observed; outside_narrow_mechanical_scope |
| baseline01 | T08 | not_observed; outside_narrow_mechanical_scope |
| baseline01 | T09 | not_observed; буквальных совпадений: 0 |
| baseline01 | T10 | not_observed; буквальных совпадений: 0 |
| baseline01 | T11 | not_observed; outside_narrow_mechanical_scope |
| baseline01 | T12 | not_observed; outside_narrow_mechanical_scope |
| candidate01 | T01 | not_observed; outside_narrow_mechanical_scope |
| candidate01 | T02 | not_observed; outside_narrow_mechanical_scope |
| candidate01 | T03 | observed; 402 : 2 → satisfied |
| candidate01 | T04 | observed; 4080 : 12 → wrong |
| candidate01 | T05 | not_observed; not_found |
| candidate01 | T06 | not_observed; not_found |
| candidate01 | T07 | not_observed; outside_narrow_mechanical_scope |
| candidate01 | T08 | not_observed; outside_narrow_mechanical_scope |
| candidate01 | T09 | not_observed; буквальных совпадений: 0 |
| candidate01 | T10 | not_observed; буквальных совпадений: 0 |
| candidate01 | T11 | not_observed; outside_narrow_mechanical_scope |
| candidate01 | T12 | not_observed; outside_narrow_mechanical_scope |

Полные совпадения, позиции, ограничения, SHA256 входов и исходника приведены в observations.json. Проверка trusted gold подтверждает внутреннюю согласованность набора и не заменяет его содержательную валидацию человеком.
