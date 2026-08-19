# Сравнение вход — эталон — ответ модели (тестовая выборка)


| № | Вход (рус.) | Эталонный ECQL | Ответ модели | Синт. |
|---|-------------|----------------|--------------|-------|
| 1 | Назови все товары на складе, количество которых меньше 10 | FETCH [INVENTORY] WHERE @quantity BELOW 10 | FETCH [INVENTORY] WHERE @quantity BELOW 10 | ✓ |
| 2 | Какие товары находятся в категории 'Офисная техника' и стоят меньше 30 тысяч? | FETCH [INVENTORY] WHERE @category IS 'Office equipment' && @price BELOW 30000 | FETCH [INVENTORY] WHERE @category IS 'Office Equipment' && @price BELOW 30000 | ✓ |
| 3 | Какие клиенты заключили сделки на сумму выше 10 000 долларов? | FETCH [DEALS] WHERE @amount ABOVE 10000 && @currency IS 'USD' | FETCH [DEALS] WHERE @amount ABOVE 10000 && @currency IS 'USD' | ✓ |
| 4 | Найди всех сотрудников в Москве | FETCH [EMPLOYEES] WHERE @city IS 'Moscow' | FETCH [EMPLOYEES] WHERE @city IS 'Moscow' | ✓ |
| 5 | Какие сотрудники работают в отделе маркетинга и имеют зарплату больше 200 000 RUB | FETCH [EMPLOYEES] WHERE @department IS 'Маркетинг' && @salary ABOVE 200000 | FETCH [EMPLOYEES] WHERE @department IS 'Marketing' && @salary ABOVE 200000 | ✓ |


# Метрики (41 пример теста)

| Метрика | Значение |
|----------|----------|
| syntax_accuracy | 0.8537 |
| exact_match | 0.6341 |
| entity_accuracy | 0.8537 |
| fields_accuracy | 0.8293 |
| ops_accuracy | 0.8049 |
| logic_accuracy | 0.7805 |
| format_accuracy | 0.9512 |
| sql_hallucination_rate | 0.0 |
| judge_avg_score | 4.41 |
| judge_pass_rate | 0.8049 |
| avg_inference_sec | 11.27 |


# LLM-as-a-judge (YandexGPT Pro)

| Вход | Эталон | Ответ | Score | Вердикт | Комментарий |
|------|--------|-------|-------|---------|-------------|
| Назови все товары на складе, количество  |  | FETCH [INVENTORY] WHERE @quantity BELOW 10 | 5 | correct | Ответ полностью соответствует эталонному ECQL |
| Какие товары находятся в категории 'Офис |  | FETCH [INVENTORY] WHERE @category IS 'Office  | 4 | partial | Ответ модели почти полностью совпадает с эталонным, за исключением регистра симв |
| Какие клиенты заключили сделки на сумму  |  | FETCH [DEALS] WHERE @amount ABOVE 10000 && @c | 5 | correct | Ответ полностью соответствует эталонному ECQL-запросу. |
| Найди всех сотрудников в Москве |  | FETCH [EMPLOYEES] WHERE @city IS 'Moscow' | 5 | correct | Ответ полностью соответствует эталонному ECQL |
| Какие сотрудники работают в отделе марке |  | FETCH [EMPLOYEES] WHERE @department IS 'Marke | 4 | partial | В запросе допущена ошибка в названии отдела: 'Marketing' вместо 'Маркетинг'. |
| Какие товары находятся в статусе 'На скл |  | FETCH [INVENTORY] WHERE @status IS 'On Shelf' | 3 | partial | Неверно указан статус товара ('On Shelf' вместо 'In stock'). |
| Какие проекты имеют бюджет ниже 50 000 е |  | FETCH [PROJECTS] WHERE @budget BELOW 50000 && | 5 | correct | Ответ полностью соответствует эталонному ECQL-запросу. |
| Найти все проекты с бюджетом выше 100 00 |  | FETCH [PROJECTS] WHERE @budget ABOVE 100000 | 5 | correct | Ответ полностью соответствует эталонному ECQL |
