# DSL Query Generator — ECQL (ДЗ «LLM-judge»)

Генератор запросов на внутреннем языке **ECQL** компании «E-Corp»: дообученная
малая LLM переводит русские запросы менеджеров в корпоративный DSL.
Датасет синтезирует модель-учитель **YandexGPT Pro**, обучение — **LoRA (r=4)**
на **CPU** (`Qwen2.5-1.5B-Instruct`, `trl.SFTTrainer`), оценка — метрики
синтаксиса/логики + **LLM-as-a-judge** (YandexGPT Pro).

## Стек

| Компонент      | Решение                                                                 |
|----------------|-------------------------------------------------------------------------|
| Учитель/судья  | YandexGPT Pro (`yandexgpt`) через Yandex Cloud Foundation Models API   |
| Модель (CPU)   | `Qwen/Qwen2.5-1.5B-Instruct`, fp32, LoRA r=4 alpha=8 (q/k/v/o)          |
| Модель (GPU)   | `Qwen/Qwen3-4B-Instruct`, QLoRA 4-bit (см. `notebooks/train_colab.py`) |
| Обучение       | `trl.SFTTrainer` + `peft.LoraConfig`, маскирование промпта, лог Loss    |
| Пакеты         | `transformers==4.49.0`, `trl==0.16.0`, `peft==0.14.0`, `accelerate==1.3.0` |
| Управление     | Python 3.12+, `uv` (CPU-сборка torch)                                   |

## Структура проекта

```
config/config.yaml        Общий конфиг (dataset, teacher, local_train, gpu_train, judge, eval)
src/ecql/
  config.py               Пути, .env, загрузка config.yaml
  ecql_spec.py            Грамматика ECQL: парсер + валидатор (FETCH/[]/@/&&/||/AS)
  yandex_client.py        YandexGPT Pro: учитель + судья (LLM-as-a-judge)
  dataset.py              Генерация/валидация/split датасета, stats
  train.py                LoRA-обучение (SFTTrainer), маскирование промпта, лог Loss в CSV
  eval.py                 Метрики на тесте + LLM-as-a-judge + генерация ответов
scripts/                  generate/validate/train/evaluate/run_notebook
notebooks/
  report_local.py         Отчёт (CPU): данные -> loss-кривая -> метрики -> judge
  report_local_run.ipynb  Исполненный отчёт (результаты прогона)
  train_colab.py          Ноутбук для Google Colab GPU (Qwen3-4B, QLoRA) — запускать вручную
data/                     ecql_dataset.jsonl (201), ecql_train.jsonl (160), ecql_test.jsonl (41)
results/                  loss_log.csv, loss_curve.png, eval_local.json, eval_table.md,
                          train_local.json, dataset_stats.json, lora_adapter_local/
results_collab/           Артефакты Colab-прогона: ecql_results/ (train_colab.json,
                          eval_colab.json, loss_log.csv, loss_curve.png), ecql_adapter/
tests/test_ecql.py        Pytest: грамматика, сплит, учитель/судья
```

## Быстрый старт

```bash
uv sync --group dev

# 1. Датасет (YandexGPT Pro как учитель; ключи в .env — копия из ../06_l16_advanced_rag)
uv run python scripts/generate_dataset.py --force --target 200

# 2. Валидация
uv run python scripts/validate_dataset.py

# 3. Обучение LoRA на CPU (Qwen2.5-1.5B, r=4, alpha=8, 5 эпох ~ 37 мин)
uv run python scripts/train_local.py --epochs 5

# 4. Оценка: метрики + LLM-as-a-judge
uv run python scripts/evaluate.py

# 5. Отчёт-ноутбук
uv run python scripts/run_notebook.py --report notebooks/report_local.py \
    --output notebooks/report_local_run.ipynb

# 6. Тесты/линтер
uv run pytest -q
uv run ruff check src/ scripts/ tests/ notebooks/
```

## Спецификация ECQL

- Запрос начинается с `FETCH`; сущности — `[EMPLOYEES] [PROJECTS] [INVENTORY] [DEALS]`
- Поля — `@salary`, `@status`, `@city`, ... (у каждой сущности свой набор, см. `ecql_spec.py`)
- Операторы: `IS`, `NOT`, `ABOVE`, `BELOW`; связки `&&` (И) и `||` (ИЛИ), скобки
- Формат вывода (необязательно): `AS JSON | AS TABLE | AS LIST`
- Пример:
  `FETCH [EMPLOYEES] WHERE @city IS 'Moscow' && @salary ABOVE 150000`

## Датасет

201 пара «русский запрос -> ECQL» сгенерирована **YandexGPT Pro** (учитель):
промпт содержит спецификацию ECQL + требования к разнообразию, ответ — JSON Lines.
Каждый пример проходит строгий парсер (`ecql_spec.parse_ecql`), дубликаты
удаляются. Split 80/20: **160 train / 41 test**.

| Показатель | Значение |
|------------|----------|
| Всего пар | 201 |
| Сущности | EMPLOYEES 56 · PROJECTS 56 · DEALS 45 · INVENTORY 44 |
| Операторы | IS 229 · ABOVE 92 · BELOW 30 · NOT 10 |
| Многоусловные запросы | 128 |
| Форматы вывода | JSON 4 · TABLE 5 · LIST 2 |

## Обучение (CPU)

Параметры — в `config/config.yaml` (`local_train`), не дублируются в коде.

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| Модель | Qwen2.5-1.5B-Instruct | малая модель = «вызов»: приёмлемое качество на слабом железе |
| LoRA r | **4** | минимальный ранг для малого датасета (201 пар) и DSL с фиксированным синтаксисом; r=4 достаточно, т.к. поверхность задачи узкая |
| LoRA alpha | **8** | классическое правило `alpha = 2*r` — стабильнее на CPU, чем alpha=16/32 |
| target_modules | q_proj, k_proj, v_proj, o_proj | стандартный набор внимания |
| Батч | 1 (накопление 4 → эффективный 4) | CPU-режим: малый микро-батч, большая память не нужна |
| lr | 2e-4, cosine, warmup 5% | рекомендованные для LoRA |
| Эпохи | 5 | loss снижался и после 2-й эпохи; 5 дали максимум точности |
| Оптимизатор | AdamW | CPU, fp32 |

**Замер времени (CPU, 16 потоков Ryzen 7 7840HS):** 5 эпох = **2211 c (~37 мин)**,
200 шагов оптимизатора. Loss: **3.19 → 0.155** (`results/loss_log.csv`,
`results/loss_curve.png`). Адаптер: `results/lora_adapter_local/` (4.4 МБ при r=4).

## Результаты оценки (41 пример теста)

| Метрика | Значение |
|---------|----------|
| Синтаксическая корректность (парсер ECQL) | **85.4%** |
| Точное совпадение с эталоном | **63.4%** |
| Совпадение сущности | 85.4% |
| Совпадение полей | 82.9% |
| Совпадение операторов (IS/NOT/ABOVE/BELOW) | 80.5% |
| Совпадение логики (&& / \|\|) | 78.0% |
| Совпадение формата (AS ...) | 95.1% |
| SQL-галлюцинации | **0%** |
| **LLM-as-a-judge** (YandexGPT Pro, средний балл 0-5) | **4.41** (80.5% ответов ≥ 4) |

Таблица сравнения 5 примеров — `results/eval_table.md`, полный разбор —
`results/eval_local.json`, исполненный отчёт — `notebooks/report_local_run.ipynb`.

### Сравнение вход — эталон — ответ (примеры)

| Вход (рус.) | Эталон | Ответ модели |
|---|---|---|
| Назови все товары на складе, количество которых меньше 10 | `FETCH [INVENTORY] WHERE @quantity BELOW 10` | `FETCH [INVENTORY] WHERE @quantity BELOW 10` ✓ |
| Какие товары в категории «Офисная техника» и стоят меньше 30 тысяч? | `... @category IS 'Office equipment' && @price BELOW 30000` | `... @category IS 'Office Equipment' && @price BELOW 30000` (регистр значения) |
| Покажи все проекты с приоритетом High, бюджетом > 300 тыс и командой > 20 | `... @budget ABOVE 300000 && @team_size ABOVE 20` | `... @budget ABOVE 300000 && @team ABOVE 20` (поле `@team`) |

### Анализ ошибок (примеры из отчёта)

1. **Перевод значений**: `'Офисная техника'` → `'Office Equipment'` вместо
   `'Office equipment'` — значения в эталоне не единственны (учитель смешивал
   русские и английские варианты), модель выбрала наиболее частотную форму.
2. **Синоним/поле**: `@team` вместо `@team_size`, `'On Shelf'` вместо `'In stock'` —
   редкие поля и значения в обучающем датасете представлены малым числом примеров,
   модель обобщает по частотности.
3. **NOT с отрицанием значения**: `NOT (@item AS 'Printer')` вместо `@name NOT 'Printer'` —
   сложная конструкция встречалась редко (NOT = 10 примеров из 201).

**Вывод:** 5 эпох LoRA r=4 достаточно, чтобы модель «выучила» новый язык запросов:
85% синтаксиса корректны, 0% SQL-галлюцинаций, судья оценивает в среднем на 4.4/5.
Остаточные ошибки — частотные (редкие поля/значения/NOT-конструкции), они уйдут
при расширении датасета.

## Результаты ноутбуков: локальный CPU vs Colab GPU

Оба ноутбука отработаны на одном датасете (**160 train / 41 test**) и одной
тестовой выборке; артефакты — в `results/` (ноутбук `report_local.py`) и
`results_collab/ecql_results/` (ноутбук `train_colab.py`).

### Производительность

| Ноутбук | Железо / режим | Модель | LoRA | Время обучения | Инференс (среднее) |
|---------|----------------|--------|------|----------------|--------------------|
| `report_local.py` | CPU fp32 (Ryzen 7 7840HS, 16 потоков) | Qwen2.5-1.5B-Instruct | r=4, α=8, 5 эпох (200 шагов) | **2211 c (~37 мин)** | **11.3 c/пример** |
| `train_colab.py` | Tesla T4, QLoRA 4-bit NF4 | Qwen3-4B | r=4, α=16, 3 эпохи (60 шагов) | **161 c (~2.7 мин)** | **3.1 c/пример** |

Источники: `results/train_local.json`, `results_collab/ecql_results/train_colab.json`.
На T4 обучение идёт **~14× быстрее**, инференс — **~3.7× быстрее**, причём
GPU-модель вдвое крупнее (4B против 1.5B). Финальный loss: 3.19 → **0.155** (CPU,
`results/loss_log.csv`) против среднего training loss **0.63** (GPU,
`results_collab/ecql_results/loss_log.csv`). Loss-кривые: `results/loss_curve.png`
и `results_collab/ecql_results/loss_curve.png`.

### Точность (41 пример теста, судья — YandexGPT Pro)

| Метрика | CPU: Qwen2.5-1.5B (`eval_local.json`) | GPU: Qwen3-4B QLoRA (`eval_colab.json`) |
|---------|---------------------------------------|------------------------------------------|
| Синтаксическая корректность (парсер ECQL) | 85.4% | **95.1%** |
| Точное совпадение с эталоном | 63.4% | **75.6%** |
| Совпадение сущности | 85.4% | **95.1%** |
| Совпадение полей | 82.9% | **87.8%** |
| Совпадение операторов | 80.5% | **90.2%** |
| Совпадение логики (&& / \|\|) | 78.0% | **87.8%** |
| Совпадение формата (AS ...) | 95.1% | 95.1% |
| SQL-галлюцинации | **0%** | **0%** |
| **LLM-as-a-judge**, средний балл 0–5 | 4.41 | **4.61** |
| Судья: доля ответов ≥ 4 | 80.5% | **90.2%** |

Источники: `results/eval_local.json`, `results_collab/ecql_results/eval_colab.json`.

**Вывод:** более крупный чекпоинт + QLoRA на GPU улучшают все метрики точности
(+9.8 п.п. синтаксис, +12.2 п.п. exact match, +0.2 балла судьи, pass rate
80.5% → 90.2%) при обучении в ~14 раз быстрее. При этом локальный CPU-прогон —
рабочий минимум: даже модель 1.5B выучивает DSL без единой SQL-галлюцинации,
а остаточные ошибки у обеих моделей одинаковы по типу (перевод значений,
редкие поля, NOT-конструкции).

## Google Colab GPU: выбор модели

Из кандидатов **Phi-4 (14B)**, **Mistral 3 / Small 3 (24B)**, **Qwen3-4B-Instruct (4B)**
выбран `Qwen/Qwen3-4B-Instruct`:

1. **Помещается в T4 16 ГБ** даже с QLoRA 4-bit (~3 ГБ весов); Phi-4 и Mistral 3
   в 4-bit занимают ~8 и ~13 ГБ и не влезают вместе с градиентами;
2. **Сильный русский трейнинг** — вход задачи русский; Phi-4 англоцентрична,
   у Mistral 3 русский слабее;
3. **Преемственность** с локальным Qwen2.5 (одинаковые chat-шаблоны и LoRA-таргеты).

## Как запустить в Google Colab (пошагово)

Ноутбук `notebooks/train_colab.py` — процент-скрипт; он же и есть Colab-ноутбук.
На GPU он запускается вручную (мы не выполняем его локально — см. ниже), по шагам:

1. **Откройте ноутбук в Colab.** Самый простой способ — скопировать содержимое
   `notebooks/train_colab.py` в новую ячейку Colab или сгенерировать `.ipynb` локально:
   ```bash
   uv run python -c "import jupytext; jupytext.write(jupytext.read('notebooks/train_colab.py', fmt='py:percent'), 'notebooks/train_colab.ipynb', fmt='ipynb')"
   ```
   и загрузить файл в Colab (`File → Upload notebook`) либо открыть с GitHub/Disk.

2. **Выберите рантайм с GPU.** В Colab: `Runtime → Change runtime type →
   Hardware accelerator = T4 GPU` (бесплатно) или A100/L4 (платно). Проверка:
   ячейка «0. Установка зависимостей» напечатает `torch cuda: True` и имя GPU.

3. **Загрузите данные.** В панели Colab перетащите `data/ecql_train.jsonl` и
   `data/ecql_test.jsonl` из этого репозитория в каталог `/content/data/`
   (создайте его через панель файлов или ячейку `!mkdir -p /content/data`).
   Ячейка «2. Данные» проверит их наличие и остановится с подсказкой, если файлов нет.

4. **Запустите ячейки сверху вниз.** Ноутбук сам поставит зависимости
   (`transformers==4.49.0`, `trl==0.16.0`, `peft==0.14.0`, `bitsandbytes`),
   загрузит `Qwen3-4B-Instruct` в 4-битном QLoRA и обучит LoRA r=4, alpha=16
   (3 эпохи, эффективный батч 8, lr=2e-4).

5. **Замеры времени.** Ячейка 5 выведет длительность обучения, ячейка 7 —
   время инференса на тестовой выборке. Фактический замер на T4: обучение
   **161 c (~2.7 мин)** против 2211 c (~37 мин) на CPU — ускорение ~14×;
   инференс 3.1 c/пример против 11.3 c (~3.7×). Детали — в разделе
   «Результаты ноутбуков».

6. **Сохраните адаптер.** Ячейка 6 пишет веса в `/content/ecql_adapter/`.
   Для сдачи загрузите их на HuggingFace Hub (раскомментируйте
   `trainer.push_to_hub(...)` с вашим токеном) или на Google Drive
   (`files.download` / подключение Drive).

7. **Оценка.** Ячейки 7-9 считают метрики (синтаксис, exact match, SQL-галлюцинации)
   прямо в Colab и печатают таблицу «вход — эталон — ответ». Полный отчёт с
   LLM-as-a-judge можно собрать локально: `uv run python scripts/evaluate.py`
   (требует только YandexGPT-ключи из `.env`).

Конфиг ноутбука совпадает с секцией `gpu_train` в `config/config.yaml`.

## Сдача

- **Код/репозиторий** — этот каталог (README, `scripts/`, `src/ecql/`, `notebooks/`)
- **Исполненный отчёт локальной загрузки** — `notebooks/report_local_run.ipynb` (+ `notebooks/report_local.py`)
- **Файл метрик** — `results/eval_local.json`, `results/eval_table.md`,
  `results/loss_curve.png`, `results/loss_log.csv`, а также артефакты Colab-прогона
  в `results_collab/ecql_results/` (`train_colab.json`, `eval_colab.json`,
  `loss_curve.png`)
- **Веса colab адаптера** — `results_colab/lora_adapter_local/` (https://huggingface.co/4rtgcom/checkpoints)
- **Colab GPU-ноутбук** — `notebooks/train_colab.py`

## Критерии

- [x] Синтетический датасет в правильном формате (JSONL, 201 пара, валидный ECQL)
- [x] Код Fine-tuning запускается, Loss снижается (3.19 → 0.155, график в `results/`)
- [x] Примеры успешной генерации DSL дообученной моделью (85% синтаксис, 63% exact match)
