# Конвертация Jupyter Notebook <-> Python-скрипт

Проект использует [jupytext](https://jupytext.readthedocs.io/) — он входит в requirements.txt.

## Notebook -> скрипт (.ipynb -> .py)

```bash
jupytext --to py:percent notebook/rag_lab.ipynb
```

Создаст файл `notebook/rag_lab.py` в формате "percent" (ячейки разделены `# %%`),
который можно:
- открыть в PyCharm/VS Code как обычный notebook (с ячейками),
- запустить целиком без Jupyter: `python notebook/rag_lab.py`.

## Скрипт -> notebook (.py -> .ipynb)

```bash
jupytext --to notebook notebook/rag_lab.py
```

Восстановит `.ipynb` из `.py` (с ячейками, но без сохранённых outputs).

## Синхронизация (держать оба файла актуальными одновременно)

Один раз свяжите пару файлов:

```bash
jupytext --set-formats ipynb,py:percent notebook/rag_lab.ipynb
```

После этого при каждом сохранении в Jupyter автоматически обновляется .py,
а при правке .py можно подтянуть изменения обратно командой:

```bash
jupytext --sync notebook/rag_lab.ipynb
```

## Альтернатива без jupytext: nbconvert (только в одну сторону, ipynb -> py)

```bash
jupyter nbconvert --to script notebook/rag_lab.ipynb --output rag_lab_plain
```

Это создаёт обычный `.py`-файл (без ячеек `# %%`), пригодный для запуска
`python notebook/rag_lab_plain.py`, но обратно в `.ipynb` его так же удобно
превратить не получится — используйте jupytext для двусторонней конвертации.
