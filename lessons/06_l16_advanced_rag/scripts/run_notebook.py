"""Запуск notebooks/report.py как Jupyter-ноутбука и сохранение результатов.

Конвертирует процент-скрипт в .ipynb, исполняет каждую ячейку и сохраняет
ноутбук с выводами по указанному пути.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
import jupytext
from nbclient import NotebookClient


def run(report_py: Path, out_ipynb: Path, kernel: str) -> None:
    nb = jupytext.read(report_py, fmt="py:percent")
    nb.metadata["kernelspec"] = {
        "display_name": kernel,
        "language": "python",
        "name": kernel,
    }
    nb.metadata["language_info"] = {"name": "python"}

    client = NotebookClient(nb, kernel_name=kernel, timeout=1200)
    client.execute()

    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_ipynb)
    print(f"Сохранено: {out_ipynb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="osinka-rag")
    parser.add_argument("--report", default="notebooks/report.py")
    parser.add_argument("--output", default="notebooks/report_run_local_20260817.ipynb")
    args = parser.parse_args()

    run(Path(args.report), Path(args.output), args.kernel)
