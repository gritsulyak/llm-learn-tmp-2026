"""Запуск notebooks/report_local.py как Jupyter-ноутбука и сохранение с выводами.

Конвертирует процент-скрипт в .ipynb, исполняет ячейки и сохраняет ноутбук.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient


def run(report_py: Path, out_ipynb: Path, kernel: str, timeout: int) -> None:
    nb = jupytext.read(report_py, fmt="py:percent")
    nb.metadata["kernelspec"] = {
        "display_name": kernel,
        "language": "python",
        "name": kernel,
    }
    nb.metadata["language_info"] = {"name": "python"}

    client = NotebookClient(nb, kernel_name=kernel, timeout=timeout)
    client.execute()

    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_ipynb)
    print(f"Сохранено: {out_ipynb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="ecql")
    parser.add_argument("--report", default="notebooks/report_local.py")
    parser.add_argument("--output", default="notebooks/report_local_run.ipynb")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    run(Path(args.report), Path(args.output), args.kernel, args.timeout)
