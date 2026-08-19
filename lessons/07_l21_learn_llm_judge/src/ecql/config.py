"""Конфигурация проекта ECQL Query Generator.

Загружает `.env` из корня проекта и константы путей/моделей.
Общие настройки пайплайнов лежат в `config/config.yaml` (см. `load_yaml`).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_YAML = CONFIG_DIR / "config.yaml"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
ADAPTER_DIR = RESULTS_DIR / "lora_adapter_local"
LOSS_LOG = RESULTS_DIR / "loss_log.csv"
EVAL_RESULTS = RESULTS_DIR / "eval_local.json"

DATASET_FULL = DATA_DIR / "ecql_dataset.jsonl"
DATASET_TRAIN = DATA_DIR / "ecql_train.jsonl"
DATASET_TEST = DATA_DIR / "ecql_test.jsonl"

# Yandex Cloud Foundation Models (модель-учитель и судья)
YC_API_KEY = os.getenv("YC_API_KEY", "")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID", "")
YC_URL = os.getenv(
    "YC_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)
YC_MODEL = os.getenv("YC_MODEL", "yandexgpt")

LOCAL_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_yaml() -> dict:
    """Загружает `config/config.yaml` как dict."""
    with CONFIG_YAML.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_json(path: Path, data) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_json(path: Path):
    import json

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
