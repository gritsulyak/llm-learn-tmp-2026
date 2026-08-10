# Архитектура: AI SMM-отдел (LangGraph + Ollama, CPU-only)

## Модели (слабый ноутбук, AMD GPU, Ollama на CPU)

Без CUDA — берём только маленькие модели (3B и ниже), иначе будет неприемлемо медленно.

| Роль | Модель | Причина |
|------|--------|---------|
| Strategist | `qwen2.5:3b` | лучше держит структуру (план/JSON) |
| Copywriter | `vikhr-llama-3.2-3b` | сильнее в креативном русском тексте |
| Editor | `qwen2.5:3b` | стабильнее следует инструкциям критики |
| Publisher | `qwen2.5:1.5b` | задача формальная, хватает минимума |

Если совсем тормозит — одна модель `qwen2.5:3b` на все роли.

## Структура папок

```
05_l13_langfuse_ai_smm/
├── doc/
│   └── TASK_RU.md
│   └── ARCHITECTURE.md
├── notebook/
│   └── report.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/
│   ├── agents/
│   │   ├── strategist/
│   │   │   ├── prompt.py
│   │   │   ├── node.py
│   │   │   └── schema.py
│   │   ├── copywriter/
│   │   │   ├── prompt.py
│   │   │   ├── node.py
│   │   │   └── schema.py
│   │   ├── editor/
│   │   │   ├── prompt.py
│   │   │   ├── node.py
│   │   │   └── schema.py
│   │   └── publisher/
│   │       ├── prompt.py
│   │       ├── node.py
│   │       └── schema.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── build_graph.py
│   │   └── router.py
│   ├── llm/
│   │   ├── ollama_client.py
│   │   └── models_config.py
│   └── observability/
│       └── langfuse_setup.py
└── tests/
    ├── agents/
    │   ├── test_strategist.py
    │   ├── test_copywriter.py
    │   ├── test_editor.py
    │   └── test_publisher.py
    ├── graph/
    │   └── test_build_graph.py
    └── llm/
        └── test_models_config.py
```

Каждая роль — своя подпапка в `src/agents/` (промпт + нода + схема), тесты зеркалят эту структуру в `tests/agents/`.
