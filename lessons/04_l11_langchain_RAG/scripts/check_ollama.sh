#!/usr/bin/env bash
# Проверка, что Ollama работает локально и модель отвечает без интернета.
set -e

MODEL="${1:-qwen3.5:4b}"

echo "== Проверка запущенных моделей =="
ollama ls

echo ""
echo "== Проверка ответа модели '$MODEL' в консоли =="
ollama run "$MODEL" "Привет! Ответь одним словом: работаешь?"

echo ""
echo "== Проверка API Ollama (http://localhost:11434) =="
curl -s http://localhost:11434/api/tags | head -c 300
echo ""
echo "✅ Готово."
