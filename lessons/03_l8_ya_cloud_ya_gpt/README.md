# HOW THIS WAS MADE:

cd github/gritsulyak/llm-learn-tmp-2026/lessons/03_l8_ya_cloud_ya_gpt

### Инициализировать uv-проект (создаст pyproject.toml)
uv init --no-workspace

### Добавить зависимости (uv заменяет pip install)
uv add requests python-dotenv yandex-gpt jupyter

### Создать .env (ключи НЕ коммитить в git)
cat > .env << 'EOF'
YC_FOLDER_ID=b1..
YC_API_KEY=...
YC_API_KEY_ID=..
EOF

### Добавить .env в .gitignore
echo ".env" >> .gitignore

