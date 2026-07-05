# %% [markdown]
# # RuTweetCorp: TF-IDF + MLP (PyTorch) для анализа тональности русского Твиттера
#
# 1. Загрузка и очистка данных
# 2. Векторизация TF-IDF и обучение MLP на PyTorch
# 3. Интерпретация весов
# 4. Метрики, матрица ошибок, ручное тестирование

# %%
# !pip install -q kagglehub scikit-learn pandas numpy matplotlib pymorphy3 pymorphy3-dicts-ru torch

# %%
import re
import html
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import kagglehub
import pymorphy3

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## 1. Загрузка данных
#
# Датасет: RuTweetCorp (корпус Ю. Рубцовой, mokoron.com) — русские твиты с
# бинарной тональностью. Метка закодирована файлом: `positive.csv` (label=1),
# `negative.csv` (label=0).
#
# https://www.kaggle.com/datasets/maximsuvorov/rutweetcorp

# %%
path = kagglehub.dataset_download("maximsuvorov/rutweetcorp")
print("Файлы датасета:", list(Path(path).glob("*.csv")))

df_pos = pd.read_csv(Path(path) / "positive.csv", usecols=["ttext", "ttype"])
df_neg = pd.read_csv(Path(path) / "negative.csv", usecols=["ttext", "ttype"])

df = pd.concat([df_pos, df_neg], ignore_index=True)
df = df.rename(columns={"ttext": "text"})
df["label"] = (df["ttype"] == 1).astype(int)  # 1 = позитив, 0 = негатив
df = df.drop(columns=["ttype"])

print(df.shape)
print(df["label"].value_counts())
df.head(3)

# %% [markdown]
# ## 2. Очистка текста регулярными выражениями
#
# Удаляем URL, `@упоминания`, `#хэштеги`, эмотиконы (`:D`, `:)`, `xD`),
# изолированные числа, спецсимволы и лишнюю пунктуацию, приводим к нижнему регистру.

# %%
def clean_text(text):
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # URL
    text = re.sub(r"@\w+", " ", text)                  # @mentions
    text = re.sub(r"#\w+", " ", text)                  # #hashtags
    text = re.sub(r"[:;xX]-?[)dpDP(]+", " ", text)      # эмотиконы :D :) :( xD
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)        # спецсимволы/пунктуация
    text = re.sub(r"\b\d+\b", " ", text)                # изолированные числа
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(clean_text)

print("До:   ", df["text"].iloc[0][:80])
print("После:", df["clean_text"].iloc[0][:80])

# %% [markdown]
# ## 3. Нормализация текста (лемматизация pymorphy3)

# %%
morph = pymorphy3.MorphAnalyzer()


def lemmatize(text):
    return " ".join(morph.parse(word)[0].normal_form for word in text.split())


df["lemma_text"] = df["clean_text"].apply(lemmatize)

print("Очищено:        ", df["clean_text"].iloc[0][:80])
print("Лемматизировано:", df["lemma_text"].iloc[0][:80])

# %% [markdown]
# ## 4. Train / test split

# %%
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"Train: {len(df_train)}, Test: {len(df_test)}")
print(df_train["label"].value_counts())
print(df_test["label"].value_counts())

# %% [markdown]
# ## 5. Векторизация TF-IDF
#
# `fit_transform` только на train, к test — только `transform` (без утечки данных).
# `min_df` / `max_df` отсекают слишком редкие и слишком частые слова.

# %%
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 1),
)

X_train = vectorizer.fit_transform(df_train["lemma_text"])
X_test = vectorizer.transform(df_test["lemma_text"])

y_train = df_train["label"].values.
y_test = df_test["label"].values

n_features = X_train.shape[1]

print(f"Размерность TF-IDF train: {X_train.shape}")
sparsity = 100 - (X_train.nnz / (X_train.shape[0] * X_train.shape[1]) * 100)
print(f"Sparsity: {sparsity:.2f}%")

# %% [markdown]
# ## 6. Архитектура MLP на PyTorch
#
# Многослойный перцептрон: два скрытых слоя (128, 64 нейрона), ReLU-активация,
# оптимизатор Adam. Обучение — мини-батчами, с ручным циклом.

# %%
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), num_classes=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(n_features, hidden_dims=(128, 64), num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

print(model)

# %% [markdown]
# ## 7. Обучение модели
#
# Переводим sparse TF-IDF в плотные тензоры батчами, чтобы не упереться в память.
# На каждой эпохе данные перемешиваются (`torch.randperm`).

# %%
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

X_train_tensor = torch.from_numpy(X_train_dense).float()
X_test_tensor = torch.from_numpy(X_test_dense).float().to(device)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test).to(device)

n_epochs = 20
batch_size = 512
n_samples = X_train_tensor.shape[0]
n_batches = max(1, n_samples // batch_size)

history = []

for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(n_samples)
    epoch_loss = 0.0

    for i in range(0, n_samples, batch_size):
        idx = permutation[i:i + batch_size]
        batch_x = X_train_tensor[idx].to(device)
        batch_y = y_train_tensor[idx].to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / n_batches
    history.append(avg_loss)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

print("Обучение завершено.")

# %%
plt.figure(figsize=(8, 4))
plt.plot(range(1, n_epochs + 1), history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss — MLP (PyTorch)")
plt.grid(True, alpha=0.3)
plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Метрики качества и матрица ошибок

# %%
model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_preds = test_logits.argmax(dim=1).cpu().numpy()
    test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

print(classification_report(
    y_test, test_preds, labels=[0, 1], target_names=["Bad", "Good"], zero_division=0
))

cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad", "Good"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix — RuTweetCorp (PyTorch MLP)")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Интерпретация весов слоёв
#
# "Сквозной" вклад каждого TF-IDF-признака в выход считаем как произведение
# весов всех линейных слоёв сети (упрощённая аппроксимация, но содержательная
# для понимания, какие слова тянут к позитивной/негативной тональности).

# %%
linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]

contribution = linear_layers[0].weight.detach().cpu().numpy().T  # (n_features, n_hidden1)
for layer in linear_layers[1:]:
    W = layer.weight.detach().cpu().numpy().T
    contribution = contribution @ W

signal = contribution[:, 1] - contribution[:, 0]  # разница вклада: класс Good vs Bad

vocab = np.array(vectorizer.get_feature_names_out())

top_positive_idx = signal.argsort()[-20:][::-1]
top_negative_idx = signal.argsort()[:20]

print("Топ-20 слов для позитивной тональности:")
for idx in top_positive_idx:
    print(f"  {vocab[idx]:<20} {signal[idx]:+.4f}")

print("\\nТоп-20 слов для негативной тональности:")
for idx in top_negative_idx:
    print(f"  {vocab[idx]:<20} {signal[idx]:+.4f}")

# %% [markdown]
# **Комментарий:** если в топ позитивных попадают слова типа "спасибо",
# "рад", "класс", а в топ негативных — "жалко", "плохо", "устал" — распределение
# логично и отражает реальную тональность. Хаотичные/редкие токены в топе —
# признак переобучения на шум, а не на смысл.

# %% [markdown]
# ## 10. Ручное тестирование на собственных примерах

# %%
manual_examples = [
    "Отличный день, я счастлив!",
    "Всё ужасно, ненавижу этот день",
    "Не сказать что плохо, но и не хорошо",
    "Ну конечно, шедевр, просто восторг",
    "Мне не понравился фильм совсем",
    "Спасибо огромное за помощь, вы супер",
    "Опять всё сломалось, как обычно",
]

manual_clean = [lemmatize(clean_text(t)) for t in manual_examples]
manual_vec = vectorizer.transform(manual_clean).toarray()
manual_tensor = torch.from_numpy(manual_vec).float().to(device)

model.eval()
with torch.no_grad():
    manual_logits = model(manual_tensor)
    manual_probs = torch.softmax(manual_logits, dim=1).cpu().numpy()

for text, probs in zip(manual_examples, manual_probs):
    label = "Good" if probs[1] > 0.5 else "Bad"
    print(f"[{label}] P(Good)={probs[1]:.3f}  —  {text}")

# %% [markdown]
# ## 11. Слабые места подхода и итоговый вывод
#
# TF-IDF представляет текст как "мешок слов" без учёта порядка — поэтому
# отрицание ("не понравился") не связывается со словом, которое оно меняет
# по смыслу. Сарказм и ирония вообще не распознаются: фраза "ну конечно,
# шедевр" формально состоит из позитивных слов, но означает противоположное.
#
# Математика TF-IDF+MLP проигрывает в анализе того что вокруг слова: 
# т.е. синтаксис, контекст, композициональность смысла.
# Трансформеры (BERT, GPT) строят контекстно-зависимые представления через
# self-attention, где смысл каждого слова зависит от всех остальных слов
# в предложении одновременно — поэтому они справляются с отрицаниями и
# сарказмом заметно лучше.
