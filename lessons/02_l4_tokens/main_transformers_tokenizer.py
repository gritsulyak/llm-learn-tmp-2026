# %% [markdown]
# # Custom Russian Tokenizer
#
# Домашнее задание: обучение собственного BPE-токенизатора на русском корпусе
# книжных переводов (Helsinki-NLP/opus_books, en-ru), сравнение с индустриальным
# решением (DeepPavlov/rubert-base-cased), анализ морфологии и метрик.

# %%
# !pip install -q datasets tokenizers transformers
# !pip install ipywidgets

# %%
import re
import json
from collections import Counter

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
from transformers import AutoTokenizer

# %% [markdown]
# ## 1. Подготовка корпуса данных
#
# Загружаем датасет opus_books с языковой парой en-ru и извлекаем только
# русские тексты из ключа `translation.ru`.

# %%
dataset = load_dataset("Helsinki-NLP/opus_books", "en-ru")
print(dataset)

# %%
raw_pairs = dataset["train"]["translation"]
ru_texts_raw = [pair["ru"] for pair in raw_pairs]

print(f"Всего пар предложений: {len(ru_texts_raw)}")
print("Пример до очистки:", ru_texts_raw[0])

# %% [markdown]
# ### Базовая предобработка
#
# Отфильтровываем пустые строки и артефакты форматирования (множественные
# пробелы, служебные символы вроде переводов строк внутри предложения).

# %%
def clean_line(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)          # множественные пробелы/переводы строк
    text = re.sub(r"[\u200b\ufeff]", "", text)  # невидимые артефакты форматирования
    return text


ru_texts = [clean_line(t) for t in ru_texts_raw]
ru_texts = [t for t in ru_texts if len(t) > 0]

print(f"Текстов после очистки: {len(ru_texts)}")
print("Пример после очистки:", ru_texts[0])

for i, text in enumerate(ru_texts[:10]):
    print(f"{i}: {text}")


# %% [markdown]
# ### Генератор батчей
#
# Чтобы не перегружать оперативную память, тексты подаются в тренер
# токенизатора порционно, а не единым списком.

# %%
def batch_iterator(texts, batch_size=1000):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


print("Пример батча (первые 2 текста из первого батча):")
first_batch = next(batch_iterator(ru_texts, batch_size=1000))
print(first_batch[:2])

# %% [markdown]
# ## 2. Проектирование и обучение токенизатора
#
# Алгоритм - BPE (Byte-Pair Encoding). Pre-tokenizer разбивает по пробелам
# и изолирует пунктуацию, чтобы знаки препинания не склеивались со словами.

# %%
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.NFKC()

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),   # разбиение по пробелам
    pre_tokenizers.Punctuation(),  # изоляция пунктуации
])

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

VOCAB_SIZE = 20000

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=special_tokens,
    show_progress=True,
)

# %%
tokenizer.train_from_iterator(batch_iterator(ru_texts), trainer=trainer, length=len(ru_texts))

vocab = tokenizer.get_vocab()
print(f"Размер итогового словаря: {len(vocab)}")
print("Примеры токенов из словаря:", list(vocab.keys())[:20])

# %%
tokenizer.save("custom_ru_tokenizer.json")
print("Токенизатор сохранён в custom_ru_tokenizer.json")

# %% [markdown]
# ## 3. Загрузка индустриального токенизатора для сравнения

# %%
industrial_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
print(f"Размер словаря индустриального токенизатора: {industrial_tokenizer.vocab_size}")

# %% [markdown]
# ### Тестовые предложения
#
# Морфологически сложные слова, составные слова, неологизмы/сленг,
# редкие имена собственные.

# %%
test_sentences = [
    "Переосмысливаемый недопонятый текст вызывает споры среди лингвистов.",
    "Сельскохозяйственный и железнодорожный секторы страны нуждаются в модернизации.",
    "Молодёжь любит загуглить новый кринж и обсудить его в чате.",
    "Вячеслав Ксенофонтович Прибыловский приехал в Домодедово рано утром.",
    "Непреодолимое желание перечитать недооценённое произведение охватило её.",
    "Постиндустриальное общество формирует новые социокультурные паттерны.",
    "Скукожившийся от холода котёнок жалобно мяукал под подъездом.",
]

# %% [markdown]
# ## Пословное сравнение токенизации

# %%
def tokenize_custom(text):
    return tokenizer.encode(text).tokens


def tokenize_industrial(text):
    return industrial_tokenizer.tokenize(text)


comparison_rows = []
for sent in test_sentences:
    custom_tokens = tokenize_custom(sent)
    industrial_tokens = tokenize_industrial(sent)
    comparison_rows.append({
        "sentence": sent,
        "custom_tokens": custom_tokens,
        "custom_count": len(custom_tokens),
        "industrial_tokens": industrial_tokens,
        "industrial_count": len(industrial_tokens),
    })

for row in comparison_rows:
    print(f"\\nПредложение: {row['sentence']}")
    print(f"  Custom     ({row['custom_count']:>2} токенов): {row['custom_tokens']}")
    print(f"  Industrial ({row['industrial_count']:>2} токенов): {row['industrial_tokens']}")

# %% [markdown]
# ### Пословное сравнение отдельных сложных слов

# %%
hard_words = [
    "переосмысливаемый",
    "недопонятый",
    "сельскохозяйственный",
    "железнодорожный",
    "загуглить",
    "кринж",
    "Вячеслав",
    "Прибыловский",
    "постиндустриальное",
    "скукожившийся",
    "задумчивый",
    "глубокомысленный",
    "великосветский",
    "ракетоноситель",
]

print(f"{'Слово':<24} {'Custom tokens':<45} {'Industrial tokens'}")
print("-" * 110)
for word in hard_words:
    c_tokens = tokenize_custom(word)
    i_tokens = tokenize_industrial(word)
    print(f"{word:<24} {str(c_tokens):<45} {i_tokens}")

# %% [markdown]
# ## 4. Расчёт базовых метрик

# %%
def count_words(text):
    return len(re.findall(r"[а-яёa-z]+", text.lower()))


total_words = sum(count_words(s) for s in test_sentences)
total_custom_tokens = sum(row["custom_count"] for row in comparison_rows)
total_industrial_tokens = sum(row["industrial_count"] for row in comparison_rows)

print(f"Слов в тестовой выборке: {total_words}")
print(f"Токенов (custom):     {total_custom_tokens}, "
      f"среднее токенов/слово: {total_custom_tokens / total_words:.2f}")
print(f"Токенов (industrial): {total_industrial_tokens}, "
      f"среднее токенов/слово: {total_industrial_tokens / total_words:.2f}")

# %% [markdown]
# ### Проверка [UNK]-токенов

# %%
unk_count_custom = sum(t == "[UNK]" for row in comparison_rows for t in row["custom_tokens"])
unk_count_industrial = sum(
    t == industrial_tokenizer.unk_token for row in comparison_rows for t in row["industrial_tokens"]
)

print(f"[UNK] токенов у custom-токенизатора:     {unk_count_custom}")
print(f"[UNK] токенов у industrial-токенизатора: {unk_count_industrial}")

unk_words = []
for row in comparison_rows:
    for word, tok in zip(row["sentence"].split(), row["custom_tokens"]):
        if tok == "[UNK]":
            unk_words.append(word)
print("Слова/токены, где custom встретил [UNK]:", unk_words)

# %% [markdown]
# ## 4.1 Опциональный эксперимент: изменение vocab_size
#
# Обучаем ещё две версии токенизатора - с уменьшенным и увеличенным словарём -
# и сравниваем, как это влияет на дробление слов.

# %%
def train_tokenizer_with_vocab_size(vocab_size, texts, special_tokens):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ])
    trainer_local = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=False,
    )
    tok.train_from_iterator(batch_iterator(texts), trainer=trainer_local, length=len(texts))
    return tok


tokenizer_small = train_tokenizer_with_vocab_size(VOCAB_SIZE // 4, ru_texts, special_tokens)
tokenizer_large = train_tokenizer_with_vocab_size(VOCAB_SIZE * 4, ru_texts, special_tokens)

print(f"{'Слово':<24} {'vocab='+str(VOCAB_SIZE//4):<35} {'vocab='+str(VOCAB_SIZE):<35} {'vocab='+str(VOCAB_SIZE*4)}")
print("-" * 130)
for word in hard_words:
    t_small = tokenizer_small.encode(word).tokens
    t_mid = tokenize_custom(word)
    t_large = tokenizer_large.encode(word).tokens
    print(f"{word:<24} {str(t_small):<35} {str(t_mid):<35} {t_large}")

# %% [markdown]
# ## 5. Отчёт и анализ
#
# **Удачные разбиения custom-токенизатора:** железнодорожный ['железнодоро', 'жный'] ; недопонятый  ['недо', 'поня', 'тый']  .
#
# **Неудачные разбиения:** ['за', 'гу', 'гли', 'ть'] ['кри', 'н', 'ж'] ['мо', 'дер', 'ни', 'за', 'ции'] - 21 века слова, их нет в корпусе. 
#
# **Окончания и приставки:** например, 'загуглить' и 'переосмысливаемый' - выделил окончания приставки для словаря 10000, 
# лучше чем индустриальный, но это может быть и не нужно. 'Вячеслав' и 'постиндустриальный' - не разобрал.
#
# **Итоговый вывод:** индустриальный токенизатор (rubert-base-cased) обучен на большем и современном корпусе (новости, википедия, форумы,
# современная лексика), поэтому справляется с неологизмами и сленгом значительно лучше. 
#     
# Собственный токенизатор, обученный только на корпусе классических книжных переводов (~XIX-XX века), не видел современных слов
# вроде  'загуглить' - помечает как [UNK] или дробит их на подслова/буквы, как 'кринж'. 
#     
# Размер словаря  влияет на степень дробления: маленький vocab_size заставляет алгоритм чаще резать слова на
# короткие фрагменты (subword/буквы), а большой - сохранять целые слова и частые словоформы как единые токены, 
# но требует больше данных для качественного обучения статистики слияний. Для современных слов - нужны современные корпусы. 
