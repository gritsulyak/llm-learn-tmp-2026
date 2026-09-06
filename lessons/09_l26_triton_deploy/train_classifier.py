"""Fine-tune TinyBERT classifier head for statement-vs-question and export ONNX.

Runtime: a few minutes on CPU (4-layer TinyBERT). Output goes straight into
model_repository/classification/1/ (gitignored: model.onnx / config.json).
Usage:
    uv run python train_classifier.py
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "huawei-noah/TinyBERT_General_4L_312D"
OUT_DIR = Path("model_repository/classification/1")
TEMP_DIR = Path("/tmp/tinybert_statement_question")
SEED = 42

statements = [
    "The sky is blue.",
    "The earth is round.",
    "The earth is not flat.",
    "Water freezes at zero degrees.",
    "I live in Moscow.",
    "I like pizza.",
    "Cats are smaller than dogs.",
    "The office is open from nine to six.",
    "My favorite color is green.",
    "Python is a programming language.",
    "The meeting starts at ten o'clock.",
    "London is the capital of England.",
    "Bread is made from flour.",
    "The train arrives at noon.",
]

questions = [
    "What is machine learning?",
    "What is the capital of France?",
    "Why is the sky blue?",
    "How does GPS work?",
    "How do I get to the airport?",
    "When does the bus leave?",
    "Where can I find a good restaurant?",
    "Who wrote the play Hamlet?",
    "Which phone is better?",
    "Is the store open today?",
    "Are you going to the party?",
    "Do you like coffee?",
    "Does the computer work?",
    "Can you help me?",
    "Could you repeat that, please?",
    "Would you like some tea?",
    "What time is it?",
]

topic_adj = {
    "the sky": ["blue", "gray", "clear", "cloudy"],
    "the earth": ["round", "large", "old", "round"],
    "water": ["wet", "cold", "transparent", "warm"],
    "winter": ["cold", "long", "snowy", "short"],
    "the internet": ["fast", "slow", "useful", "popular"],
    "coffee": ["strong", "bitter", "hot", "aromatic"],
    "the movie": ["long", "boring", "funny", "popular"],
    "trains": ["fast", "crowded", "reliable", "expensive"],
    "bicycles": ["cheap", "eco friendly", "simple", "fast"],
    "the test": ["easy", "hard", "long", "important"],
    "the server": ["fast", "busy", "remote", "reliable"],
    "the model": ["accurate", "slow", "small", "large"],
}

things = ["pizza", "tea", "guitar", "garden", "library", "hotel",
          "camera", "bike", "map", "umbrella", "notebook", "piano"]
verbs = ["works", "runs", "grows", "opens", "moves", "helps",
         "fits", "improves", "exists", "happens"]
actions = ["enjoy", "need", "love", "understand", "remember", "believe"]
adverbs = ["well", "fast", "slowly", "often", "rarely", "quickly"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_dataset() -> list[tuple[str, int]]:
    data: list[tuple[str, int]] = [(s, 0) for s in statements]
    data += [(q, 1) for q in questions]

    for topic, adjs in topic_adj.items():
        for adj in adjs:
            data.append((f"{topic} is {adj}.", 0))
            data.append((f"Is {topic} {adj}?", 1))
            data.append((f"Why is {topic} {adj}?", 1))
    for thing in things:
        data.append((f"I like {thing}.", 0))
        data.append((f"Where can I find {thing}?", 1))
        data.append((f"Do you like {thing}?", 1))
    for verb in verbs:
        data.append((f"The system {verb} well.", 0))
        data.append((f"How does the system {verb}?", 1))
        for adv in ["well", "slowly"]:
            data.append((f"The system {verb} {adv}.", 0))
            data.append((f"How {adv} does the system {verb}?", 1))
    for act in actions:
        data.append((f"Most people {act} good service.", 0))
        data.append((f"Why do people {act} good service?", 1))
    for adv in adverbs:
        data.append((f"The machine runs {adv}.", 0))
        data.append((f"How {adv} does the machine run?", 1))

    known_question_words = ["When", "Where", "Who", "Which", "Whose"]
    for w in known_question_words:
        data.append((f"{w} did the manager leave?", 1))
        data.append((f"{w} does the office close?", 1))

    # shuffle and dedupe (keep first)
    rng = random.Random(SEED)
    rng.shuffle(data)
    seen: set[str] = set()
    deduped = []
    for text, label in data:
        if text not in seen:
            seen.add(text)
            deduped.append((text, label))
    return deduped


def main() -> None:
    dataset = build_dataset()
    labels = [l for _, l in dataset]
    print(f"dataset: {len(dataset)} samples, "
          f"statements={labels.count(0)}, questions={labels.count(1)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    texts = [t for t, _ in dataset]
    labels_t = torch.tensor(labels, dtype=torch.long)

    def encode(batch_texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(n * 0.85)
    train_idx, eval_idx = idx[:split], idx[split:]

    enc_all = encode(texts)
    train_enc = {k: v[train_idx] for k, v in enc_all.items()}
    eval_enc = {k: v[eval_idx] for k, v in enc_all.items()}
    train_labels = labels_t[train_idx]
    eval_labels = labels_t[eval_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    batch_size = 16
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(train_idx))
        total = 0
        correct = 0
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            batch = {k: v[b] for k, v in train_enc.items()}
            out = model(**batch, labels=train_labels[b])
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = out.logits.detach().argmax(-1)
            correct += (preds == train_labels[b]).sum().item()
            total += len(b)
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            out = model(labels=eval_labels, **eval_enc)
            eval_acc = (out.logits.argmax(-1) == eval_labels).float().mean().item()
        model.train()
        print(f"epoch {epoch + 1}: train_acc={train_acc:.3f} "
              f"eval_acc={eval_acc:.3f} loss={float(loss):.4f}")

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    model.save_pretrained(TEMP_DIR)
    tokenizer.save_pretrained(TEMP_DIR)

    from optimum.onnxruntime import ORTModelForSequenceClassification

    print("exporting to ONNX ...")
    try:
        os.remove(OUT_DIR / "model.onnx")
    except FileNotFoundError:
        pass
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        TEMP_DIR, export=True
    )
    ort_model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"\nOK -> {OUT_DIR}")
    probe = ["What is machine learning?", "The earth is not flat.",
             "How does GPS work?", "I like pizza."]
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT_DIR / "model.onnx"))
    for text in probe:
        enc = tokenizer(text, padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt")
        feeds = {
            "input_ids": enc["input_ids"].numpy().astype(np.int64),
            "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
            "token_type_ids": enc["token_type_ids"].numpy().astype(np.int64),
        }
        logits = sess.run(None, feeds)[0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        print(f"  {text!r:32} label={int(p.argmax())} p={float(p.max()):.3f}")


if __name__ == "__main__":
    main()