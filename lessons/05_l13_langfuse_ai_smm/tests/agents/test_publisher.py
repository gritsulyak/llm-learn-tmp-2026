from src.agents.publisher.cleanup import (
    clean_final_text,
    remove_cjk,
    strip_meta_prefix,
)


def test_strip_meta_prefix_removes_leading_meta_line():
    text = "Вот готовый пост для Telegram:\n\nПривет! Заходи в наш магазин."
    assert strip_meta_prefix(text) == "Привет! Заходи в наш магазин."


def test_strip_meta_prefix_keeps_normal_text():
    text = "Привет! Заходи в наш магазин."
    assert strip_meta_prefix(text) == text


def test_remove_cjk_removes_cjk_characters():
    text = "Помогите宝宝们提升技能 и играйте!"
    cleaned = remove_cjk(text)
    assert "宝宝" not in cleaned
    assert "提升技能" not in cleaned
    assert cleaned.strip() != ""


def test_clean_final_text_full_case():
    text = "```\nВот готовый ответ:\n\nПривет! Заходи к нам!\n```"
    assert clean_final_text(text) == "Привет! Заходи к нам!"