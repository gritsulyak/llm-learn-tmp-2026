from unittest.mock import MagicMock, patch

from src.agents.strategist.node import strategist_node


PLAN_JSON = """{
  "content_plan": [
    {"day": "Понедельник", "topic": "Новый сорт кофе", "format": "пост", "cta_hint": "зайти в кофейню"},
    {"day": "Вторник", "topic": "История бариста", "format": "карусель", "cta_hint": "подписаться"},
    {"day": "Среда", "topic": "Акция дня", "format": "пост", "cta_hint": "заказать сегодня"},
    {"day": "Четверг", "topic": "Опрос вкусов", "format": "опрос", "cta_hint": "проголосовать"},
    {"day": "Пятница", "topic": "Пятничный сет", "format": "пост", "cta_hint": "прийти вечером"},
    {"day": "Субота", "topic": "Отзывы гостей", "format": "сторис", "cta_hint": "оставить отзыв"},
    {"day": "Воскресенье", "topic": "Итоги недели", "format": "пост", "cta_hint": "подписаться на рассылку"}
  ]
}"""


@patch("src.agents.strategist.node.get_llm")
def test_strategist_node_returns_seven_day_plan(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=PLAN_JSON)
    mock_get_llm.return_value = mock_llm

    state = {"niche": "Сеть кофеен в Москве", "social_network": "VK"}
    result = strategist_node(state)

    assert len(result["content_plan"]) == 7
    assert len(result["posts_to_write"]) == 3
    assert result["current_post_index"] == 0
    assert result["phase"] == "writing"
