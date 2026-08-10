from unittest.mock import MagicMock, patch

from src.agents.editor.node import editor_node


def _state():
    return {
        "current_post_index": 0,
        "posts_to_write": [
            {"topic": "Скидка на кофе", "format": "пост", "cta_hint": "приходи сегодня"}
        ],
        "draft_text": "Мы осуществляем деятельность по продаже кофе.",
        "revision_round": 0,
    }


@patch("src.agents.editor.node.get_llm")
def test_editor_node_parses_revise_verdict(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"verdict": "revise", "comments": "убери канцеляризм"}'
    )
    mock_get_llm.return_value = mock_llm

    result = editor_node(_state())

    assert result["editor_verdict"] == "revise"
    assert "канцеляризм" in result["editor_comments"]
    assert result["revision_round"] == 1


@patch("src.agents.editor.node.get_llm")
def test_editor_node_parses_ok_verdict(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"verdict": "ok", "comments": ""}')
    mock_get_llm.return_value = mock_llm

    result = editor_node(_state())

    assert result["editor_verdict"] == "ok"
    assert result["revision_round"] == 0
