from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.copywriter.prompt import COPYWRITER_SYSTEM_PROMPT
from src.graph.state import SMMState
from src.llm.models_config import MODEL_BY_ROLE
from src.llm.ollama_client import get_llm


def copywriter_node(state: SMMState) -> dict:
    idx = state["current_post_index"]
    post_spec = state["posts_to_write"][idx]
    rev = state.get("revision_round", 0)
    print(f"[copywriter] post {idx + 1}/{len(state['posts_to_write'])}, revision {rev}")

    llm = get_llm(MODEL_BY_ROLE["copywriter"])

    task = (
        f"ТЗ от Стратега:\n"
        f"Тема: {post_spec['topic']}\n"
        f"Формат: {post_spec['format']}\n"
        f"CTA-подсказка: {post_spec['cta_hint']}\n"
        f"Ниша: {state['niche']}"
    )

    messages = [SystemMessage(content=COPYWRITER_SYSTEM_PROMPT), HumanMessage(content=task)]

    if state.get("editor_comments"):
        messages.append(
            HumanMessage(
                content=(
                    "Твой предыдущий вариант:\n"
                    f"{state['draft_text']}\n\n"
                    "Комментарии Редактора (исправь именно это):\n"
                    f"{state['editor_comments']}"
                )
            )
        )

    resp = llm.invoke(messages)

    return {
        "draft_text": resp.content.strip(),
        "phase": "editing",
    }
