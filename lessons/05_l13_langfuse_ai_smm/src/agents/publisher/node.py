from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.publisher.cleanup import clean_final_text
from src.agents.publisher.prompt import PUBLISHER_SYSTEM_PROMPT
from src.graph.state import SMMState
from src.llm.models_config import MODEL_BY_ROLE
from src.llm.ollama_client import get_llm


def publisher_node(state: SMMState) -> dict:
    idx = state["current_post_index"]
    post_spec = state["posts_to_write"][idx]
    print(f"[publisher] finalizing post {idx + 1} for {state['social_network']}")

    llm = get_llm(MODEL_BY_ROLE["publisher"])

    prompt = (
        f"Соцсеть: {state['social_network']}\n"
        f"Тема: {post_spec['topic']}\n\n"
        f"Текст, одобренный Редактором:\n{state['draft_text']}"
    )
    resp = llm.invoke([SystemMessage(content=PUBLISHER_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    final_text = clean_final_text(resp.content)

    finished = state.get("finished_posts", []) + [
        {
            "topic": post_spec["topic"],
            "draft": state["draft_text"],
            "final_text": final_text,
        }
    ]

    return {
        "finished_posts": finished,
        "current_post_index": idx + 1,
        "revision_round": 0,
        "editor_comments": "",
        "phase": "publishing",
    }
