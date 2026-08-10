import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.editor.prompt import EDITOR_SYSTEM_PROMPT
from src.agents.editor.schema import EditorVerdict
from src.graph.state import SMMState
from src.llm.models_config import MODEL_BY_ROLE
from src.llm.ollama_client import get_llm


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


def editor_node(state: SMMState) -> dict:
    idx = state["current_post_index"]
    post_spec = state["posts_to_write"][idx]
    print(f"[editor] checking post {idx + 1}, draft_len={len(state['draft_text'])}")

    llm = get_llm(MODEL_BY_ROLE["editor"])

    prompt = (
        f"Тема из контент-плана: {post_spec['topic']}\n"
        f"Формат: {post_spec['format']}\n"
        f"CTA-подсказка: {post_spec['cta_hint']}\n\n"
        f"Текст Копирайтера:\n{state['draft_text']}"
    )

    resp = llm.invoke([SystemMessage(content=EDITOR_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    data = _extract_json(resp.content)
    verdict = EditorVerdict.model_validate(data)

    rev = state.get("revision_round", 0)
    if verdict.verdict == "revise":
        rev += 1
    print(f"[editor] verdict={verdict.verdict}, revision_round={rev}")

    return {
        "editor_verdict": verdict.verdict,
        "editor_comments": verdict.comments,
        "revision_round": rev,
    }
