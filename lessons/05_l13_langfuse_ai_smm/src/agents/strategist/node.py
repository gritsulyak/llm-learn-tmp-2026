import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.strategist.prompt import STRATEGIST_SYSTEM_PROMPT
from src.agents.strategist.schema import ContentPlan
from src.graph.state import SMMState
from src.llm.models_config import MODEL_BY_ROLE
from src.llm.ollama_client import get_llm


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


def strategist_node(state: SMMState) -> dict:
    print(f"[strategist] niche={state['niche']!r}")

    llm = get_llm(MODEL_BY_ROLE["strategist"])
    user_prompt = (
        f"Ниша: {state['niche']}\n"
        f"Соцсеть: {state['social_network']}\n"
        "Сформируй контент-план на неделю (7 дней)."
    )
    resp = llm.invoke(
        [
            SystemMessage(content=STRATEGIST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    data = _extract_json(resp.content)
    plan = ContentPlan.model_validate(data).content_plan
    plan_dicts = [item.model_dump() for item in plan]

    posts_to_write = plan_dicts[:3]
    print(f"[strategist] plan items={len(plan_dicts)}, posts_to_write={len(posts_to_write)}")

    return {
        "content_plan": plan_dicts,
        "posts_to_write": posts_to_write,
        "current_post_index": 0,
        "revision_round": 0,
        "phase": "writing",
    }
