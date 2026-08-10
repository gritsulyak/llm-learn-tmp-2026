from typing import Literal, TypedDict

Phase = Literal["planning", "writing", "editing", "publishing", "done"]


class SMMState(TypedDict):
    niche: str
    social_network: str

    content_plan: list[dict]           # [{day, topic, format, cta_hint}, ...]
    posts_to_write: list[dict]          # subset of content_plan chosen for drafting

    current_post_index: int
    draft_text: str
    editor_verdict: Literal["ok", "revise"]
    editor_comments: str
    revision_round: int
    max_revision_rounds: int

    finished_posts: list[dict]          # [{topic, draft, final_text}, ...]

    phase: Phase
    error: str | None
