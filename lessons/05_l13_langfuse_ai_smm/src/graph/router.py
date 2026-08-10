from typing import Literal

from src.graph.state import SMMState


def route_after_editor(state: SMMState) -> Literal["copywriter", "publisher"]:
    """Conditional edge: bad draft -> back to Copywriter, good draft -> Publisher.

    Hard cap on revision cycles to avoid infinite loops.
    """
    if state["editor_verdict"] == "ok":
        return "publisher"

    if state["revision_round"] >= state["max_revision_rounds"]:
        print(
            f"[router] max revision rounds reached "
            f"({state['revision_round']}/{state['max_revision_rounds']}), forcing publish"
        )
        return "publisher"

    return "copywriter"


def route_after_publisher(state: SMMState) -> Literal["copywriter", "__end__"]:
    """After a post is published, either move to the next post or finish."""
    if state["current_post_index"] < len(state["posts_to_write"]):
        return "copywriter"
    return "__end__"
