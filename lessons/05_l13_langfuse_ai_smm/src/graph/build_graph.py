from langgraph.graph import END, StateGraph

from src.agents.copywriter.node import copywriter_node
from src.agents.editor.node import editor_node
from src.agents.publisher.node import publisher_node
from src.agents.strategist.node import strategist_node
from src.graph.router import route_after_editor, route_after_publisher
from src.graph.state import SMMState


def build_graph():
    g = StateGraph(SMMState)

    g.add_node("strategist", strategist_node)
    g.add_node("copywriter", copywriter_node)
    g.add_node("editor", editor_node)
    g.add_node("publisher", publisher_node)

    g.set_entry_point("strategist")
    g.add_edge("strategist", "copywriter")
    g.add_edge("copywriter", "editor")

    g.add_conditional_edges(
        "editor",
        route_after_editor,
        {"copywriter": "copywriter", "publisher": "publisher"},
    )

    g.add_conditional_edges(
        "publisher",
        route_after_publisher,
        {"copywriter": "copywriter", "__end__": END},
    )

    return g.compile()
