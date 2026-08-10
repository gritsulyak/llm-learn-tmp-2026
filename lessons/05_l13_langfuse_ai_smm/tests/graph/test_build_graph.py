from src.graph.build_graph import build_graph


def test_build_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    for expected in {"strategist", "copywriter", "editor", "publisher"}:
        assert expected in nodes
