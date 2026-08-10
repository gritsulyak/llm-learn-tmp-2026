from src.graph.router import route_after_editor, route_after_publisher


def _base_state(**overrides):
    state = {
        "editor_verdict": "ok",
        "revision_round": 0,
        "max_revision_rounds": 3,
        "current_post_index": 1,
        "posts_to_write": [{"topic": "a"}, {"topic": "b"}],
    }
    state.update(overrides)
    return state


def test_route_after_editor_ok_goes_to_publisher():
    state = _base_state(editor_verdict="ok")
    assert route_after_editor(state) == "publisher"


def test_route_after_editor_revise_goes_back_to_copywriter():
    state = _base_state(editor_verdict="revise", revision_round=1, max_revision_rounds=3)
    assert route_after_editor(state) == "copywriter"


def test_route_after_editor_hits_revision_limit_forces_publish():
    state = _base_state(editor_verdict="revise", revision_round=3, max_revision_rounds=3)
    assert route_after_editor(state) == "publisher"


def test_route_after_publisher_continues_when_posts_remain():
    state = _base_state(current_post_index=1, posts_to_write=[{"topic": "a"}, {"topic": "b"}])
    assert route_after_publisher(state) == "copywriter"


def test_route_after_publisher_ends_when_all_posts_done():
    state = _base_state(current_post_index=2, posts_to_write=[{"topic": "a"}, {"topic": "b"}])
    assert route_after_publisher(state) == "__end__"
