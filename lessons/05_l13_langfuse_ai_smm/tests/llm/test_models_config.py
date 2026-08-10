from src.llm.models_config import MODEL_BY_ROLE


def test_all_roles_have_a_model():
    for role in ("strategist", "copywriter", "editor", "publisher"):
        assert role in MODEL_BY_ROLE
        assert isinstance(MODEL_BY_ROLE[role], str)
        assert MODEL_BY_ROLE[role]
