from src.config import settings

cfg = settings()

MODEL_BY_ROLE = {
    "strategist": cfg.strategist_model,
    "copywriter": cfg.copywriter_model,
    "editor": cfg.editor_model,
    "publisher": cfg.publisher_model,
}
