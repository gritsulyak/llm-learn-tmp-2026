import os

from src.config import settings


def langfuse_enabled() -> bool:
    cfg = settings()
    return bool(cfg.langfuse_public_key and cfg.langfuse_secret_key)


def get_langfuse_handler():
    """Fresh CallbackHandler for a single run, or None if disabled."""
    if not langfuse_enabled():
        return None

    cfg = settings()
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", cfg.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", cfg.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", cfg.langfuse_host)

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def trace_config(niche: str, social_network: str) -> dict:
    """LangGraph invoke config tying this run's trace to a distinct langfuse trace."""
    cfg = settings()
    config = {"recursion_limit": 60}
    if langfuse_enabled():
        config["metadata"] = {
            "langfuse_trace_name": f"AI SMM: {niche} / {social_network}",
            "langfuse_session_id": f"{social_network}_{niche}",
            "niche": niche,
            "social_network": social_network,
        }
    return config


def shutdown_tracing() -> None:
    if not langfuse_enabled():
        return
    from langfuse import get_client

    get_client().shutdown()
