import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    strategist_model: str = os.getenv("STRATEGIST_MODEL", "qwen2.5:3b")
    copywriter_model: str = os.getenv("COPYWRITER_MODEL", "vikhr-llama-3.2-3b")
    editor_model: str = os.getenv("EDITOR_MODEL", "qwen2.5:3b")
    publisher_model: str = os.getenv("PUBLISHER_MODEL", "qwen2.5:1.5b")

    max_revision_rounds: int = int(os.getenv("MAX_REVISION_ROUNDS", "3"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))
    llm_request_timeout_s: float = float(os.getenv("LLM_REQUEST_TIMEOUT_S", "300"))

    niche: str = os.getenv("NICHE", "Сеть кофеен в Москве")
    social_network: str = os.getenv("SOCIAL_NETWORK", "VK")

    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY") or None
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY") or None
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    output_dir: str = os.getenv("OUTPUT_DIR", "data/logs")


def settings() -> Settings:
    return Settings()
