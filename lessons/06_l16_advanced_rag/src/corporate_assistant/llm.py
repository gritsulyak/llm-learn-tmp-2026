"""LLM: обёртка над Yandex Cloud Foundation Models API."""

from __future__ import annotations

from typing import Any, Sequence

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import CustomLLM

from corporate_assistant.config import (
    LLM_TEMPERATURE,
    YC_API_KEY,
    YC_FOLDER_ID,
    YC_MODEL,
    YC_URL,
)

_ROLE_MAP = {
    MessageRole.SYSTEM: "system",
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
}


class YandexGPT(CustomLLM):
    """LlamaIndex-совместимая обёртка над Yandex Cloud Foundation Models."""

    api_key: str = ""
    folder_id: str = ""
    model: str = "yandexgpt-lite"
    url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    temperature: float = 0.0
    max_tokens: int = 2000

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=8192,
            num_output=self.max_tokens,
            is_chat_model=True,
        )

    def _messages_for_api(
        self, messages: Sequence[ChatMessage]
    ) -> list[dict[str, str]]:
        return [
            {"role": _ROLE_MAP.get(m.role, "user"), "text": m.content or ""}
            for m in messages
        ]

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),
            },
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
        }
        resp = requests.post(self.url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["result"]["alternatives"][0]["message"]["text"]

    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        api_messages = self._messages_for_api(messages)
        text = self._call_api(api_messages)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        api_messages = [{"role": "user", "text": prompt}]
        text = self._call_api(api_messages)
        return CompletionResponse(text=text)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def _gen() -> CompletionResponseGen:
            yield self.complete(prompt, formatted=formatted, **kwargs)

        return _gen()


def build_llm() -> YandexGPT:
    return YandexGPT(
        api_key=YC_API_KEY,
        folder_id=YC_FOLDER_ID,
        model=YC_MODEL,
        url=YC_URL,
        temperature=LLM_TEMPERATURE,
    )
