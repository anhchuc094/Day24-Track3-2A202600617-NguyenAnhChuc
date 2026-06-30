from __future__ import annotations

"""Shared OpenAI-compatible LLM client — supports OpenAI and Groq via same SDK."""

import os
from importlib.util import find_spec

from config import (
    OPENAI_API_KEY, GROQ_API_KEY,
    LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE,
)


def _active_api_key() -> str:
    """Trả về API key của provider đang dùng."""
    if LLM_PROVIDER == "groq":
        return GROQ_API_KEY
    return OPENAI_API_KEY


def llm_available() -> bool:
    return bool(_active_api_key()) and find_spec("openai") is not None


def create_llm_client():
    """Tạo OpenAI-compatible client.
    
    Groq dùng cùng OpenAI SDK với base_url trỏ về Groq endpoint.
    """
    if not llm_available():
        return None

    from openai import OpenAI

    if LLM_PROVIDER == "groq":
        return OpenAI(api_key=GROQ_API_KEY, base_url=LLM_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)


def chat_completion(
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> str:
    """Gọi LLM API (OpenAI hoặc Groq — tùy config).
    
    Note: Groq không hỗ trợ response_format={"type": "json_object"} ở mọi model.
    Nếu dùng Groq, response_format sẽ được bỏ qua để tránh lỗi.
    """
    client = create_llm_client()
    if client is None:
        return ""

    kwargs: dict = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # Groq có hạn chế với response_format — chỉ set khi dùng OpenAI
    if response_format is not None and LLM_PROVIDER != "groq":
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def configure_ragas_environment() -> None:
    """Cấu hình env vars để RAGAS/LangChain dùng đúng provider.
    
    - OpenAI: set OPENAI_API_KEY (nếu chưa có)
    - Groq: set OPENAI_API_KEY=GROQ_API_KEY + OPENAI_BASE_URL → Groq endpoint
    """
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        # RAGAS dùng LangChain OpenAI client → redirect về Groq
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
        if not os.environ.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = LLM_BASE_URL
    elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
