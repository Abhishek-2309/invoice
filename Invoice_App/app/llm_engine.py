import os
import httpx
from typing import List, Dict, Any

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "changeme")

DEFAULT_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "180"))
DEFAULT_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "3000"))
DEFAULT_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.0"))


async def chat_async(
    messages: List[Dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None
) -> str:
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        "max_tokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]



def chat(
    messages: List[Dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None
) -> str:
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        "max_tokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    r = httpx.post(
        f"{VLLM_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

