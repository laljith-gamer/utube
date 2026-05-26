"""LLM provider router.

Primary:  NVIDIA NIM   (gpt-oss-120b)
Fallback: Cerebras     (gpt-oss-120b, 14,400 RPD free)
Fallback: Groq         (gpt-oss-120b or llama-3.3-70b)
Fallback: OpenRouter   (free models)

All four expose an OpenAI-compatible API, so we use the openai SDK
with different base_urls.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI

from ..utils import env

LOG = logging.getLogger("utube.llm")


@dataclass
class LLMProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    model: str

    @property
    def api_key(self) -> str:
        return env(self.api_key_env)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


PROVIDERS: list[LLMProviderConfig] = [
    LLMProviderConfig(
        name="nvidia_nim",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_NIM_API_KEY",
        model="openai/gpt-oss-120b",
    ),
    LLMProviderConfig(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        model="gpt-oss-120b",
    ),
    LLMProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="openai/gpt-oss-120b",
    ),
    LLMProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="meta-llama/llama-3.3-70b-instruct:free",
    ),
]


class LLMRouter:
    """Tries each configured provider until one succeeds."""

    def __init__(self, providers: Iterable[LLMProviderConfig] | None = None):
        self.providers = list(providers) if providers else PROVIDERS
        self.active = [p for p in self.providers if p.configured]
        if not self.active:
            LOG.warning("No LLM providers configured. Set at least one *_API_KEY.")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        # gpt-oss reasoning consumes tokens against max_tokens before any answer
        # is emitted. Ensure plenty of headroom so we don't get empty/truncated output.
        if reasoning_effort:
            min_budget = {"low": 1500, "medium": 4000, "high": 6000}.get(reasoning_effort, 4000)
            max_tokens = max(max_tokens, min_budget)

        last_err: Exception | None = None
        for p in self.active:
            try:
                LOG.info("LLM call → %s (%s)", p.name, p.model)
                client = OpenAI(api_key=p.api_key, base_url=p.base_url, timeout=120.0)
                kwargs: dict[str, Any] = {
                    "model": p.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if reasoning_effort and "gpt-oss" in p.model:
                    kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                if not content:
                    # gpt-oss on some providers puts the answer in reasoning_content;
                    # OpenAI SDK surfaces unknown fields via model_extra.
                    extra = getattr(msg, "model_extra", None) or {}
                    for key in ("reasoning_content", "reasoning", "thinking"):
                        candidate = extra.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            content = candidate.strip()
                            LOG.info("  (used %s field as content)", key)
                            break
                if not content:
                    finish = getattr(resp.choices[0], "finish_reason", None)
                    raise RuntimeError(f"Empty response (finish_reason={finish})")
                return content
            except Exception as e:  # noqa: BLE001
                LOG.warning("LLM provider %s failed: %s", p.name, e)
                last_err = e
                # If 429 or quota, brief pause before next provider
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(2)
                continue
        raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Return parsed JSON, with robust fallback to regex extraction."""
        text = self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
            reasoning_effort=reasoning_effort,
        )
        return _parse_json(text)


def _parse_json(text: str) -> dict[str, Any]:
    """Try strict json.loads, then strip code fences, then regex, then a truncation repair."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip ``` fences
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Last resort: try to repair a truncated JSON object by closing
    # any unterminated string and any open braces/brackets.
    m = re.search(r"\{.*", text, flags=re.DOTALL)
    if m:
        repaired = _try_repair_truncated(m.group(0))
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}")


def _try_repair_truncated(s: str) -> str | None:
    """Best-effort: close an unterminated string, then close open braces/brackets."""
    # Walk the string tracking string state and brace/bracket depth
    in_str = False
    escape = False
    stack: list[str] = []
    for ch in s:
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return None  # malformed beyond simple truncation
    repaired = s
    if in_str:
        repaired += '"'
    # If we end mid key (e.g. ..."key": ) drop the trailing ", ?key?" before closing
    repaired = re.sub(r",\s*$", "", repaired)
    repaired = re.sub(r":\s*$", ': null', repaired)
    while stack:
        repaired += stack.pop()
    return repaired
