"""LLM provider router — fully driven by config/providers.yaml > llm.

No URLs, model names, or token floors are hardcoded here. Edit the YAML to
add/reorder providers, swap models, or retune reasoning budgets.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.llm")


class LLMRouter:
    """Tries each provider in `llm.chain` until one succeeds."""

    def __init__(self, config_key: str = "llm") -> None:
        cfg = get_config()
        self.cfg = cfg.get_path(config_key, {}) or {}
        chain = self.cfg.get("chain", [])
        providers_block = self.cfg.get("providers", {}) or {}
        self.timeout = self.cfg.get("request_timeout_sec", 120)
        self.token_floor = self.cfg.get("reasoning_token_floor", {}) or {}
        self.retry_pause = self.cfg.get("retry_on_rate_limit_seconds", 2)

        self.active: list[dict[str, Any]] = []
        for name in chain:
            p = providers_block.get(name)
            if not p:
                LOG.warning("LLM provider %r in chain but not defined in providers", name)
                continue
            key = env(p.get("api_key_env", ""))
            if not key:
                continue
            self.active.append({"name": name, "api_key": key, **p})

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
        # Reasoning models burn tokens internally; bump max_tokens to a safe floor.
        if reasoning_effort:
            min_budget = self.token_floor.get(reasoning_effort, 4000)
            max_tokens = max(max_tokens, min_budget)

        last_err: Exception | None = None
        for p in self.active:
            try:
                provider_params = p.get("params", {})
                provider_timeout = p.get("request_timeout_sec", self.timeout)
                provider_retries = p.get("max_retries", 0 if "request_timeout_sec" in p else 2)
                
                LOG.info(
                    "LLM call → %s (%s) timeout=%ss retries=%s",
                    p["name"], p["model"], provider_timeout, provider_retries
                )
                
                is_gemini = "gemini" in p["model"].lower() or p.get("api_key_env") == "GEMINI_API_KEY"
                is_puter = p.get("api_key_env") == "PUTER_AUTH_TOKEN" or p["name"] == "puter_rewrite"

                if is_puter:
                    from .puter import PuterProvider
                    content = PuterProvider.chat(
                        model=p["model"],
                        messages=messages,
                        max_tokens=provider_params.get("max_tokens", max_tokens),
                        temperature=provider_params.get("temperature", temperature),
                        json_mode=json_mode
                    )
                    if not content:
                        raise RuntimeError("Empty response from Puter.")
                    return content
                elif is_gemini:
                    from google import genai
                    from google.genai import types
                    
                    client = genai.Client(api_key=p["api_key"])
                    
                    sys_inst = None
                    gemini_msgs = []
                    for m in messages:
                        if m["role"] == "system":
                            sys_inst = m["content"]
                        else:
                            role = "user" if m["role"] == "user" else "model"
                            gemini_msgs.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
                    
                    config_kwargs = {
                        "max_output_tokens": provider_params.get("max_tokens", max_tokens),
                    }
                    if json_mode:
                        config_kwargs["response_mime_type"] = "application/json"
                        
                    if reasoning_effort and p.get("supports_reasoning_effort"):
                        if reasoning_effort == "low":
                            tl = types.ThinkingLevel.LOW
                        elif reasoning_effort == "high":
                            tl = types.ThinkingLevel.HIGH
                        else:
                            tl = types.ThinkingLevel.MEDIUM
                        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=tl)
                        
                    if sys_inst:
                        config_kwargs["system_instruction"] = sys_inst
                        
                    config = types.GenerateContentConfig(**config_kwargs)
                    resp = client.models.generate_content(model=p["model"], contents=gemini_msgs, config=config)
                    content = (resp.text or "").strip()
                    finish = getattr(resp.candidates[0] if resp.candidates else None, "finish_reason", None)
                    
                    if not content:
                        raise RuntimeError(
                            f"Empty response from Gemini (finish_reason={finish}). "
                            f"Try lower reasoning_effort or higher max_tokens."
                        )
                    return content
                else:
                    client = OpenAI(
                        api_key=p["api_key"], 
                        base_url=p["base_url"], 
                        timeout=provider_timeout,
                        max_retries=provider_retries
                    )
                    kwargs: dict[str, Any] = {
                        "model": p["model"],
                        "messages": messages,
                        "max_tokens": provider_params.get("max_tokens", max_tokens),
                        "temperature": provider_params.get("temperature", temperature),
                    }
                    if "top_p" in provider_params:
                        kwargs["top_p"] = provider_params["top_p"]
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    if reasoning_effort and p.get("supports_reasoning_effort"):
                        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
                    resp = client.chat.completions.create(**kwargs)
                    choice = resp.choices[0]
                    msg = choice.message
                    content = (msg.content or "").strip()
                    finish = getattr(choice, "finish_reason", None)
    
                    if not content:
                        # gpt-oss reasoning models put their thinking in
                        # `reasoning_content`. Sometimes (rarely) the actual answer
                        # also leaks there. Only treat it as the answer if it
                        # *looks* like the kind of output we asked for —
                        # otherwise it's just truncated thinking.
                        extra = getattr(msg, "model_extra", None) or {}
                        for key in ("reasoning_content", "reasoning", "thinking"):
                            cand = extra.get(key)
                            if not (isinstance(cand, str) and cand.strip()):
                                continue
                            cand = cand.strip()
                            if json_mode and not _looks_like_json(cand):
                                LOG.warning(
                                    "  %s present but does not look like JSON "
                                    "(finish_reason=%s, %d chars) — likely truncated reasoning, "
                                    "falling through to next provider",
                                    key, finish, len(cand),
                                )
                                continue
                            content = cand
                            LOG.info("  (used %s field as content)", key)
                            break
    
                    if not content:
                        raise RuntimeError(
                            f"Empty response (finish_reason={finish}). "
                            f"Likely the model ran out of tokens during reasoning. "
                            f"Try lower reasoning_effort or higher max_tokens."
                        )
                    return content
            except Exception as e:  # noqa: BLE001
                LOG.warning("LLM provider %s failed: %s", p["name"], e)
                last_err = e
                if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                    time.sleep(self.retry_pause)
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
        text = self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
            reasoning_effort=reasoning_effort,
        )
        return _parse_json(text)


# ---------- robust JSON extraction ----------

def _looks_like_json(s: str) -> bool:
    """Cheap test: does this string contain a JSON object (vs prose)?"""
    s = s.lstrip()
    # Strip optional code fence
    s = re.sub(r"^```(?:json)?\s*", "", s)
    return s.startswith("{") or s.startswith("[")


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    # Remove opening fence: ```json or ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    # Remove closing fence
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict[str, Any]:
    """Robust JSON extraction: strip fences → parse → find block → repair truncated.

    Order of attempts:
    1. Direct json.loads on raw text.
    2. Strip markdown code fences, try again.
    3. Extract first {...} block, try again.
    4. Repair truncated JSON (open brackets/strings) and try again.
    If all fail, raise ValueError with the first 400 chars for debugging.
    """
    original = text
    text = text.strip()

    # Attempt 1: raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip code fences, then parse
    clean = _strip_fences(text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Attempt 3: find first { ... } (greedy) block and parse
    m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Attempt 4: find opening brace and try to repair truncation
    m = re.search(r"\{.*", clean, flags=re.DOTALL)
    if m:
        truncated = m.group(0)
        LOG.warning(
            "_parse_json: output appears truncated at %d chars, attempting repair …",
            len(original),
        )
        repaired = _try_repair_truncated(truncated)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from LLM output: {original[:400]}")


def _try_repair_truncated(s: str) -> str | None:
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
                return None
    repaired = s
    if in_str:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired)
    repaired = re.sub(r":\s*$", ": null", repaired)
    while stack:
        repaired += stack.pop()
    return repaired
