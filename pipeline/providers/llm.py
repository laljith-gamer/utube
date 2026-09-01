"""LLM provider router — fully driven by config/providers.yaml > llm.

No URLs, model names, or token floors are hardcoded here. Edit the YAML to
add/reorder providers, swap models, or retune reasoning budgets.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
import random
from enum import Enum, auto
from typing import Any

from openai import OpenAI

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.llm")

class ProviderStatus(Enum):
    SUCCESS = auto()
    TRANSIENT = auto()
    OUTPUT = auto()
    PERMANENT = auto()

@dataclasses.dataclass
class ProviderResult:
    status: ProviderStatus
    provider: str
    model: str
    content: str = ""
    parsed: dict[str, Any] | None = None
    failure_type: str = ""
    finish_reason: str = ""
    error_summary: str = ""
    latency_ms: float = 0.0
    attempt: int = 1
    retryable: bool = False


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

    def get_provider(self, idx: int) -> dict[str, Any] | None:
        if 0 <= idx < len(self.active):
            return self.active[idx]
        return None

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        provider_idx: int | None = None,
        attempt: int = 1,
    ) -> ProviderResult:
        if reasoning_effort:
            min_budget = self.token_floor.get(reasoning_effort, 4000)
            max_tokens = max(max_tokens, min_budget)

        providers_to_try = [self.active[provider_idx]] if provider_idx is not None and 0 <= provider_idx < len(self.active) else self.active

        last_err: Exception | None = None
        for p in providers_to_try:
            start_time = time.time()
            provider_name = p["name"]
            model_name = p["model"]
            result = ProviderResult(
                status=ProviderStatus.TRANSIENT,
                provider=provider_name,
                model=model_name,
                attempt=attempt,
                retryable=True
            )
            
            provider_params = p.get("params", {})
            provider_timeout = p.get("request_timeout_sec", self.timeout)
            max_provider_retries = p.get("max_retries", 4)
            
            current_messages = list(messages)
            accumulated_content = ""
            continuation_count = 0
            MAX_CONTINUATIONS = 5

            provider_success = False

            while continuation_count <= MAX_CONTINUATIONS:
                chunk_content = ""
                finish = "stop"
                retry_success = False

                for retry_attempt in range(max_provider_retries + 1):
                    try:
                        fault = os.environ.get("CONCEPT_FAULT_INJECTION")
                        if fault:
                            if fault == "gemini_503" and "gemini" in provider_name.lower():
                                raise RuntimeError("503 Service Unavailable")
                            if fault == "openrouter_length" and "openrouter" in provider_name.lower():
                                result.content = accumulated_content + "{\n  \"angles\": [\n    {\n      \"angle\": \"Some angle"
                                result.finish_reason = "length"
                                result.status = ProviderStatus.OUTPUT
                                result.failure_type = "truncated_output"
                                result.latency_ms = (time.time() - start_time) * 1000
                                return result
                            if fault == "nvidia_truncated" and "nvidia" in provider_name.lower():
                                result.content = accumulated_content + "{\n  \"angles\": ["
                                result.finish_reason = "stop"
                                result.status = ProviderStatus.SUCCESS
                                result.latency_ms = (time.time() - start_time) * 1000
                                return result

                        LOG.info(
                            "LLM call → %s (%s) timeout=%ss retries=%s attempt=%s (cont: %s)",
                            provider_name, model_name, provider_timeout, max_provider_retries, retry_attempt + 1, continuation_count
                        )
                        
                        is_gemini = "gemini" in model_name.lower() or p.get("api_key_env") == "GEMINI_API_KEY"
                        is_puter = p.get("api_key_env") == "PUTER_AUTH_TOKEN" or "puter" in provider_name.lower()
                        
                        if not is_puter:
                            raise RuntimeError(f"HARD ASSERTION FAILED: NON-PUTER PROVIDER DETECTED IN PUTER-ONLY MODE ({provider_name})")

                        if is_puter:
                            from .puter import PuterProvider
                            
                            # Preflight & get the exact model ID
                            actual_model = PuterProvider.preflight(auth_token=p.get("key"))
                            if retry_attempt == 0:
                                LOG.info("  (Using exact Puter runtime model ID: %s)", actual_model)
                            resp_dict = PuterProvider.chat(
                                model=actual_model,
                                messages=current_messages,
                                max_tokens=provider_params.get("max_tokens", max_tokens),
                                temperature=provider_params.get("temperature", temperature),
                                json_mode=json_mode,
                                auth_token=p.get("key")
                            )
                            if resp_dict.get("error"):
                                if resp_dict.get("is_rate_limit"):
                                    raise RuntimeError(f"Rate Limit 429: {resp_dict['error']}")
                                raise RuntimeError(f"Puter Error: {resp_dict['error']}")
                                
                            chunk_content = resp_dict.get("text", "")
                            finish = resp_dict.get("finishReason", "stop")

                        elif is_gemini:
                            from google import genai
                            from google.genai import types
                            
                            client = genai.Client(api_key=p["api_key"])
                            
                            sys_inst = None
                            gemini_msgs = []
                            for m in current_messages:
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
                            resp = client.models.generate_content(model=model_name, contents=gemini_msgs, config=config)
                            chunk_content = (resp.text or "").strip()
                            raw_finish = getattr(resp.candidates[0] if resp.candidates else None, "finish_reason", None)
                            finish = str(raw_finish.name if hasattr(raw_finish, "name") else raw_finish).lower()

                        else:
                            client = OpenAI(
                                api_key=p["api_key"], 
                                base_url=p["base_url"], 
                                timeout=provider_timeout,
                                max_retries=max_provider_retries
                            )
                            kwargs: dict[str, Any] = {
                                "model": model_name,
                                "messages": current_messages,
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
                            chunk_content = (msg.content or "").strip()
                            finish = str(getattr(choice, "finish_reason", None)).lower()
            
                            if not chunk_content:
                                extra = getattr(msg, "model_extra", None) or {}
                                for key in ("reasoning_content", "reasoning", "thinking"):
                                    cand = extra.get(key)
                                    if not (isinstance(cand, str) and cand.strip()):
                                        continue
                                    cand = cand.strip()
                                    if json_mode and not _looks_like_json(cand):
                                        LOG.warning(
                                            "  %s present but does not look like JSON "
                                            "(finish_reason=%s, %d chars)",
                                            key, finish, len(cand),
                                        )
                                        continue
                                    chunk_content = cand
                                    LOG.info("  (used %s field as content)", key)
                                    break

                        if not chunk_content:
                            result.status = ProviderStatus.OUTPUT
                            result.failure_type = "empty_response"
                            result.error_summary = f"Empty response (finish_reason={finish})."
                            retry_success = False
                            break # Empty response isn't retryable
                        
                        retry_success = True
                        break # Exited retry loop successfully

                    except Exception as e:
                        err_str = str(e).lower()
                        result.error_summary = str(e)
                        last_err = e
                        LOG.warning("LLM provider %s failed (attempt %d): %s", provider_name, retry_attempt + 1, e)
                        
                        is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
                        is_transient = "503" in err_str or "502" in err_str or "500" in err_str or "timeout" in err_str or "network" in err_str or "fetch" in err_str
                        
                        if is_rate_limit or is_transient:
                            if retry_attempt < max_provider_retries:
                                base_delay = p.get("retry_on_rate_limit_seconds", 5)
                                backoff = base_delay * (2 ** retry_attempt)
                                jitter = random.uniform(0, 0.2 * backoff)
                                wait_time = backoff + jitter
                                LOG.info("Sleeping %.2fs before retrying...", wait_time)
                                time.sleep(wait_time)
                                continue
                            else:
                                result.status = ProviderStatus.TRANSIENT
                                result.failure_type = "rate_limit" if is_rate_limit else "network_error"
                                break # Out of retries
                        else:
                            result.status = ProviderStatus.PERMANENT
                            result.failure_type = "authentication_failed" if ("auth" in err_str or "key" in err_str or "401" in err_str or "403" in err_str) else "unknown_error"
                            result.retryable = False
                            break # Permanent error

                if not retry_success:
                    break # The provider completely failed, skip to next provider in chain
                
                accumulated_content += chunk_content
                result.content = accumulated_content
                result.finish_reason = finish
                provider_success = True
                
                # Check for continuation
                if finish in ("length", "max_tokens", "truncated"):
                    LOG.info("LLM output truncated (finish_reason=%s), continuing generation (continuation %d/%d)...", finish, continuation_count + 1, MAX_CONTINUATIONS)
                    current_messages.append({"role": "assistant", "content": chunk_content})
                    current_messages.append({"role": "user", "content": "Continue from exactly where you stopped. Do not repeat previous content. Return the remaining content only. Finish the requested task completely."})
                    continuation_count += 1
                else:
                    result.status = ProviderStatus.SUCCESS
                    break # Done with generation!

            result.latency_ms = (time.time() - start_time) * 1000
            
            if provider_success and result.finish_reason not in ("length", "max_tokens"):
                result.status = ProviderStatus.SUCCESS
                return result
                
            if result.status == ProviderStatus.PERMANENT:
                continue # Try next provider
                
            if provider_idx is not None:
                return result

        # If we exhausted the chain without a specific provider targeting, return a generic failure
        return ProviderResult(
            status=ProviderStatus.PERMANENT,
            provider="chain",
            model="unknown",
            failure_type="all_providers_failed",
            error_summary=f"All providers failed. Last err: {last_err}",
            retryable=False
        )

    def chat_json_structured(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        provider_idx: int | None = None,
        attempt: int = 1,
    ) -> ProviderResult:
        result = self.chat_structured(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
            reasoning_effort=reasoning_effort,
            provider_idx=provider_idx,
            attempt=attempt
        )
        
        if result.status != ProviderStatus.SUCCESS:
            return result
            
        try:
            parsed = _parse_json(result.content)
            result.parsed = parsed
        except ValueError as e:
            result.status = ProviderStatus.OUTPUT
            result.failure_type = "malformed_json"
            result.error_summary = str(e)
            
        return result

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        # Fallback for old callers
        res = self.chat_structured(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            reasoning_effort=reasoning_effort
        )
        if res.status != ProviderStatus.SUCCESS:
            raise RuntimeError(f"Chat failed: {res.error_summary}")
        return res.content

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

    # Attempt 2.5: Extract from a markdown code block if present anywhere
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
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
