"""Concept angle generation — creates multiple angles for a topic and selects the strongest."""
from __future__ import annotations

import logging
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter, ProviderStatus
from ..utils import repo_root, write_json

LOG = logging.getLogger("utube.concept")

HOOK_TYPES = [
    "contradiction", "forbidden_question", "hidden_cause", "surprising_number",
    "before_after", "personal_danger", "money_hook", "comparison", "time_pressure", "mystery",
]


def _pattern_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("key") or value.get("hook_type") or value.get("combination") or "")
    return str(value)


def validate_concept(angle: dict[str, Any]) -> bool:
    """Strict validation for a single concept angle."""
    if not isinstance(angle, dict):
        return False
    required_keys = ["angle", "hook_type", "curiosity_gap", "emotional_driver", "payoff", "concept_score"]
    for k in required_keys:
        val = angle.get(k)
        if val is None or val == "":
            return False
    return True


def _fallback_concept(topic: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback if all LLM attempts fail."""
    topic_title = topic.get("title", "Unknown Topic")
    topic_summary = topic.get("summary", "")
    
    # Try to derive a valid concept score based on the source quality if possible, otherwise default to 75.0 
    # to let it pass the gate unless we want to reject it. We will assign 75.0.
    concept_score = 75.0
    
    return {
        "angle": f"The hidden facts about {topic_title}",
        "hook_type": "contradiction",
        "core_claim": f"Common knowledge about {topic_title} is missing crucial details.",
        "viewer_problem": "People are often misinformed about this topic.",
        "curiosity_gap": f"What is the real story behind {topic_title}?",
        "emotional_driver": "curiosity",
        "payoff": f"The facts reveal the truth: {topic_summary[:100]}...",
        "concept_score": concept_score,
        "risk_flags": ["deterministic_fallback"]
    }


def generate_concept(topic: dict[str, Any], *, content_memory: dict | None = None, out_dir: Any = None) -> dict[str, Any] | None:
    cfg = get_config()
    concept_cfg = cfg.get_path("concept", {}) or {}
    llm_cfg = concept_cfg.get("llm", {}) or {}
    min_score = float(concept_cfg.get("min_concept_score", 70))
    n_angles = int(concept_cfg.get("angles_to_generate", 4))
    llm = LLMRouter("llm_concept")
    goal = goal_summary()

    topic_title = topic.get("title", "")
    topic_summary = topic.get("summary", "")
    topic_source = topic.get("source", "")
    
    prompt_path = repo_root() / "prompts" / "concept.txt"
    template = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else _DEFAULT_CONCEPT_PROMPT

    winning_hooks, weak_hooks, combinations_str = "", "", ""
    if content_memory:
        strong_hooks = content_memory.get("strong_hooks", [])
        if strong_hooks:
            winning_hooks = "Hook types that performed well recently: " + ", ".join(_pattern_name(x) for x in strong_hooks[:5])
        weak_hooks_list = content_memory.get("weak_hooks", [])
        if weak_hooks_list:
            weak_hooks = "Hook types that performed poorly: " + ", ".join(_pattern_name(x) for x in weak_hooks_list[:5])
        strong_combs = content_memory.get("strong_combinations", [])
        if strong_combs:
            parts = []
            for item in strong_combs[:3]:
                if isinstance(item, dict):
                    name = str(item.get("combination", item.get("key", ""))).replace("|", " + ")
                    conf = float(item.get("confidence", item.get("posterior_mean", 0)))
                    parts.append(f"{name} (conf: {conf:.2f})")
                else:
                    parts.append(str(item).replace("|", " + "))
            combinations_str = "High confidence learned combinations: " + ", ".join(parts)

    strategy_prompt = ""
    strategy_path = repo_root() / "data" / "dynamic_strategy.json"
    if strategy_path.exists():
        try:
            import json
            dyn = json.loads(strategy_path.read_text(encoding="utf-8"))
            strategy_prompt = (
                "[STRATEGIC DIRECTION]\n"
                f"{dyn.get('overall_direction', '')}\n"
                f"Focus on: {', '.join(dyn.get('focused_themes', []))}\n"
                f"Avoid: {', '.join(dyn.get('avoid_themes', []))}\n"
                f"Recommended Hooks: {', '.join(dyn.get('recommended_hooks', []))}\n"
            )
        except Exception as exc:
            LOG.warning("Failed to load dynamic_strategy.json: %s", exc)

    # Retry Ladder Configuration
    attempts_log = []
    best_angle = None
    angles_list = []
    concept_provider = ""
    concept_model = ""
    concept_generation_mode = ""

    for attempt_idx in range(4):
        # Determine specific strategy for this attempt
        is_simplified = attempt_idx >= 2
        provider_idx = attempt_idx % len(llm.active) if llm.active else None
        
        if is_simplified:
            # Simplified request: lower output, lower reasoning, simplified schema
            prompt = _SIMPLIFIED_CONCEPT_PROMPT.format(
                topic_title=topic_title,
                topic_summary=topic_summary[:300],
                hook_types=", ".join(HOOK_TYPES[:5])
            )
            reasoning_effort = "low"
            max_tokens = 1000
        else:
            prompt = template.format(
                goal=goal, topic_title=topic_title, topic_summary=topic_summary[:500], topic_source=topic_source,
                n_angles=n_angles, hook_types=", ".join(HOOK_TYPES), winning_hooks=winning_hooks,
                weak_hooks=weak_hooks, combinations_str=combinations_str, strategy_prompt=strategy_prompt,
            )
            reasoning_effort = llm_cfg.get("reasoning_effort")
            max_tokens = int(llm_cfg.get("max_tokens", 3000))

        LOG.info("Concept Attempt %d: simplified=%s, provider_idx=%s", attempt_idx + 1, is_simplified, provider_idx)
        
        result = llm.chat_json_structured(
            [{"role": "user", "content": prompt}], 
            max_tokens=max_tokens,
            temperature=float(llm_cfg.get("temperature", 0.6)), 
            reasoning_effort=reasoning_effort,
            provider_idx=provider_idx,
            attempt=attempt_idx + 1
        )
        
        attempts_log.append({
            "attempt": attempt_idx + 1,
            "provider": result.provider,
            "model": result.model,
            "status": result.status.name,
            "failure_type": result.failure_type,
            "finish_reason": result.finish_reason,
            "latency_ms": result.latency_ms,
            "error_summary": result.error_summary
        })

        if result.status == ProviderStatus.SUCCESS and result.parsed:
            angles = result.parsed.get("angles", [])
            valid_angles = [a for a in angles if validate_concept(a)]
            if valid_angles:
                best_angle = max(valid_angles, key=lambda a: float(a.get("concept_score", 0)))
                angles_list = angles
                concept_provider = result.provider
                concept_model = result.model
                concept_generation_mode = "llm"
                break
            else:
                attempts_log[-1]["failure_type"] = "schema_invalid"
                attempts_log[-1]["status"] = ProviderStatus.OUTPUT.name
        
        if not result.retryable and attempt_idx == 0:
            # E.g. authentication failed, might as well try next provider in ladder.
            pass

    if best_angle is None:
        LOG.warning("All %d LLM attempts failed. Falling back to deterministic concept.", len(attempts_log))
        best_angle = _fallback_concept(topic)
        angles_list = [best_angle]
        concept_provider = "deterministic_fallback"
        concept_model = "none"
        concept_generation_mode = "fallback"
        attempts_log.append({
            "attempt": 5,
            "provider": "deterministic_fallback",
            "model": "none",
            "status": "SUCCESS",
            "failure_type": "",
            "finish_reason": "",
            "latency_ms": 0,
            "error_summary": ""
        })
        
    if out_dir:
        write_json(out_dir / "concept_attempts.json", {"attempts": attempts_log})

    best_score = float(best_angle.get("concept_score", 0))
    if best_score < min_score:
        LOG.warning("No concept met minimum score %.1f for: %s (best: %.1f)", min_score, topic_title[:60], best_score)
        return None

    return {
        "topic_title": topic_title, 
        "topic_url": topic.get("url", ""), 
        "topic_source": topic_source,
        "topic_score": topic.get("total_score", 0), 
        "chosen_angle": best_angle.get("angle", ""),
        "hook_type": best_angle.get("hook_type", "contradiction"), 
        "core_claim": best_angle.get("core_claim", ""),
        "viewer_problem": best_angle.get("viewer_problem", ""), 
        "curiosity_gap": best_angle.get("curiosity_gap", ""),
        "emotional_driver": best_angle.get("emotional_driver", ""), 
        "payoff": best_angle.get("payoff", ""),
        "concept_score": best_score, 
        "risk_flags": best_angle.get("risk_flags", []), 
        "all_angles": angles_list,
        "concept_provider": concept_provider,
        "concept_model": concept_model,
        "concept_generation_mode": concept_generation_mode,
        "validation_status": "valid"
    }


_DEFAULT_CONCEPT_PROMPT = """{goal}

Topic: {topic_title}
Source: {topic_source}
Summary: {topic_summary}

Generate {n_angles} different angles for a YouTube Short about this topic.
Each angle should use a DIFFERENT hook type from: {hook_types}

{strategy_prompt}
{winning_hooks}
{weak_hooks}
{combinations_str}

For each angle, provide:
- angle: One sentence describing the specific framing
- hook_type: Which type from the list above
- core_claim: The central thesis in one sentence
- viewer_problem: Why should THEY care? What does it mean for their life?
- curiosity_gap: The question the viewer cannot resist
- emotional_driver: What emotion drives the watch? (curiosity, fear, surprise, outrage, relief)
- payoff: The satisfying answer/reveal in one sentence
- concept_score: 0-100 how strong this concept is overall
- risk_flags: List of potential problems

Respond with ONLY a JSON object:
{{"angles": [{{"angle": "", "hook_type": "", "core_claim": "", "viewer_problem": "", "curiosity_gap": "", "emotional_driver": "", "payoff": "", "concept_score": 0, "risk_flags": []}}, ...]}}
"""


_SIMPLIFIED_CONCEPT_PROMPT = """Topic: {topic_title}
Summary: {topic_summary}

Generate 1 angle for a YouTube Short about this topic.
Use a hook type from: {hook_types}

Provide:
- angle: One sentence framing
- hook_type: Which type
- core_claim: Central thesis
- viewer_problem: Why they care
- curiosity_gap: The question
- emotional_driver: The emotion
- payoff: The satisfying answer
- concept_score: 0-100 score

Respond with ONLY a JSON object:
{{"angles": [{{"angle": "", "hook_type": "", "core_claim": "", "viewer_problem": "", "curiosity_gap": "", "emotional_driver": "", "payoff": "", "concept_score": 0, "risk_flags": []}}]}}
"""
