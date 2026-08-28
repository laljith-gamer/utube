"""Concept angle generation — creates multiple angles for a topic and selects the strongest."""
from __future__ import annotations

import logging
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

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


def generate_concept(topic: dict[str, Any], *, content_memory: dict | None = None) -> dict[str, Any] | None:
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

    prompt = template.format(
        goal=goal, topic_title=topic_title, topic_summary=topic_summary[:500], topic_source=topic_source,
        n_angles=n_angles, hook_types=", ".join(HOOK_TYPES), winning_hooks=winning_hooks,
        weak_hooks=weak_hooks, combinations_str=combinations_str, strategy_prompt=strategy_prompt,
    )
    try:
        result = llm.chat_json(
            [{"role": "user", "content": prompt}], max_tokens=int(llm_cfg.get("max_tokens", 3000)),
            temperature=float(llm_cfg.get("temperature", 0.6)), reasoning_effort=llm_cfg.get("reasoning_effort"),
        )
    except Exception as exc:
        LOG.error("Concept generation failed: %s", exc)
        return None

    angles = result.get("angles", [])
    if not angles:
        return None
    best = max(angles, key=lambda angle: float(angle.get("concept_score", 0)))
    best_score = float(best.get("concept_score", 0))
    if best_score < min_score:
        LOG.warning("No concept met minimum score %.1f for: %s (best: %.1f)", min_score, topic_title[:60], best_score)
        return None

    return {
        "topic_title": topic_title, "topic_url": topic.get("url", ""), "topic_source": topic_source,
        "topic_score": topic.get("total_score", 0), "chosen_angle": best.get("angle", ""),
        "hook_type": best.get("hook_type", "contradiction"), "core_claim": best.get("core_claim", ""),
        "viewer_problem": best.get("viewer_problem", ""), "curiosity_gap": best.get("curiosity_gap", ""),
        "emotional_driver": best.get("emotional_driver", ""), "payoff": best.get("payoff", ""),
        "concept_score": best_score, "risk_flags": best.get("risk_flags", []), "all_angles": angles,
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
