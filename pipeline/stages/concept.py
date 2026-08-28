"""Concept angle generation — creates multiple angles for a topic and selects the strongest.

Sits between topic scoring and research in the pipeline. For a scored topic,
generates 3-5 possible framing angles (each with a different hook type),
scores them, and returns the best concept for scriptwriting.
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.concept")

# Hook archetypes that map to proven Shorts patterns
HOOK_TYPES = [
    "contradiction",        # "Most people believe X, but actually Y"
    "forbidden_question",   # "Why does no one talk about X?"
    "hidden_cause",         # "The REAL reason X happens"
    "surprising_number",    # "73% of phones do X without you knowing"
    "before_after",         # "X went from impossible to normal in 2 years"
    "personal_danger",      # "Your phone is doing X right now"
    "money_hook",           # "This costs you $X/year and you don't know it"
    "comparison",           # "X vs Y — one of them is lying"
    "time_pressure",        # "In 5 years, X will be..."
    "mystery",              # "Nobody knows why X works"
]


def generate_concept(
    topic: dict[str, Any],
    *,
    content_memory: dict | None = None,
) -> dict[str, Any] | None:
    """Generate and score concept angles for a topic. Returns the best concept.

    Returns None if no concept meets the minimum quality bar.

    Output structure:
    {
        "topic_title": str,
        "chosen_angle": str,
        "hook_type": str,
        "core_claim": str,
        "viewer_problem": str,
        "curiosity_gap": str,
        "emotional_driver": str,
        "payoff": str,
        "concept_score": float,
        "all_angles": [...]  # for debugging
    }
    """
    cfg = get_config()
    concept_cfg = cfg.get_path("concept", {}) or {}
    llm_cfg = concept_cfg.get("llm", {}) or {}
    min_score = float(concept_cfg.get("min_concept_score", 70))
    n_angles = int(concept_cfg.get("angles_to_generate", 4))

    llm = LLMRouter("llm_concept")
    goal = goal_summary()

    # Build context from topic
    topic_title = topic.get("title", "")
    topic_summary = topic.get("summary", "")
    topic_source = topic.get("source", "")

    # Load prompt template
    prompt_path = repo_root() / "prompts" / "concept.txt"
    if prompt_path.exists():
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_CONCEPT_PROMPT

    # Winning patterns from content memory
    winning_hooks = ""
    weak_hooks = ""
    combinations_str = ""
    if content_memory:
        strong_hooks = content_memory.get("strong_hooks", [])
        if strong_hooks:
            winning_hooks = "Hook types that performed well recently: " + ", ".join(strong_hooks[:5])
            
        weak_hooks_list = content_memory.get("weak_hooks", [])
        if weak_hooks_list:
            weak_hooks = "Hook types that performed poorly: " + ", ".join(weak_hooks_list[:5])
            
        strong_combs = content_memory.get("strong_combinations", [])
        if strong_combs:
            combs = [f"{c['combination'].replace('|', ' + ')} (conf: {c['confidence']:.2f})" for c in strong_combs[:3]]
            combinations_str = "High confidence combinations (Hook + Emotion): " + ", ".join(combs)

    strategy_path = repo_root() / "data" / "dynamic_strategy.json"
    strategy_prompt = ""
    if strategy_path.exists():
        try:
            import json
            with open(strategy_path, "r", encoding="utf-8") as f:
                dyn_strat = json.load(f)
                direction = dyn_strat.get('overall_direction', '')
                focus = ", ".join(dyn_strat.get('focused_themes', []))
                avoid = ", ".join(dyn_strat.get('avoid_themes', []))
                rec_hooks = ", ".join(dyn_strat.get('recommended_hooks', []))
                strategy_prompt = f"[STRATEGIC DIRECTION]\n{direction}\nFocus on: {focus}\nAvoid: {avoid}\nRecommended Hooks: {rec_hooks}\n"
        except Exception as e:
            LOG.warning("Failed to load dynamic_strategy.json: %s", e)

    prompt = template.format(
        goal=goal,
        topic_title=topic_title,
        topic_summary=topic_summary[:500],
        topic_source=topic_source,
        n_angles=n_angles,
        hook_types=", ".join(HOOK_TYPES),
        winning_hooks=winning_hooks,
        weak_hooks=weak_hooks,
        combinations_str=combinations_str,
        strategy_prompt=strategy_prompt,
    )

    try:
        result = llm.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=int(llm_cfg.get("max_tokens", 3000)),
            temperature=float(llm_cfg.get("temperature", 0.6)),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
        )
    except Exception as e:
        LOG.error("Concept generation failed: %s", e)
        return None

    angles = result.get("angles", [])
    if not angles:
        LOG.warning("LLM returned no concept angles for: %s", topic_title[:60])
        return None

    # Find best angle by score
    best = None
    best_score = -1
    for angle in angles:
        score = float(angle.get("concept_score", 0))
        if score > best_score:
            best_score = score
            best = angle

    if best is None or best_score < min_score:
        LOG.warning(
            "No concept met minimum score %.1f for: %s (best: %.1f)",
            min_score, topic_title[:60], best_score,
        )
        return None

    concept = {
        "topic_title": topic_title,
        "topic_url": topic.get("url", ""),
        "topic_source": topic_source,
        "topic_score": topic.get("total_score", 0),
        "chosen_angle": best.get("angle", ""),
        "hook_type": best.get("hook_type", "contradiction"),
        "core_claim": best.get("core_claim", ""),
        "viewer_problem": best.get("viewer_problem", ""),
        "curiosity_gap": best.get("curiosity_gap", ""),
        "emotional_driver": best.get("emotional_driver", ""),
        "payoff": best.get("payoff", ""),
        "concept_score": best_score,
        "risk_flags": best.get("risk_flags", []),
        "all_angles": angles,  # Keep for debugging
    }

    LOG.info(
        "Concept selected: [%s] %s (score: %.1f)",
        concept["hook_type"], concept["chosen_angle"][:60], best_score,
    )
    return concept


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
- risk_flags: List of potential problems (e.g. "too abstract", "needs code to explain")

Respond with ONLY a JSON object:
{{"angles": [{{"angle": "", "hook_type": "", "core_claim": "", "viewer_problem": "", "curiosity_gap": "", "emotional_driver": "", "payoff": "", "concept_score": 0, "risk_flags": []}}, ...]}}
"""
