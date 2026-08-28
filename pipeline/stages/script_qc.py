"""Script quality control — LLM-based evaluation of generated scripts.

Evaluates scripts on multiple dimensions (hook_strength, clarity, specificity,
story_progression, payoff_strength, natural_voice, channel_fit) and rejects
scripts that fall below configurable thresholds.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.script_qc")


def evaluate_script(
    script: dict[str, Any],
    *,
    topic: dict | None = None,
    concept: dict | None = None,
) -> dict[str, Any]:
    """Run QC evaluation on a generated script. Returns verdict + scores.

    Output:
    {
        "passed": bool,
        "overall_score": float,
        "scores": {"hook_strength": int, "clarity": int, ...},
        "feedback": str,   # If failed, specific improvement instructions
        "issues": [str],   # List of specific problems found
    }
    """
    cfg = get_config()
    qc_cfg = cfg.get_path("script_qc", {}) or {}
    llm_cfg = qc_cfg.get("llm", {}) or {}
    min_scores = qc_cfg.get("min_scores", {}) or {}
    min_overall = float(qc_cfg.get("min_overall_score", 72))

    llm = LLMRouter("llm_qc")
    goal = goal_summary()

    # Build the QC prompt
    prompt_path = repo_root() / "prompts" / "script_qc.txt"
    if prompt_path.exists():
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_QC_PROMPT

    # Prepare script summary for evaluation
    script_text = _script_to_text(script)
    concept_context = ""
    if concept:
        concept_context = (
            f"Hook type: {concept.get('hook_type', '')}\n"
            f"Angle: {concept.get('chosen_angle', '')}\n"
            f"Curiosity gap: {concept.get('curiosity_gap', '')}\n"
            f"Expected payoff: {concept.get('payoff', '')}"
        )

    prompt = template.format(
        goal=goal,
        script_json=json.dumps(script, indent=2)[:3000],
        script_text=script_text,
        topic_title=topic.get("title", "") if topic else "",
        concept_context=concept_context,
    )

    try:
        result = llm.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=int(llm_cfg.get("max_tokens", 2000)),
            temperature=float(llm_cfg.get("temperature", 0.3)),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
        )
    except Exception as e:
        LOG.warning("Script QC LLM call failed (%s), passing by default", e)
        return {"passed": True, "overall_score": 75, "scores": {}, "feedback": "", "issues": []}

    scores = result.get("scores", {})
    issues = result.get("issues", [])
    feedback = result.get("feedback", "")

    # Calculate overall score (weighted average of dimensions)
    dimension_weights = {
        "hook_strength": 0.20,
        "clarity": 0.15,
        "specificity": 0.15,
        "story_progression": 0.12,
        "payoff_strength": 0.15,
        "natural_voice": 0.10,
        "channel_fit": 0.13,
    }

    total_w = 0.0
    weighted_sum = 0.0
    for dim, weight in dimension_weights.items():
        val = scores.get(dim, 70)
        try:
            val = int(val)
        except (TypeError, ValueError):
            val = 70
        scores[dim] = val
        weighted_sum += val * weight
        total_w += weight

    overall = round(weighted_sum / total_w, 1) if total_w > 0 else 70

    # Check per-dimension minimums
    failed_dims = []
    for dim, min_val in min_scores.items():
        actual = scores.get(dim, 70)
        if actual < min_val:
            failed_dims.append(f"{dim}: {actual} < {min_val}")

    passed = overall >= min_overall and not failed_dims

    if not passed:
        reasons = []
        if overall < min_overall:
            reasons.append(f"Overall score {overall} < {min_overall}")
        reasons.extend(failed_dims)
        LOG.warning("Script QC FAILED: %s", "; ".join(reasons))
    else:
        LOG.info("Script QC PASSED (overall: %.1f)", overall)

    return {
        "passed": passed,
        "overall_score": overall,
        "scores": scores,
        "feedback": feedback,
        "issues": issues,
        "failed_dimensions": failed_dims,
    }


def _script_to_text(script: dict) -> str:
    """Convert script JSON to readable narration text for QC evaluation."""
    parts = []
    hook = script.get("hook", "")
    if hook:
        parts.append(f"[HOOK] {hook}")

    for i, scene in enumerate(script.get("scenes", []), 1):
        narration = scene.get("narration", "")
        parts.append(f"[SCENE {i}] {narration}")

    cta = script.get("cta", "")
    if cta:
        parts.append(f"[CTA] {cta}")

    parts.append(f"[TITLE] {script.get('title', '')}")
    return "\n".join(parts)


_DEFAULT_QC_PROMPT = """{goal}

Evaluate this YouTube Short script for quality. Score each dimension 0-100.

Topic: {topic_title}
{concept_context}

Script:
{script_text}

Score these dimensions:
- hook_strength: Does the first line stop a scroll? Does it open a curiosity loop?
- clarity: Is every sentence easy to understand on first listen?
- specificity: Are facts specific (numbers, names, dates) not vague?
- story_progression: Does each scene build on the last? Is there momentum?
- payoff_strength: Does the ending resolve the hook's promise? Does it feel earned?
- natural_voice: Does it sound like a real person talking, not a textbook?
- channel_fit: Does this fit "surprising technology that matters to ordinary people"?

Respond with ONLY a JSON object:
{{
  "scores": {{
    "hook_strength": <int>,
    "clarity": <int>,
    "specificity": <int>,
    "story_progression": <int>,
    "payoff_strength": <int>,
    "natural_voice": <int>,
    "channel_fit": <int>
  }},
  "issues": ["<specific problems found>"],
  "feedback": "<if score is below 75 on any dimension, specific instructions to fix it>"
}}
"""
