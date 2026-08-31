"""Script JSON generation — stable prompt plus dynamic weekly strategy."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.script")


def _load_strategy() -> dict:
    path = repo_root() / "data" / "dynamic_strategy.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _normalize_phrase(text: str) -> str:
    """Normalize narration for deterministic duplicate detection."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> list[str]:
    return _normalize_phrase(text).split()


def _repetition_issues(script: dict[str, Any]) -> list[str]:
    """Detect exact and distinctive phrase reuse across spoken units.

    We compare hook, scenes, and CTA because all of them are spoken in the final
    narration timeline. Short common words are intentionally ignored.
    """
    units: list[tuple[str, str]] = []
    hook = str(script.get("hook", "")).strip()
    if hook:
        units.append(("hook", hook))
    for i, scene in enumerate(script.get("scenes", [])):
        text = str(scene.get("narration", "")).strip()
        if text:
            units.append((f"scene_{i + 1}", text))
    cta = str(script.get("cta", "")).strip()
    if cta:
        units.append(("cta", cta))

    issues: list[str] = []
    normalized = [(name, _normalize_phrase(text), _words(text)) for name, text in units]

    # Exact duplicate spoken units.
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a_name, a_norm, _ = normalized[i]
            b_name, b_norm, _ = normalized[j]
            if a_norm and a_norm == b_norm:
                issues.append(f"Exact repetition: {a_name} == {b_name}")

    # Distinctive 5-word phrase reuse. This catches paraphrased/reused fragments
    # while avoiding false positives from normal 1-4 word overlaps.
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a_name, _, a_words = normalized[i]
            b_name, _, b_words = normalized[j]
            if len(a_words) < 5 or len(b_words) < 5:
                continue
            a_grams = {tuple(a_words[k:k + 5]) for k in range(len(a_words) - 4)}
            b_grams = {tuple(b_words[k:k + 5]) for k in range(len(b_words) - 4)}
            overlap = a_grams & b_grams
            if overlap:
                phrase = " ".join(next(iter(overlap)))
                issues.append(f"Repeated 5-word phrase between {a_name} and {b_name}: '{phrase}'")

    # Scene-opening repetition is a separate stylistic problem.
    openings: list[tuple[str, str]] = []
    for name, _, words in normalized:
        if name.startswith("scene_") and words:
            openings.append((name, words[0]))
    for i in range(1, len(openings)):
        if openings[i][1] == openings[i - 1][1]:
            issues.append(f"Consecutive scene openings repeat '{openings[i][1]}'")

    return issues


def _validate_script_structure(script: dict[str, Any]) -> None:
    issues = _repetition_issues(script)
    if issues:
        raise ValueError("Script repetition detected: " + "; ".join(issues))


def generate_script(llm: LLMRouter, *, slot: dict, topic: dict, research: dict, concept: dict | None = None, previous_qc: dict | None = None, previous_repetition: Any | None = None) -> dict[str, Any]:
    cfg = get_config()
    scfg = cfg.get_path("script", {}) or {}
    template = (repo_root() / "prompts" / "script.txt").read_text(encoding="utf-8")
    target_duration = int(cfg.get_path("video.target_duration_sec", 35))
    min_scenes = int(scfg.get("min_scenes", 5))
    max_scenes = int(scfg.get("max_scenes", 8))
    wps = float(scfg.get("words_per_second", 2.5))
    target_words = int(target_duration * wps)
    title_max = int(cfg.get_path("youtube.title_max_chars", 100))
    hashtags_count = int(cfg.get_path("output.hashtags_count", scfg.get("hashtags_count", 5)))

    angle = concept.get("chosen_angle", "") if concept else topic.get("angle", "")
    hook_type = concept.get("hook_type", "contradiction") if concept else "contradiction"
    curiosity_gap = concept.get("curiosity_gap", "") if concept else ""
    emotional_driver = concept.get("emotional_driver", "curiosity") if concept else "curiosity"
    strategy = _load_strategy()
    strategy_context = "\n[CURRENT LEARNED STRATEGY]\n" + json.dumps({
        "strategy_version": strategy.get("strategy_version", 0),
        "overall_direction": strategy.get("overall_direction", ""),
        "focused_themes": strategy.get("focused_themes", []),
        "avoid_themes": strategy.get("avoid_themes", []),
        "recommended_hooks": strategy.get("recommended_hooks", []),
        "recommended_emotions": strategy.get("recommended_emotions", []),
        "duration_recommendation": strategy.get("duration_recommendation", ""),
        "experiments": strategy.get("experiments", []),
    }, indent=2) + "\n"

    qc_feedback = ""
    if previous_qc and previous_qc.get("feedback"):
        qc_feedback = f"\n\n[PREVIOUS QC FEEDBACK TO FIX]\n{previous_qc['feedback']}\nIssues: {', '.join(previous_qc.get('issues', []))}"
        
    rep_feedback = ""
    if previous_repetition and not previous_repetition.passed:
        rep_feedback = f"\n\n[REPETITION FEEDBACK TO FIX]\nIssues: {', '.join(previous_repetition.all_issues)}"

    prompt = template.format(
        goal=goal_summary(), niche_title=slot.get("title", ""), voice_style=slot.get("voice_style", "neutral"), visual_style=slot.get("style", ""),
        topic_title=topic.get("title", ""), angle=angle, hook_type=hook_type, curiosity_gap=curiosity_gap,
        emotional_driver=emotional_driver, research_brief=json.dumps(research, indent=2), target_duration=target_duration,
        target_words=target_words, wpm=int(wps * 60), min_scenes=min_scenes, max_scenes=max_scenes,
        format_label=cfg.get_path("channel.format", "shorts"), hook_max_seconds=float(scfg.get("hook_max_seconds", 1.5)),
        title_max_chars=title_max, ai_disclosure=cfg.get_path("channel.ai_disclosure", "AI-assisted"),
        hashtags_count=hashtags_count, hashtags_count_minus_one=hashtags_count - 1,
    )
    prompt += strategy_context + qc_feedback + rep_feedback

    def _call(extra_feedback: str = "") -> dict[str, Any]:
        call_prompt = prompt + extra_feedback
        return llm.chat_json(
            [{"role": "user", "content": call_prompt}],
            max_tokens=int(scfg.get("max_tokens", 12000)),
            temperature=float(scfg.get("temperature", 0.7)),
            reasoning_effort=scfg.get("reasoning_effort", "low"),
        )

    script = _call()
    required = ["hook", "scenes", "title", "description", "hashtags", "thumbnail_prompt"]
    missing = [k for k in required if k not in script]
    if missing:
        raise ValueError(f"Script JSON missing fields: {missing}")
    if not isinstance(script["scenes"], list) or not script["scenes"]:
        raise ValueError("Script has no scenes")

    # LLMs can still occasionally repeat a hook or distinctive phrase despite
    # instructions. Give the model one targeted rewrite opportunity before QC.
    repetition_issues = _repetition_issues(script)
    if repetition_issues:
        LOG.warning("Script repetition detected; requesting targeted rewrite: %s", "; ".join(repetition_issues))
        feedback = (
            "\n\n[MANDATORY REPETITION REPAIR]\n"
            "The generated script contains repeated spoken material. Rewrite the script so every spoken unit is unique. "
            "The hook is spoken separately and MUST NOT appear or be paraphrased in any scene. "
            "Do not reuse any distinctive 5+ word phrase. Keep the same topic, factual claims, story promise, and CTA intent. "
            "Return the complete corrected JSON only.\n"
            "Detected issues:\n- " + "\n- ".join(repetition_issues)
        )
        script = _call(feedback)
        missing = [k for k in required if k not in script]
        if missing:
            raise ValueError(f"Rewritten script JSON missing fields: {missing}")
        if not isinstance(script["scenes"], list) or not script["scenes"]:
            raise ValueError("Rewritten script has no scenes")
        _validate_script_structure(script)
    else:
        _validate_script_structure(script)

    script["_learning"] = {"strategy_version": strategy.get("strategy_version", 0)}
    if concept:
        script["_concept"] = {
            "hook_type": hook_type,
            "angle": angle,
            "emotional_driver": emotional_driver,
            "concept_score": concept.get("concept_score", 0),
        }
    LOG.info("Script generated: %d scenes, title=%r, strategy_v=%s", len(script["scenes"]), script.get("title", "")[:80], strategy.get("strategy_version", 0))
    return script
