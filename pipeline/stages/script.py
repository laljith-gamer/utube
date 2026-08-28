"""Script JSON generation — concept-driven, with configurable parameters.

Accepts a concept object from the concept stage and uses its angle, hook_type,
curiosity_gap, and emotional_driver to generate a targeted script.
Every parameter comes from pipeline.yaml > script or quality.yaml.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.script")


def generate_script(
    llm: LLMRouter,
    *,
    slot: dict,
    topic: dict,
    research: dict,
    concept: dict | None = None,
) -> dict[str, Any]:
    """Generate a script JSON from the research brief and concept.

    If concept is provided, uses its hook_type, angle, curiosity_gap, and
    emotional_driver for targeted generation. Otherwise falls back to
    topic-level angle.
    """
    cfg = get_config()
    scfg = cfg.get_path("script", {}) or {}
    template = (repo_root() / "prompts" / "script.txt").read_text(encoding="utf-8")

    target_duration = int(cfg.get_path("video.target_duration_sec", 35))
    min_scenes = int(scfg.get("min_scenes", 5))
    max_scenes = int(scfg.get("max_scenes", 8))
    wps = float(scfg.get("words_per_second", 2.5))
    target_words = int(target_duration * wps)
    title_max = int(cfg.get_path("youtube.title_max_chars", 100))

    # Use quality.yaml hashtags_count, fallback to pipeline.yaml
    hashtags_count = int(cfg.get_path("output.hashtags_count", scfg.get("hashtags_count", 5)))

    # Extract concept fields with fallbacks
    angle = ""
    hook_type = "contradiction"
    curiosity_gap = ""
    emotional_driver = "curiosity"
    if concept:
        angle = concept.get("chosen_angle", "")
        hook_type = concept.get("hook_type", "contradiction")
        curiosity_gap = concept.get("curiosity_gap", "")
        emotional_driver = concept.get("emotional_driver", "curiosity")
    if not angle:
        angle = topic.get("angle", "")

    prompt = template.format(
        goal=goal_summary(),
        niche_title=slot.get("title", ""),
        voice_style=slot.get("voice_style", "neutral"),
        visual_style=slot.get("style", ""),
        topic_title=topic.get("title", ""),
        angle=angle,
        hook_type=hook_type,
        curiosity_gap=curiosity_gap,
        emotional_driver=emotional_driver,
        research_brief=json.dumps(research, indent=2),
        target_duration=target_duration,
        target_words=target_words,
        wpm=int(wps * 60),
        min_scenes=min_scenes,
        max_scenes=max_scenes,
        format_label=cfg.get_path("channel.format", "shorts"),
        hook_max_seconds=float(scfg.get("hook_max_seconds", 1.5)),
        title_max_chars=title_max,
        ai_disclosure=cfg.get_path("channel.ai_disclosure", "AI-assisted"),
        hashtags_count=hashtags_count,
        hashtags_count_minus_one=hashtags_count - 1,
    )

    script = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=int(scfg.get("max_tokens", 12000)),
        temperature=float(scfg.get("temperature", 0.7)),
        reasoning_effort=scfg.get("reasoning_effort", "low"),
    )

    required = ["hook", "scenes", "title", "description", "hashtags", "thumbnail_prompt"]
    missing = [k for k in required if k not in script]
    if missing:
        raise ValueError(f"Script JSON missing fields: {missing}")
    if not isinstance(script["scenes"], list) or not script["scenes"]:
        raise ValueError("Script has no scenes")

    # Attach concept metadata for downstream tracking
    if concept:
        script["_concept"] = {
            "hook_type": hook_type,
            "angle": angle,
            "emotional_driver": emotional_driver,
            "concept_score": concept.get("concept_score", 0),
        }

    LOG.info("Script generated: %d scenes, title=%r",
             len(script["scenes"]), script.get("title", "")[:80])
    return script
