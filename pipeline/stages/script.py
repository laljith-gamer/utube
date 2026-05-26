"""Script JSON generation — every parameter from pipeline.yaml > script."""
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
) -> dict[str, Any]:
    cfg = get_config()
    scfg = cfg.get_path("script", {}) or {}
    template = (repo_root() / "prompts" / "script.txt").read_text(encoding="utf-8")

    target_duration = int(cfg.get_path("video.target_duration_sec", 35))
    num_scenes = int(cfg.get_path("video.num_scenes", 7))
    wps = float(scfg.get("words_per_second", 2.5))
    target_words = int(target_duration * wps)
    title_max = int(cfg.get_path("youtube.title_max_chars", 100))

    prompt = template.format(
        goal=goal_summary(),
        niche_title=slot.get("title", ""),
        voice_style=slot.get("voice_style", "neutral"),
        visual_style=slot.get("style", ""),
        topic_title=topic.get("title", ""),
        angle=topic.get("angle", ""),
        research_brief=json.dumps(research, indent=2),
        target_duration=target_duration,
        target_words=target_words,
        wpm=int(wps * 60),
        num_scenes=num_scenes,
        format_label=cfg.get_path("channel.format", "shorts"),
        hook_max_seconds=int(scfg.get("hook_max_seconds", 3)),
        title_max_chars=title_max,
        ai_disclosure=cfg.get_path("channel.ai_disclosure", "AI-assisted"),
    )

    script = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=int(scfg.get("max_tokens", 6000)),
        temperature=float(scfg.get("temperature", 0.8)),
        reasoning_effort=scfg.get("reasoning_effort", "high"),
    )

    required = ["hook", "scenes", "title", "description", "hashtags", "thumbnail_prompt"]
    missing = [k for k in required if k not in script]
    if missing:
        raise ValueError(f"Script JSON missing fields: {missing}")
    if not isinstance(script["scenes"], list) or not script["scenes"]:
        raise ValueError("Script has no scenes")

    LOG.info("Script generated: %d scenes, title=%r",
             len(script["scenes"]), script.get("title", "")[:80])
    return script
