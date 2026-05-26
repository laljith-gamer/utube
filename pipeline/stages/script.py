"""Generate the structured script JSON via gpt-oss-120b."""
from __future__ import annotations

import json
import logging
from typing import Any

from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.script")


def generate_script(
    llm: LLMRouter,
    *,
    slot: dict,
    topic: dict,
    research: dict,
    target_duration: int,
    num_scenes: int,
) -> dict[str, Any]:
    template = (repo_root() / "prompts" / "script.txt").read_text(encoding="utf-8")
    target_words = int(target_duration * 2.5)  # ~150 wpm

    prompt = template.format(
        niche_title=slot.get("title", ""),
        voice_style=slot.get("voice_style", "neutral"),
        visual_style=slot.get("style", ""),
        topic_title=topic.get("title", ""),
        angle=topic.get("angle", ""),
        research_brief=json.dumps(research, indent=2),
        target_duration=target_duration,
        target_words=target_words,
        num_scenes=num_scenes,
    )

    script = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=6000,
        temperature=0.8,
        reasoning_effort="high",
    )

    # Validate shape
    required = ["hook", "scenes", "title", "description", "hashtags", "thumbnail_prompt"]
    missing = [k for k in required if k not in script]
    if missing:
        raise ValueError(f"Script JSON missing fields: {missing}")
    if not isinstance(script["scenes"], list) or not script["scenes"]:
        raise ValueError("Script has no scenes")

    LOG.info(
        "Script generated: %d scenes, title=%r",
        len(script["scenes"]),
        script.get("title", "")[:80],
    )
    return script
