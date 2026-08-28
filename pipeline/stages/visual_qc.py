"""Visual Quality Control — tracking and fallback enforcement.

Evaluates the generated visuals for the entire video.
Rejects the video if too many scenes fall back to motion filler.
Logs metadata for performance tracking.
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import get_config

LOG = logging.getLogger("utube.visual_qc")


def evaluate_visuals(visuals: list[dict[str, Any]]) -> dict[str, Any]:
    """Run QC evaluation on generated visuals.

    Returns:
    {
        "passed": bool,
        "filler_count": int,
        "total_scenes": int,
        "sources": {"svd": int, "stock": int, "image_motion": int, "filler": int},
        "issues": [str]
    }
    """
    cfg = get_config()
    qc_cfg = cfg.get_path("visual_qc", {}) or {}
    max_filler = int(qc_cfg.get("max_filler_scenes", 2))

    total_scenes = len(visuals)
    filler_count = 0
    sources = {"svd": 0, "stock": 0, "image_motion": 0, "filler": 0}

    for scene in visuals:
        source = scene.get("source", "filler")
        sources[source] = sources.get(source, 0) + 1
        if source == "filler":
            filler_count += 1

    passed = filler_count <= max_filler
    issues = []
    
    if not passed:
        issues.append(f"Too many filler scenes: {filler_count} > {max_filler}")
        LOG.warning("Visual QC FAILED: %d filler scenes (max allowed %d)", filler_count, max_filler)
    else:
        LOG.info("Visual QC PASSED: %d filler scenes (out of %d total)", filler_count, total_scenes)

    return {
        "passed": passed,
        "filler_count": filler_count,
        "total_scenes": total_scenes,
        "sources": sources,
        "issues": issues,
    }
