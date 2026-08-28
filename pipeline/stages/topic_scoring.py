"""Topic scoring engine — multi-dimensional candidate evaluation.

Replaces random theme selection with data-driven scoring. Each candidate
is evaluated on 10 weighted dimensions, filtered by hard rejections,
and selected via exploitation/exploration balance.
"""
from __future__ import annotations

import logging
import random
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.topic_scoring")


def score_candidates(
    candidates: list[dict[str, Any]],
    *,
    content_memory: dict | None = None,
    recent_hashes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Score every candidate on multiple dimensions. Returns sorted list (best first).

    Each candidate gets:
      - Heuristic scores (freshness, source_quality, specificity, novelty)
      - LLM-assisted scores (audience_fit, curiosity_gap, story_potential, visual_potential, shareability)
      - Weighted total score (0-100)
    """
    if not candidates:
        LOG.warning("No candidates to score")
        return []

    cfg = get_config()
    scoring_cfg = cfg.get_path("topic_scoring", {}) or {}
    weights = scoring_cfg.get("weights", {}) or {}
    recent_hashes = recent_hashes or []

    # ── Phase 1: Heuristic scoring ──────────────────────────────────────
    for c in candidates:
        c["scores"] = {}

        # Freshness: high normalized_hotness = fresh & trending
        hotness = c.get("normalized_hotness", 50)
        c["scores"]["freshness"] = min(100, hotness)

        # Source quality: some sources are more reliable
        source_quality_map = {
            "hackernews": 80, "github_trending": 65, "devto": 55,
            "wikipedia_otd": 70,
        }
        src = c.get("source", "")
        base_src = src.split(":")[0] if ":" in src else src
        if base_src.startswith("reddit"):
            base_src = "reddit"
        c["scores"]["source_quality"] = source_quality_map.get(base_src, 60)
        if base_src.startswith("rss"):
            c["scores"]["source_quality"] = 72

        # Specificity: titles with numbers, names, dates score higher
        title = c.get("title", "")
        spec_score = 40
        import re
        if re.search(r"\d", title):
            spec_score += 20
        if re.search(r"\$[\d,]+|billion|million|trillion|percent|%", title, re.I):
            spec_score += 15
        if len(title.split()) >= 5:
            spec_score += 10
        if any(w in title.lower() for w in ["why", "how", "secret", "hidden", "real reason", "truth", "myth"]):
            spec_score += 15
        c["scores"]["specificity"] = min(100, spec_score)

        # Novelty: not in recent hashes
        content_hash = c.get("content_hash", "")
        if any(content_hash.startswith(rh[:20]) for rh in recent_hashes if rh):
            c["scores"]["novelty"] = 0
        else:
            c["scores"]["novelty"] = 75  # Default novel

        # Evergreen: historical / scientific topics tend to be evergreen
        evergreen_signals = ["history", "invented", "origin", "discovered", "first", "oldest", "science"]
        if any(s in title.lower() for s in evergreen_signals):
            c["scores"]["evergreen_value"] = 80
        else:
            c["scores"]["evergreen_value"] = 50

    # ── Phase 2: LLM-assisted batch scoring ─────────────────────────────
    batch_size = scoring_cfg.get("llm_batch_size", 10)
    # Sort by heuristic average for pre-filter, take top N for LLM
    for c in candidates:
        heuristic_avg = sum(c["scores"].values()) / max(1, len(c["scores"]))
        c["_heuristic_avg"] = heuristic_avg

    candidates.sort(key=lambda c: c["_heuristic_avg"], reverse=True)
    top_batch = candidates[:batch_size]

    try:
        llm_scores = _llm_batch_score(top_batch, scoring_cfg)
        for c, scores in zip(top_batch, llm_scores):
            c["scores"].update(scores)
    except Exception as e:  # noqa: BLE001
        LOG.warning("LLM batch scoring failed (%s), using heuristics only", e)
        for c in top_batch:
            c["scores"].update({
                "audience_fit": 50, "curiosity_gap": 50,
                "story_potential": 50, "visual_potential": 50,
                "shareability": 50,
            })

    # For candidates not sent to LLM, fill defaults
    for c in candidates[batch_size:]:
        c["scores"].update({
            "audience_fit": 40, "curiosity_gap": 40,
            "story_potential": 40, "visual_potential": 40,
            "shareability": 40,
        })

    # ── Phase 3: Weighted total ─────────────────────────────────────────
    default_weights = {
        "audience_fit": 0.20, "curiosity_gap": 0.18, "story_potential": 0.15,
        "visual_potential": 0.12, "freshness": 0.10, "specificity": 0.08,
        "shareability": 0.07, "novelty": 0.05, "source_quality": 0.03,
        "evergreen_value": 0.02,
    }
    w = {**default_weights, **weights}
    # Normalize weights to sum to 1
    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}

    for c in candidates:
        total = 0.0
        for dim, weight in w.items():
            total += c["scores"].get(dim, 50) * weight
        c["total_score"] = round(total, 1)

    candidates.sort(key=lambda c: c["total_score"], reverse=True)

    # Clean up temp key
    for c in candidates:
        c.pop("_heuristic_avg", None)

    LOG.info(
        "Scored %d candidates. Top: %s (%.1f), Bottom: %s (%.1f)",
        len(candidates),
        candidates[0].get("title", "")[:50] if candidates else "?",
        candidates[0].get("total_score", 0) if candidates else 0,
        candidates[-1].get("title", "")[:50] if candidates else "?",
        candidates[-1].get("total_score", 0) if candidates else 0,
    )
    return candidates


def select_best(
    scored: list[dict[str, Any]],
    *,
    min_score: float | None = None,
    exploration_ratio: float | None = None,
) -> dict[str, Any] | None:
    """Select the best candidate, with occasional exploration.

    Returns None if no candidate meets the minimum quality bar —
    this means "no good topic today".
    """
    cfg = get_config()
    scoring_cfg = cfg.get_path("topic_scoring", {}) or {}
    if min_score is None:
        min_score = float(scoring_cfg.get("min_topic_score", 72))
    if exploration_ratio is None:
        exploration_ratio = float(scoring_cfg.get("exploration_ratio", 0.20))

    # Filter by minimum score
    qualified = [c for c in scored if c.get("total_score", 0) >= min_score]
    if not qualified:
        LOG.warning(
            "No candidate met minimum score %.1f. Best was: %s (%.1f)",
            min_score,
            scored[0].get("title", "?")[:60] if scored else "none",
            scored[0].get("total_score", 0) if scored else 0,
        )
        return None

    # Exploitation vs exploration
    if len(qualified) > 1 and random.random() < exploration_ratio:
        # Pick from rank 2-5 for exploration
        explore_pool = qualified[1:min(5, len(qualified))]
        chosen = random.choice(explore_pool)
        LOG.info("Exploration pick: %s (%.1f) instead of top %s (%.1f)",
                 chosen.get("title", "")[:50], chosen.get("total_score", 0),
                 qualified[0].get("title", "")[:50], qualified[0].get("total_score", 0))
        return chosen

    LOG.info("Top pick: %s (%.1f)", qualified[0].get("title", "")[:50], qualified[0].get("total_score", 0))
    return qualified[0]


def _llm_batch_score(candidates: list[dict], scoring_cfg: dict) -> list[dict]:
    """Use LLM to score audience_fit, curiosity_gap, story_potential, visual_potential, shareability."""
    llm = LLMRouter("llm_research")
    llm_cfg = scoring_cfg.get("llm", {}) or {}

    goal = goal_summary()

    # Build candidate summaries for the prompt
    candidate_lines = []
    for i, c in enumerate(candidates):
        line = f"[{i}] {c.get('title', '')} (source: {c.get('source', '')}, hotness: {c.get('normalized_hotness', 0)})"
        if c.get("summary"):
            line += f"\n    Summary: {c['summary'][:200]}"
        candidate_lines.append(line)

    prompt_path = repo_root() / "prompts" / "topic_scoring.txt"
    if prompt_path.exists():
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_SCORING_PROMPT

    prompt = template.format(
        goal=goal,
        n_candidates=len(candidates),
        candidates="\n".join(candidate_lines),
    )

    result = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=int(llm_cfg.get("max_tokens", 3000)),
        temperature=float(llm_cfg.get("temperature", 0.3)),
        reasoning_effort=llm_cfg.get("reasoning_effort"),
    )

    # Parse — expect {"scores": [{"audience_fit": N, "curiosity_gap": N, ...}, ...]}
    raw_scores = result.get("scores", [])
    if not isinstance(raw_scores, list):
        raw_scores = []

    out: list[dict] = []
    for i in range(len(candidates)):
        if i < len(raw_scores) and isinstance(raw_scores[i], dict):
            s = raw_scores[i]
            out.append({
                "audience_fit": _clamp(s.get("audience_fit", 50)),
                "curiosity_gap": _clamp(s.get("curiosity_gap", 50)),
                "story_potential": _clamp(s.get("story_potential", 50)),
                "visual_potential": _clamp(s.get("visual_potential", 50)),
                "shareability": _clamp(s.get("shareability", 50)),
            })
        else:
            out.append({
                "audience_fit": 50, "curiosity_gap": 50,
                "story_potential": 50, "visual_potential": 50,
                "shareability": 50,
            })

    return out


def _clamp(v: Any, lo: int = 0, hi: int = 100) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return 50


_DEFAULT_SCORING_PROMPT = """{goal}

Score each candidate for a YouTube Short on this channel. For each, rate 0-100:
- audience_fit: Would the same viewer who watched yesterday want this?
- curiosity_gap: How strong is the "I need to know" pull?
- story_potential: Can this be told as a 30-second narrative arc?
- visual_potential: Are there concrete, filmable visuals?
- shareability: Would a viewer send this to a friend?

Candidates:
{candidates}

Respond with ONLY a JSON object:
{{"scores": [{{"audience_fit": N, "curiosity_gap": N, "story_potential": N, "visual_potential": N, "shareability": N}}, ...]}}
"""
