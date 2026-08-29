"""Topic scoring engine — quality, strategy, memory, and exploration."""
from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.topic_scoring")


def _topic_family(title: str, summary: str = "") -> str:
    text = f"{title} {summary}".lower()
    rules = {
        "ai scams": ("voice cloning", "deepfake", "impersonat", "ai scam", "ai fraud"),
        "ai": ("artificial intelligence", " ai ", "chatgpt", "agent", "machine learning"),
        "cybersecurity": ("hack", "malware", "phishing", "password", "cyber", "security"),
        "privacy": ("privacy", "tracking", "location", "spy", "permission", "data"),
        "consumer technology": ("phone", "iphone", "android", "laptop", "browser", "wifi"),
        "internet mechanics": ("internet", "dns", "server", "cookie", "cloud"),
    }
    for family, needles in rules.items():
        if any(n in text for n in needles):
            return family
    return "other"


def _pattern_lookup(memory: dict | None, category: str, key: str) -> dict | None:
    if not memory or not key:
        return None
    patterns = memory.get("winning_patterns", {}).get(category, {})
    weak = memory.get("weak_patterns", {}).get(category, {})
    return patterns.get(key) or weak.get(key)


def _memory_adjustment(candidate: dict, memory: dict | None) -> tuple[float, dict]:
    if not memory:
        return 0.0, {}
    family = _topic_family(candidate.get("title", ""), candidate.get("summary", ""))
    p = _pattern_lookup(memory, "topic_families", family)
    if not p:
        return 0.0, {"topic_family": family, "matches": []}
    strength = float(p.get("evidence_strength", 0.0))
    mean = float(p.get("posterior_mean", 0.5))
    delta = (mean - 0.5) * 12.0 * strength
    evidence = {"topic_family": family, "matches": [{"category": "topic_families", "key": family, "delta": round(delta, 2)}]}
    return max(-10.0, min(10.0, delta)), evidence


def _obvious_hard_rejection(candidate: dict, recent_hashes: list[str], memory: dict | None) -> str | None:
    title = str(candidate.get("title", ""))
    summary = str(candidate.get("summary", ""))
    text = f"{title} {summary}".lower()
    if not title.strip(): return "Missing topic title"
    if len(title.split()) < 3: return "Topic too broad — cannot payoff in one sentence"
    if any(x in text for x in ("requires code", "source code", "programming tutorial", "write code")): return "Requires charts or code to explain"
    if any(x in text for x in ("press release", "announces", "announcement")) and not any(x in text for x in ("why", "impact", "danger", "surprise")): return "Pure announcement with no conflict"
    family = _topic_family(title, summary)
    for rh in recent_hashes:
        if rh and candidate.get("content_hash", "").startswith(rh[:20]): return "Duplicate recent topic"
    for item in (memory or {}).get("weak_topic_families", []):
        if isinstance(item, dict) and item.get("key") == family and float(item.get("evidence_strength", 0)) >= 0.7: return "Historically weak topic family"
    return None


def score_candidates(candidates: list[dict[str, Any]], *, content_memory: dict | None = None, recent_hashes: list[str] | None = None) -> list[dict[str, Any]]:
    if not candidates:
        LOG.warning("No candidates to score")
        return []
    cfg = get_config(); scoring_cfg = cfg.get_path("topic_scoring", {}) or {}; weights = scoring_cfg.get("weights", {}) or {}; recent_hashes = recent_hashes or []
    strategy_path = repo_root() / "data" / "dynamic_strategy.json"
    dynamic_strategy: dict = {}
    if strategy_path.exists():
        try: dynamic_strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        except Exception as exc: LOG.warning("Failed to load dynamic_strategy.json: %s", exc)
    focused = [str(x).lower() for x in dynamic_strategy.get("focused_themes", [])]
    avoided = [str(x).lower() for x in dynamic_strategy.get("avoid_themes", [])]
    for c in candidates:
        c["scores"] = {}; title = str(c.get("title", "")); summary = str(c.get("summary", "")); text = f"{title} {summary}".lower(); hotness = float(c.get("normalized_hotness", 50) or 50)
        c["scores"]["freshness"] = max(0, min(100, hotness))
        source_quality_map = {"hackernews": 80, "github_trending": 65, "devto": 55, "wikipedia_otd": 70}; src = str(c.get("source", "")); base_src = src.split(":")[0] if ":" in src else src
        if base_src.startswith("reddit"): base_src = "reddit"
        c["scores"]["source_quality"] = 72 if base_src.startswith("rss") else source_quality_map.get(base_src, 60)
        spec = 40
        if re.search(r"\d", title): spec += 20
        if re.search(r"\$[\d,]+|billion|million|trillion|percent|%", title, re.I): spec += 15
        if len(title.split()) >= 5: spec += 10
        if any(w in title.lower() for w in ("why", "how", "secret", "hidden", "real reason", "truth", "myth")): spec += 15
        c["scores"]["specificity"] = min(100, spec)
        content_hash = c.get("content_hash", ""); c["scores"]["novelty"] = 0 if any(content_hash.startswith(rh[:20]) for rh in recent_hashes if rh) else 75
        c["scores"]["evergreen_value"] = 80 if any(s in title.lower() for s in ("history", "invented", "origin", "discovered", "first", "oldest", "science")) else 50
        strat = 50 + 20 * sum(1 for theme in focused if theme and theme in text) - 30 * sum(1 for theme in avoided if theme and theme in text)
        c["scores"]["strategy_alignment"] = max(0, min(100, strat))
        c["hard_rejection"] = _obvious_hard_rejection(c, recent_hashes, content_memory)
        if c["hard_rejection"]: c["scores"]["hard_rejection"] = 0
        memory_bonus, memory_evidence = _memory_adjustment(c, content_memory); c["memory_adjustment"] = round(memory_bonus, 2); c["memory_evidence"] = memory_evidence
    batch_size = max(1, int(scoring_cfg.get("llm_batch_size", 10)))
    for c in candidates: c["_heuristic_avg"] = sum(c["scores"].values()) / max(1, len(c["scores"]))
    candidates.sort(key=lambda c: c["_heuristic_avg"], reverse=True); top_batch = candidates[:batch_size]
    try:
        llm_scores = _llm_batch_score(top_batch, scoring_cfg)
        for c, scores in zip(top_batch, llm_scores): c["scores"].update(scores)
    except Exception as exc:
        LOG.warning("LLM batch scoring failed (%s), using heuristics only", exc)
        for c in top_batch: c["scores"].update({"audience_fit": 50, "curiosity_gap": 50, "story_potential": 50, "visual_potential": 50, "shareability": 50})
    for c in candidates[batch_size:]: c["scores"].update({"audience_fit": 40, "curiosity_gap": 40, "story_potential": 40, "visual_potential": 40, "shareability": 40})
    default_weights = {"audience_fit": .18, "curiosity_gap": .16, "story_potential": .13, "visual_potential": .10, "strategy_alignment": .12, "freshness": .08, "specificity": .07, "shareability": .07, "novelty": .05, "source_quality": .02, "evergreen_value": .02}; w = {**default_weights, **weights}; total_w = sum(w.values())
    if total_w > 0: w = {k: v / total_w for k, v in w.items()}
    for c in candidates:
        if c.get("hard_rejection"): c["total_score"] = 0.0
        else: c["total_score"] = round(max(0.0, min(100.0, sum(c["scores"].get(dim, 50) * weight for dim, weight in w.items()) + float(c.get("memory_adjustment", 0.0)))), 1)
        c.pop("_heuristic_avg", None)
    candidates.sort(key=lambda c: c["total_score"], reverse=True)
    LOG.info("Scored %d candidates. Top: %s (%.1f)", len(candidates), candidates[0].get("title", "")[:50], candidates[0].get("total_score", 0))
    return candidates


def select_best(scored: list[dict[str, Any]], *, min_score: float | None = None, exploration_ratio: float | None = None) -> dict[str, Any] | None:
    cfg = get_config(); scoring_cfg = cfg.get_path("topic_scoring", {}) or {}; min_score = float(scoring_cfg.get("min_topic_score", 72)) if min_score is None else min_score; exploration_ratio = float(scoring_cfg.get("exploration_ratio", .20)) if exploration_ratio is None else exploration_ratio
    qualified = [c for c in scored if c.get("total_score", 0) >= min_score and not c.get("hard_rejection")]
    if not qualified:
        if not scored: return None
        # Production mode: scoring ranks candidates; it does not block the day.
        # Hard rejections remain absolute. If every candidate is hard-rejected,
        # stopping is safer than manufacturing a topic.
        valid = [c for c in scored if not c.get("hard_rejection")]
        if not valid:
            LOG.warning("All candidates are hard-rejected; no safe topic to publish")
            return None
        best = valid[0]
        LOG.warning("No candidate met %.1f; production mode selecting best valid candidate %.1f: %s", min_score, best.get("total_score", 0), best.get("title", "")[:60])
        best["selection_mode"] = "best_valid"
        return best
    if len(qualified) > 1 and random.random() < exploration_ratio:
        chosen = random.choice(qualified[1:min(5, len(qualified))]); LOG.info("Exploration pick: %s (%.1f)", chosen.get("title", "")[:50], chosen.get("total_score", 0)); return chosen
    LOG.info("Top pick: %s (%.1f)", qualified[0].get("title", "")[:50], qualified[0].get("total_score", 0)); return qualified[0]


def _llm_batch_score(candidates: list[dict], scoring_cfg: dict) -> list[dict]:
    llm = LLMRouter("llm_research"); llm_cfg = scoring_cfg.get("llm", {}) or {}; goal = goal_summary(); candidate_lines = []
    for i, c in enumerate(candidates): candidate_lines.append(f"[{i}] {c.get('title', '')} (source: {c.get('source', '')}, hotness: {c.get('normalized_hotness', 0)})\n    Summary: {str(c.get('summary', ''))[:250]}")
    prompt_path = repo_root() / "prompts" / "topic_scoring.txt"; template = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else _DEFAULT_SCORING_PROMPT
    prompt = template.format(goal=goal, target_duration=int(get_config().get_path("video.target_duration_sec", 35)), n_candidates=len(candidates), candidates="\n".join(candidate_lines))
    result = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=int(llm_cfg.get("max_tokens", 3000)), temperature=float(llm_cfg.get("temperature", .3)), reasoning_effort=llm_cfg.get("reasoning_effort")); raw_scores = result.get("scores", []) if isinstance(result, dict) else []; raw_scores = raw_scores if isinstance(raw_scores, list) else []
    out = []
    for i in range(len(candidates)):
        s = raw_scores[i] if i < len(raw_scores) and isinstance(raw_scores[i], dict) else {}; out.append({k: _clamp(s.get(k, 50)) for k in ("audience_fit", "curiosity_gap", "story_potential", "visual_potential", "shareability")})
    return out


def _clamp(v: Any, lo: int = 0, hi: int = 100) -> int:
    try: return max(lo, min(hi, int(v)))
    except (TypeError, ValueError): return 50


_DEFAULT_SCORING_PROMPT = """{goal}

Score each candidate for a YouTube Short on this channel. For each, rate 0-100:
- audience_fit: Would the same viewer who watched yesterday want this?
- curiosity_gap: How strong is the \"I need to know\" pull?
- story_potential: Can this be told as a 30-second narrative arc?
- visual_potential: Are there concrete, filmable visuals?
- shareability: Would a viewer send this to a friend?

Reject mentally any topic that is generic, has no viewer consequence, has no curiosity gap, has no concrete payoff, is only an announcement, is too broad, or requires code/charts to explain.

Candidates:
{candidates}

Respond with ONLY a JSON object:
{{\"scores\": [{{\"audience_fit\": N, \"curiosity_gap\": N, \"story_potential\": N, \"visual_potential\": N, \"shareability\": N}}, ...]}}"""
