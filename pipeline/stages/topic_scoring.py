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
        
        # Pull features from evidence packet
        ev = c.get("evidence_packet", {})
        c["scores"]["velocity"] = max(0, min(100, int(ev.get("velocity", 0) * 10)))
        c["scores"]["acceleration"] = max(0, min(100, int(ev.get("acceleration", 0) * 20)))
        c["scores"]["corroboration"] = min(100, int(ev.get("independent_source_count", 1) * 25))
        
        gt_signal = ev.get("google_trends_signal", {})
        c["scores"]["trend_momentum"] = max(0, min(100, int(ev.get("trend_momentum", 0) / 1000)))
        c["scores"]["trend_volume"] = max(0, min(100, int(gt_signal.get("traffic_volume", 0) / 10000))) if gt_signal.get("present") else 0
        
        novelty_mem = int(ev.get("novelty", 1.0) * 100)
        
        content_hash = c.get("content_hash", ""); c["scores"]["novelty"] = 0 if any(content_hash.startswith(rh[:20]) for rh in recent_hashes if rh) else novelty_mem
        c["scores"]["evergreen_value"] = 80 if any(s in title.lower() for s in ("history", "invented", "origin", "discovered", "first", "oldest", "science")) else 50
        strat = 50 + 20 * sum(1 for theme in focused if theme and theme in text) - 30 * sum(1 for theme in avoided if theme and theme in text)
        c["scores"]["strategy_alignment"] = max(0, min(100, strat))
        c["hard_rejection"] = _obvious_hard_rejection(c, recent_hashes, content_memory)
        if c["hard_rejection"]: c["scores"]["hard_rejection"] = 0
        memory_bonus, memory_evidence = _memory_adjustment(c, content_memory); c["memory_adjustment"] = round(memory_bonus, 2); c["memory_evidence"] = memory_evidence
        default_weights = {"strategy_alignment": .15, "freshness": .10, "specificity": .10, "novelty": .10, "source_quality": .10, "evergreen_value": .05, "velocity": .10, "acceleration": .10, "corroboration": .10, "trend_momentum": .05, "trend_volume": .05}
    w = {**default_weights, **weights}; total_w = sum(w.values())
    if total_w > 0: w = {k: v / total_w for k, v in w.items()}
    for c in candidates:
        if c.get("hard_rejection"): c["total_score"] = 0.0
        else: c["total_score"] = round(max(0.0, min(100.0, sum(c["scores"].get(dim, 50) * weight for dim, weight in w.items()) + float(c.get("memory_adjustment", 0.0)))), 1)
        c.pop("_heuristic_avg", None)
    candidates.sort(key=lambda c: c["total_score"], reverse=True)
    LOG.info("Scored %d candidates. Top: %s (%.1f)", len(candidates), candidates[0].get("title", "")[:50], candidates[0].get("total_score", 0))
    return candidates


def _generate_shortlist(scored: list[dict], size: int = 12) -> list[dict]:
    # Need to get top overall, top emerging (velocity/acceleration), top corroboration, and diverse lanes
    valid = [c for c in scored if not c.get("hard_rejection") and c.get("total_score", 0) > 0]
    if not valid: return []
    shortlist = []
    seen = set()
    
    def add_candidates(cands: list[dict], count: int):
        added = 0
        for c in cands:
            h = c.get("content_hash", "") or c.get("title", "")
            if h not in seen and added < count:
                shortlist.append(c)
                seen.add(h)
                added += 1

    # Top 3 overall
    add_candidates(valid, 3)
    
    # Top 3 emerging
    emerging = sorted(valid, key=lambda c: c.get("scores", {}).get("velocity", 0) + c.get("scores", {}).get("acceleration", 0), reverse=True)
    add_candidates(emerging, 3)
    
    # Top 3 evidence-rich
    evidence = sorted(valid, key=lambda c: c.get("scores", {}).get("corroboration", 0), reverse=True)
    add_candidates(evidence, 3)
    
    # Top from diverse lanes (topic_family)
    families = {}
    for c in valid:
        fam = _topic_family(c.get("title", ""), c.get("summary", ""))
        if fam not in families: families[fam] = []
        families[fam].append(c)
    
    for fam, cands in families.items():
        add_candidates(cands, 1)
        if len(shortlist) >= size: break
        
    # Fill remaining
    add_candidates(valid, size - len(shortlist))
    return shortlist[:size]


def select_best(scored: list[dict[str, Any]], *, min_score: float | None = None, exploration_ratio: float | None = None) -> dict[str, Any] | None:
    cfg = get_config(); scoring_cfg = cfg.get_path("topic_scoring", {}) or {}
    min_score = float(scoring_cfg.get("min_topic_score", 65)) if min_score is None else min_score
    
    # Floor: Must meet min_score OR have high corroboration/novelty
    qualified = []
    for c in scored:
        if c.get("hard_rejection"): continue
        ts = c.get("total_score", 0)
        ev = c.get("evidence_packet", {})
        corr = ev.get("independent_source_count", 1)
        novelty = c.get("scores", {}).get("novelty", 0)
        if ts >= min_score or (corr >= 3 and novelty >= 60):
            qualified.append(c)
            
    if not qualified:
        LOG.warning("No candidate met the strict quality floor. Falling back to the best available candidates.")
        qualified = [c for c in scored if not c.get("hard_rejection")]
        if not qualified:
            LOG.warning("No candidates available without hard rejection. Aborting.")
            return None

    shortlist = _generate_shortlist(qualified, size=15)
    if not shortlist: return None
    
    LOG.info("Generated diverse shortlist of %d candidates for Opus Stage A.", len(shortlist))
    
    # Stage A: Shortlist -> Finalists
    finalist_indices = _llm_stage_a(shortlist, scoring_cfg)
    if not finalist_indices:
        LOG.warning("LLM Stage A failed to pick finalists. Falling back to top 3 deterministic.")
        finalist_indices = [0, 1, 2][:len(shortlist)]
    
    finalists = [shortlist[i] for i in finalist_indices if 0 <= i < len(shortlist)]
    LOG.info("Opus Stage A selected %d finalists.", len(finalists))
    
    # Retrieval Enrichment
    _enrich_finalists(finalists)
    
    # Stage B: Finalists -> Winner
    decision = _llm_stage_b(finalists, scoring_cfg)
    
    winner_idx = decision.get("selected_index", -1)
    if winner_idx == -1 or not (0 <= winner_idx < len(finalists)):
        LOG.warning("LLM Stage B rejected all finalists or returned invalid index.")
        return None
        
    winner = finalists[winner_idx]
    winner["decision_object"] = decision
    LOG.info("Winner selected via Stage B: %s (Reason: %s)", winner.get("title", "")[:50], decision.get("overall_reason", ""))
    return winner


def _llm_stage_a(shortlist: list[dict], scoring_cfg: dict) -> list[int]:
    llm = LLMRouter("llm_research")
    candidate_lines = []
    for i, c in enumerate(shortlist):
        ev = c.get('evidence_packet', {})
        corrob = ev.get('independent_source_count', 1)
        vel = c.get('scores', {}).get('velocity', 0)
        gt_signal = ev.get('google_trends_signal', {})
        trend_str = f", Google Trends: {gt_signal.get('traffic_volume', 0)} searches (momentum: {ev.get('trend_momentum', 0)})" if gt_signal.get("present") else ""
        sources = ", ".join(ev.get('canonical_sources', [c.get('source', '')]))
        candidate_lines.append(f"[{i}] {c.get('title', '')} (Sources: {sources}, Corrob: {corrob}, Velocity: {vel}{trend_str})\n    Summary: {str(c.get('summary', ''))[:300]}")
        
    prompt_path = repo_root() / "prompts" / "topic_picker_stage_a.txt"
    if not prompt_path.exists(): return []
    prompt = prompt_path.read_text(encoding="utf-8").format(
        goal=goal_summary(),
        niche_title=get_config().get_path("channel.niche", "Technology"),
        n_candidates=len(shortlist),
        candidates="\n".join(candidate_lines)
    )
    
    result = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.3)
    if isinstance(result, dict) and "finalists" in result and isinstance(result["finalists"], list):
        return [int(x) for x in result["finalists"] if str(x).isdigit()]
    return []


def _enrich_finalists(finalists: list[dict]):
    from .research import fetch_source_text
    LOG.info("Enriching %d finalists...", len(finalists))
    for c in finalists:
        url = c.get("external_url") or c.get("url", "")
        if url and not url.startswith("https://reddit.com"):
            try:
                text = fetch_source_text(url)
                if text:
                    if "evidence_packet" not in c: c["evidence_packet"] = {}
                    c["evidence_packet"]["enriched_text"] = text[:3000]
            except Exception as exc:
                LOG.warning("Failed to enrich %s: %s", url, exc)


def _llm_stage_b(finalists: list[dict], scoring_cfg: dict) -> dict:
    llm = LLMRouter("llm_research")
    cfg = get_config()
    candidate_lines = []
    for i, c in enumerate(finalists):
        ev = c.get('evidence_packet', {})
        corrob = ev.get('independent_source_count', 1)
        sources = ", ".join(ev.get('canonical_sources', [c.get('source', '')]))
        gt_signal = ev.get('google_trends_signal', {})
        trend_str = f"\nGoogle Trends Signal: Volume {gt_signal.get('traffic_volume', 0)}, Momentum {ev.get('trend_momentum', 0)}, Region {gt_signal.get('geography', '')}\nTrends Context: {gt_signal.get('news_context', '')}" if gt_signal.get("present") else ""
        enriched = ev.get('enriched_text', '(No full text available)')
        candidate_lines.append(f"[{i}] {c.get('title', '')}\nSources: {sources}\nIndependent Count: {corrob}{trend_str}\nSummary: {str(c.get('summary', ''))[:300]}\nEnriched Evidence:\n{enriched[:1000]}\n")
        
    prompt_path = repo_root() / "prompts" / "topic_picker_stage_b.txt"
    if not prompt_path.exists(): return {"selected_index": -1}
    prompt = prompt_path.read_text(encoding="utf-8").format(
        goal=goal_summary(),
        niche_title=cfg.get_path("channel.niche", "Technology"),
        n_finalists=len(finalists),
        target_duration=cfg.get_path("video.target_duration_sec", 35),
        format_label=cfg.get_path("channel.format", "shorts"),
        finalists="\n".join(candidate_lines)
    )
    
    result = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.3)
    if isinstance(result, dict):
        return result
    return {"selected_index": -1}
