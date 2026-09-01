"""Discover trending topics from multiple sources.

All limits, UA, and timeouts read from pipeline.yaml > discover.
Enriches each candidate with normalized metadata for the scoring engine.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
import time
from typing import Any
import math

import feedparser
import requests

from ..config import get_config
from ..ledger import Ledger
from ..utils import repo_root
from .discovery_memory import DiscoveryMemory

LOG = logging.getLogger("utube.discover")


def _cfg() -> dict:
    return get_config().get_path("discover", {}) or {}


def discover_candidates(*, limit: int | None = None) -> list[dict[str, Any]]:
    """Collect candidates from ALL configured subtopic sources.

    Unlike the old ``discover_for_niche`` (which collected per-slot), this
    collects from every unique source across all subtopics, deduplicates,
    normalises hotness per source, and returns the full enriched pool.
    """
    cfg = _cfg()
    if limit is None:
        limit = int(cfg.get("total_candidates_limit", 40))
    per_limits = cfg.get("per_source_limits", {}) or {}

    # Collect unique source configs from all subtopics
    lanes_cfg = get_config().get_path("subtopics", []) or []
    source_specs: list[dict] = []
    seen_source_keys: set[str] = set()
    for st in lanes_cfg:
        for src in st.get("sources", []):
            key = _source_key(src)
            if key not in seen_source_keys:
                seen_source_keys.add(key)
                source_specs.append(src)

    candidates: list[dict] = []
    for src in source_specs:
        try:
            t = src["type"]
            if t == "hackernews":
                candidates += _hackernews(int(per_limits.get("hackernews", 15)))
            elif t == "reddit":
                n = int(per_limits.get("reddit_per_subreddit", 8))
                for sub in src.get("subreddits", []):
                    candidates += _reddit(sub, src.get("time_filter", "day"), n)
            elif t == "rss":
                n = int(per_limits.get("rss", 8))
                for url in src.get("urls", []):
                    candidates += _rss(url, n)
            elif t == "wikipedia_otd":
                candidates += _wikipedia_otd(int(per_limits.get("wikipedia_otd", 10)))
            elif t == "github_trending":
                candidates += _github_trending(int(per_limits.get("github_trending", 10)))
            elif t == "devto":
                candidates += _devto(int(per_limits.get("devto", 10)))
            elif t == "google_trends":
                geos = src.get("geos", ["US"])
                candidates += _google_trends(geos, int(per_limits.get("google_trends", 15)))
        except Exception as e:  # noqa: BLE001
            LOG.warning("Source %s failed: %s", src, e)

    # Dedupe by URL
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        u = c.get("url", "")
        if u in seen_urls:
            continue
        seen_urls.add(u)
        unique.append(c)

    # Normalize scores 0-100 per source type
    source_max: dict[str, float] = {}
    for c in unique:
        src = c.get("source", "unknown")
        score = c.get("score", 0)
        source_max[src] = max(source_max.get(src, 0), score)

    for c in unique:
        src = c.get("source", "unknown")
        max_s = source_max.get(src, 0)
        c["raw_score"] = c.get("score", 0)
        if max_s > 0:
            c["normalized_hotness"] = int((c["raw_score"] / max_s) * 100)
        else:
            c["normalized_hotness"] = 50  # Baseline for unscored sources

        # Generate a stable content hash for dedup against ledger
        c["content_hash"] = _content_hash(c.get("title", ""))

        # Extract keywords from title
        c["keywords"] = _extract_keywords(c.get("title", ""))

    # Cluster candidates into events
    clusters = _cluster_candidates(unique)

    LOG.info("Discovered %d unique candidates, clustered into %d events from %d sources", len(unique), len(clusters), len(source_specs))
    return clusters


# Backward compatibility alias
def discover_for_niche(slot: dict, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Legacy wrapper — collects from slot-specific sources only."""
    cfg = _cfg()
    if limit is None:
        limit = int(cfg.get("total_candidates_limit", 25))
    per_limits = cfg.get("per_source_limits", {}) or {}

    candidates: list[dict] = []
    for src in slot.get("sources", []):
        try:
            t = src["type"]
            if t == "hackernews":
                candidates += _hackernews(int(per_limits.get("hackernews", 15)))
            elif t == "reddit":
                n = int(per_limits.get("reddit_per_subreddit", 8))
                for sub in src.get("subreddits", []):
                    candidates += _reddit(sub, src.get("time_filter", "day"), n)
            elif t == "rss":
                n = int(per_limits.get("rss", 8))
                for url in src.get("urls", []):
                    candidates += _rss(url, n)
            elif t == "wikipedia_otd":
                candidates += _wikipedia_otd(int(per_limits.get("wikipedia_otd", 10)))
            elif t == "github_trending":
                candidates += _github_trending(int(per_limits.get("github_trending", 10)))
            elif t == "devto":
                candidates += _devto(int(per_limits.get("devto", 10)))
            elif t == "google_trends":
                geos = src.get("geos", ["US"])
                candidates += _google_trends(geos, int(per_limits.get("google_trends", 15)))
        except Exception as e:  # noqa: BLE001
            LOG.warning("Source %s failed: %s", src, e)

    seen, out = set(), []
    for c in candidates:
        u = c.get("url", "")
        if u in seen:
            continue
        seen.add(u)
        out.append(c)

    source_max = {}
    for c in out:
        src = c.get("source", "unknown")
        score = c.get("score", 0)
        source_max[src] = max(source_max.get(src, 0), score)

    for c in out:
        src = c.get("source", "unknown")
        max_s = source_max.get(src, 0)
        c["raw_score"] = c.get("score", 0)
        if max_s > 0:
            c["score"] = int((c["raw_score"] / max_s) * 100)
        else:
            c["score"] = 50

    LOG.info("Discovered %d candidates for slot %s", len(out), slot.get("id"))
    return out


# ──────────────────────── Helpers ────────────────────────────────────────────


def _source_key(src: dict) -> str:
    """Unique key for a source config to avoid duplicate fetches."""
    t = src.get("type", "")
    if t == "reddit":
        return f"reddit:{','.join(sorted(src.get('subreddits', [])))}"
    if t == "rss":
        return f"rss:{','.join(sorted(src.get('urls', [])))}"
    if t == "google_trends":
        return f"google_trends:{','.join(sorted(src.get('geos', [])))}"
    return t


def _cluster_candidates(candidates: list[dict]) -> list[dict]:
    """Group candidates into event clusters based on semantic keyword overlap."""
    clusters = []
    
    # Helper to calculate Jaccard similarity of keywords
    def sim(k1: list[str], k2: list[str]) -> float:
        s1, s2 = set(k1), set(k2)
        if not s1 or not s2: return 0.0
        return len(s1 & s2) / len(s1 | s2)
        
    for c in candidates:
        matched = False
        for cluster in clusters:
            # If high keyword overlap (>= 0.4 Jaccard), merge into cluster
            if sim(c.get("keywords", []), cluster["keywords"]) >= 0.4:
                cluster["candidates"].append(c)
                # Expand keywords
                cluster["keywords"] = list(set(cluster["keywords"] + c.get("keywords", [])))
                matched = True
                break
        if not matched:
            clusters.append({
                "title": c["title"], # Representative title
                "url": c["url"],     # Representative URL
                "summary": c.get("summary", ""),
                "keywords": c.get("keywords", []),
                "candidates": [c]
            })
            
    # Build Evidence Packets
    memory = DiscoveryMemory()
    out = []
    for cl in clusters:
        ev_id = _content_hash(cl["title"])
        cands = cl["candidates"]
        sources = [c["source"] for c in cands]
        urls = [c.get("url") for c in cands if c.get("url")]
        classes = {c.get("source_class") for c in cands if c.get("source_class")}
        
        # Calculate independent sources (unique base sources)
        base_sources = {s.split(":")[0] for s in sources}
        indep_count = len(base_sources)
        
        # Calculate best source score
        best_cand = max(cands, key=lambda x: x.get("normalized_hotness", 0))
        
        gt_cand_for_vol = next((c for c in cands if c["source"].startswith("google_trends")), None)
        traffic_vol = gt_cand_for_vol.get("traffic_volume", 0) if gt_cand_for_vol else 0
        mem_stats = memory.record_event(ev_id, len(cands), traffic_vol)
        
        packet = {
            "event_id": ev_id,
            "representative_title": cl["title"],
            "canonical_sources": list(set(sources)),
            "all_relevant_urls": list(set(urls)),
            "source_classes": list(classes),
            "independent_source_count": indep_count,
            "mention_count": len(cands),
            "velocity": mem_stats["velocity"],
            "acceleration": mem_stats["acceleration"],
            "novelty": mem_stats["novelty"],
            "trend_momentum": mem_stats.get("trend_momentum", 0)
        }
        
        # Extract Google Trends specific signal
        gt_cand = next((c for c in cands if c["source"].startswith("google_trends")), None)
        if gt_cand:
            packet["google_trends_signal"] = {
                "present": True,
                "query": gt_cand["title"],
                "geography": gt_cand.get("geography", "GLOBAL"),
                "traffic_volume": gt_cand.get("traffic_volume", 0),
                "news_context": gt_cand.get("summary", "")
            }
            # Remove google_trends from canonical_sources as it's demand, not evidence
            packet["canonical_sources"] = [s for s in packet["canonical_sources"] if not s.startswith("google_trends")]

        out.append({
            "title": cl["title"],
            "url": cl["url"],
            "summary": cl["summary"],
            "source": best_cand.get("source", "unknown"),
            "normalized_hotness": best_cand.get("normalized_hotness", 0),
            "content_hash": ev_id,
            "keywords": cl["keywords"],
            "evidence_packet": packet
        })
        
    memory.save()
    return out


def _content_hash(title: str) -> str:
    """Short stable hash for dedup — slug-like."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    h = hashlib.md5(slug.encode()).hexdigest()[:8]
    return f"{slug[:40]}-{h}"


def _extract_keywords(title: str) -> list[str]:
    """Pull simple keywords from title for subtopic matching."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out",
        "it", "its", "this", "that", "these", "those", "and", "but", "or",
        "nor", "not", "no", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "how", "what", "which", "who", "whom",
        "why", "when", "where", "new", "now", "also", "about", "over", "up",
    }
    words = re.findall(r"[a-z]+", title.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def _ua() -> str:
    return _cfg().get("user_agent", "utube-bot/1.0")


def _timeout() -> int:
    return int(_cfg().get("request_timeout_sec", 20))


def _hackernews(limit: int) -> list[dict]:
    ledger = Ledger.load(repo_root() / "ledger.json")
    health_score = ledger.get_source_health("hackernews")
    if health_score < 0.5:
        limit = max(1, int(limit * health_score))
        
    try:
        r = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"tags": "front_page", "hitsPerPage": limit},
            headers={"User-Agent": _ua()},
            timeout=_timeout(),
        )
        r.raise_for_status()
        out = []
        for h in r.json().get("hits", []):
            out.append({
                "title": h.get("title") or "",
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "score": h.get("points") or 0,
                "summary": "",
                "source": "hackernews",
                "source_class": "tier2",
                "credibility_tier": 2,
                "num_comments": h.get("num_comments", 0),
            })
        ledger.record_source_health("hackernews", True)
        return out
    except Exception as exc:
        ledger.record_source_health("hackernews", False, str(exc))
        LOG.warning("HackerNews failed: %s", exc)
        return []


_REDDIT_RATE_LIMITED = False

def _reddit(subreddit: str, time_filter: str, limit: int) -> list[dict]:
    global _REDDIT_RATE_LIMITED
    if _REDDIT_RATE_LIMITED:
        return []
        
    """Fetch Reddit JSON, with an RSS fallback for hosted CI runners."""
    ledger = Ledger.load(repo_root() / "ledger.json")
    src_key = f"reddit:{subreddit}"
    health_score = ledger.get_source_health(src_key)
    if health_score < 0.5:
        limit = max(1, int(limit * health_score))
    params = {"t": time_filter, "limit": limit}
    headers = {"User-Agent": _ua()}
    
    # Polite delay to prevent Reddit from immediately 429'ing the IP
    time.sleep(1.5)
    
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=_timeout())
        if r.status_code == 429:
            LOG.warning("Reddit rate-limited for r/%s (429). Skipping RSS fallback.", subreddit)
            return []
        elif r.status_code == 403:
            LOG.warning("Reddit blocked JSON access for r/%s; trying RSS fallback", subreddit)
        else:
            r.raise_for_status()
            out = []
            for c in r.json().get("data", {}).get("children", []):
                d = c.get("data", {})
                if d.get("over_18") or d.get("stickied"):
                    continue
                out.append({
                    "title": d.get("title", ""),
                    "url": "https://reddit.com" + d.get("permalink", ""),
                    "external_url": d.get("url"),
                    "score": d.get("score", 0),
                    "summary": (d.get("selftext") or "")[:500],
                    "source": src_key,
                    "source_class": "tier2",
                    "credibility_tier": 2,
                    "num_comments": d.get("num_comments", 0),
                })
            ledger.record_source_health(src_key, True)
            return out
    except requests.RequestException as exc:
        LOG.warning("Reddit JSON failed for r/%s: %s; trying RSS fallback", subreddit, exc)

    # Reddit's RSS endpoint is often available when JSON is blocked by CI IPs.
    try:
        rss = requests.get(
            f"https://www.reddit.com/r/{subreddit}/top/.rss",
            params={"t": time_filter, "limit": limit},
            headers=headers,
            timeout=_timeout(),
        )
        if rss.status_code == 429:
            LOG.warning("Reddit RSS rate-limited (429). Halting all further Reddit fetches.")
            _REDDIT_RATE_LIMITED = True
            return []
        rss.raise_for_status()
        feed = feedparser.parse(rss.content)
        out = []
        for entry in feed.entries[:limit]:
            link = entry.get("link", "")
            title = entry.get("title", "")
            summary = re.sub(r"<[^>]+>", " ", entry.get("summary", ""))[:500]
            out.append({
                "title": title,
                "url": link,
                "external_url": link,
                "score": 0,
                "summary": summary,
                "source": src_key,
                "source_class": "tier2",
                "credibility_tier": 2,
                "num_comments": 0,
            })
        ledger.record_source_health(src_key, True)
        return out
    except Exception as exc:
        ledger.record_source_health(src_key, False, str(exc))
        LOG.warning("Reddit RSS fallback failed for r/%s: %s", subreddit, exc)
        return []


def _rss(url: str, limit: int) -> list[dict]:
    ledger = Ledger.load(repo_root() / "ledger.json")
    health_score = ledger.get_source_health(url)
    if health_score < 0.5:
        limit = max(1, int(limit * health_score))
        
    try:
        feed = feedparser.parse(url)
        out = []
        now = time.time()
        for e in feed.entries[:limit]:
            score = 0
            if e.get("published_parsed"):
                pub_ts = time.mktime(e.published_parsed)
                age_hours = (now - pub_ts) / 3600
                score = max(0, int(100 * (1 - age_hours / 168)))

            out.append({
                "title": e.get("title", ""),
                "url": e.get("link", ""),
                "score": score,
                "summary": (e.get("summary") or "")[:500],
                "source": f"rss:{feed.feed.get('title', 'rss')}",
                "source_class": "tier1",
                "credibility_tier": 1,
            })
        ledger.record_source_health(url, True)
        return out
    except Exception as exc:
        ledger.record_source_health(url, False, str(exc))
        LOG.warning("RSS failed for %s: %s", url, exc)
        return []


def _wikipedia_otd(limit: int) -> list[dict]:
    ledger = Ledger.load(repo_root() / "ledger.json")
    health_score = ledger.get_source_health("wikipedia_otd")
    if health_score < 0.5:
        limit = max(1, int(limit * health_score))
        
    try:
        today = datetime.now(timezone.utc)
        url = (
            f"https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/events/"
            f"{today.month:02d}/{today.day:02d}"
        )
        r = requests.get(url, headers={"User-Agent": _ua()}, timeout=_timeout())
        r.raise_for_status()
        out = []
        for ev in r.json().get("events", [])[:limit]:
            pages = ev.get("pages", [])
            link = pages[0].get("content_urls", {}).get("desktop", {}).get("page", "") if pages else ""
            out.append({
                "title": f"On {today.strftime('%B %d')}, {ev.get('year')}: {ev.get('text','')}",
                "url": link or "https://en.wikipedia.org/wiki/Main_Page",
                "score": 0,
                "summary": ev.get("text", ""),
                "source": "wikipedia_otd",
                "source_class": "tier1",
                "credibility_tier": 1,
            })
        ledger.record_source_health("wikipedia_otd", True)
        return out
    except Exception as exc:
        ledger.record_source_health("wikipedia_otd", False, str(exc))
        LOG.warning("Wikipedia failed: %s", exc)
        return []


def _github_trending(limit: int) -> list[dict]:
    ledger = Ledger.load(repo_root() / "ledger.json")
    health_score = ledger.get_source_health("github_trending")
    if health_score < 0.5:
        limit = max(1, int(limit * health_score))
        
    try:
        token = __import__("os").getenv("GITHUB_TOKEN", "")
        since = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
        headers = {"Accept": "application/vnd.github+json", "User-Agent": _ua()}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(
            "https://api.github.com/search/repositories",
            params={
                "q": f"stars:>50 pushed:>={since}",
                "sort": "stars",
                "order": "desc",
                "per_page": limit,
            },
            headers=headers,
            timeout=_timeout(),
        )
        if r.status_code == 401:
            LOG.warning("GitHub API Unauthorized (401). Ensure GITHUB_TOKEN is set. Skipping.")
            return []
        r.raise_for_status()
        out = []
        for repo in r.json().get("items", [])[:limit]:
            out.append({
                "title": f"{repo.get('full_name')}: {repo.get('description','') or ''}",
                "url": repo.get("html_url", ""),
                "score": repo.get("stargazers_count", 0),
                "summary": repo.get("description", "") or "",
                "source": "github_trending",
                "source_class": "tier4",
                "credibility_tier": 4,
            })
        ledger.record_source_health("github_trending", True)
        return out
    except Exception as e:  # noqa: BLE001
        ledger.record_source_health("github_trending", False, str(e))
        LOG.warning("github_trending unavailable: %s", e)
        return []


def _devto(limit: int) -> list[dict]:
    ledger = Ledger.load(repo_root() / "ledger.json")
    health_score = ledger.get_source_health("devto")
    if health_score < 0.5:
        limit = max(1, int(limit * health_score))
        
    try:
        r = requests.get("https://dev.to/api/articles", params={"top": "1"}, timeout=_timeout())
        r.raise_for_status()
        out = []
        for a in r.json()[:limit]:
            out.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "score": a.get("public_reactions_count", 0),
                "summary": a.get("description", "") or "",
                "source": "devto",
                "source_class": "tier4",
                "credibility_tier": 4,
            })
        ledger.record_source_health("devto", True)
        return out
    except Exception as exc:
        ledger.record_source_health("devto", False, str(exc))
        LOG.warning("devto failed: %s", exc)
        return []


def _google_trends(geos: list[str], limit: int) -> list[dict]:
    import xml.etree.ElementTree as ET
    ledger = Ledger.load(repo_root() / "ledger.json")
    
    out = []
    for geo in geos:
        geo_str = "" if geo == "GLOBAL" else geo
        url = f"https://trends.google.com/trending/rss?geo={geo_str}" if geo_str else "https://trends.google.com/trending/rss"
        health_score = ledger.get_source_health(f"google_trends:{geo}")
        
        current_limit = limit
        if health_score < 0.5:
            current_limit = max(1, int(limit * health_score))
            
        try:
            r = requests.get(url, headers={"User-Agent": _ua()}, timeout=_timeout())
            r.raise_for_status()
            root = ET.fromstring(r.content)
            
            # Find all <item> tags
            items = root.findall(".//item")
            for item in items[:current_limit]:
                title = item.findtext("title", "")
                
                # Extract traffic volume (e.g. "50K+", "2M+")
                traffic_str = item.findtext("{https://trends.google.com/trending/rss}approx_traffic", "0")
                traffic_str_clean = traffic_str.replace("+", "").replace(",", "")
                traffic_vol = 0
                if "K" in traffic_str_clean:
                    traffic_vol = int(float(traffic_str_clean.replace("K", "")) * 1000)
                elif "M" in traffic_str_clean:
                    traffic_vol = int(float(traffic_str_clean.replace("M", "")) * 1000000)
                elif traffic_str_clean.isdigit():
                    traffic_vol = int(traffic_str_clean)
                
                news_items = item.findall("{https://trends.google.com/trending/rss}news_item")
                news_context = []
                for ni in news_items:
                    ni_title = ni.findtext("{https://trends.google.com/trending/rss}news_item_title", "")
                    ni_snippet = ni.findtext("{https://trends.google.com/trending/rss}news_item_snippet", "")
                    if ni_title:
                        news_context.append(f"{ni_title} - {ni_snippet}")
                summary = " | ".join(news_context)
                
                # Derive a score from traffic volume to fit into the normalized hotness 0-100 system
                # A baseline of 100K searches will be a 50 score
                score = min(100, int((math.log10(max(1, traffic_vol)) / 6.0) * 100))
                
                link = item.findtext("link", "")
                
                out.append({
                    "title": title,
                    "url": link,
                    "score": score,
                    "summary": summary,
                    "source": f"google_trends:{geo}",
                    "source_class": "demand_signal",
                    "credibility_tier": 0,  # Not factual evidence
                    "geography": geo,
                    "traffic_volume": traffic_vol
                })
            ledger.record_source_health(f"google_trends:{geo}", True)
        except Exception as exc:
            ledger.record_source_health(f"google_trends:{geo}", False, str(exc))
            LOG.warning("Google Trends RSS failed for geo %s: %s", geo, exc)
            
    return out