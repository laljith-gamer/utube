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

import feedparser
import requests

from ..config import get_config

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

    LOG.info("Discovered %d unique candidates from %d sources", len(unique), len(source_specs))
    return unique


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
    return t


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
            "num_comments": h.get("num_comments", 0),
        })
    return out


def _reddit(subreddit: str, time_filter: str, limit: int) -> list[dict]:
    """Fetch Reddit JSON, with an RSS fallback for hosted CI runners."""
    params = {"t": time_filter, "limit": limit}
    headers = {"User-Agent": _ua()}
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=_timeout())
        if r.status_code == 429:
            LOG.warning("Reddit rate-limited for r/%s; trying RSS fallback", subreddit)
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
                    "source": f"reddit:{subreddit}",
                    "num_comments": d.get("num_comments", 0),
                })
            return out
    except requests.RequestException as exc:
        LOG.warning("Reddit JSON failed for r/%s: %s; trying RSS fallback", subreddit, exc)

    # Reddit's RSS endpoint is often available when JSON is blocked by CI IPs.
    rss = requests.get(
        f"https://www.reddit.com/r/{subreddit}/top/.rss",
        params={"t": time_filter, "limit": limit},
        headers=headers,
        timeout=_timeout(),
    )
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
            "source": f"reddit:{subreddit}",
            "num_comments": 0,
        })
    return out


def _rss(url: str, limit: int) -> list[dict]:
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
        })
    return out


def _wikipedia_otd(limit: int) -> list[dict]:
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
        })
    return out


def _github_trending(limit: int) -> list[dict]:
    """Approximate GitHub daily trending with recently-pushed, popular repos.

    The previous third-party gitterapp endpoint now returns 404. GitHub's
    authenticated search API is more stable and gives us a first-party,
    current signal without scraping HTML.
    """
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
        r.raise_for_status()
        out = []
        for repo in r.json().get("items", [])[:limit]:
            out.append({
                "title": f"{repo.get('full_name')}: {repo.get('description','') or ''}",
                "url": repo.get("html_url", ""),
                "score": repo.get("stargazers_count", 0),
                "summary": repo.get("description", "") or "",
                "source": "github_trending",
            })
        return out
    except Exception as e:  # noqa: BLE001
        LOG.warning("github_trending unavailable: %s", e)
        return []


def _devto(limit: int) -> list[dict]:
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
        })
    return out