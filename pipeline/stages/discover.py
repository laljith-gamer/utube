"""Discover trending topics from per-niche free sources."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import feedparser
import requests

LOG = logging.getLogger("utube.discover")

USER_AGENT = "utube-bot/1.0 (+https://github.com/laljith-gamer/utube)"


def discover_for_niche(slot: dict, *, limit: int = 25) -> list[dict[str, Any]]:
    """Return up to `limit` candidate topics from the slot's configured sources."""
    candidates: list[dict] = []
    for src in slot.get("sources", []):
        try:
            t = src["type"]
            if t == "hackernews":
                candidates += _hackernews(limit=15)
            elif t == "reddit":
                for sub in src.get("subreddits", []):
                    candidates += _reddit(sub, src.get("time_filter", "day"), limit=8)
            elif t == "rss":
                for url in src.get("urls", []):
                    candidates += _rss(url, limit=8)
            elif t == "wikipedia_otd":
                candidates += _wikipedia_otd(limit=10)
            elif t == "github_trending":
                candidates += _github_trending(limit=10)
            elif t == "devto":
                candidates += _devto(limit=10)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Source %s failed: %s", src, e)

    # Dedupe by URL
    seen, out = set(), []
    for c in candidates:
        u = c.get("url", "")
        if u in seen:
            continue
        seen.add(u)
        out.append(c)
    LOG.info("Discovered %d candidates for slot %s", len(out), slot.get("id"))
    return out[:limit]


# ---------- individual sources ----------

def _hackernews(*, limit: int) -> list[dict]:
    r = requests.get(
        "https://hn.algolia.com/api/v1/search",
        params={"tags": "front_page", "hitsPerPage": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
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
        })
    return out


def _reddit(subreddit: str, time_filter: str, *, limit: int) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    r = requests.get(
        url,
        params={"t": time_filter, "limit": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    if r.status_code == 429:
        LOG.warning("Reddit rate-limited for r/%s", subreddit)
        return []
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
        })
    return out


def _rss(url: str, *, limit: int) -> list[dict]:
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:limit]:
        out.append({
            "title": e.get("title", ""),
            "url": e.get("link", ""),
            "score": 0,
            "summary": (e.get("summary") or "")[:500],
            "source": f"rss:{feed.feed.get('title','rss')}",
        })
    return out


def _wikipedia_otd(*, limit: int) -> list[dict]:
    today = datetime.now(timezone.utc)
    url = (
        f"https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/events/"
        f"{today.month:02d}/{today.day:02d}"
    )
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
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


def _github_trending(*, limit: int) -> list[dict]:
    # No official API; use https://api.gitterapp.com/repositories (community mirror) or scrape
    try:
        r = requests.get(
            "https://api.gitterapp.com/repositories",
            params={"since": "daily"},
            timeout=20,
        )
        r.raise_for_status()
        out = []
        for repo in r.json()[:limit]:
            out.append({
                "title": f"{repo.get('author')}/{repo.get('name')}: {repo.get('description','')}",
                "url": repo.get("url", ""),
                "score": repo.get("stars", 0),
                "summary": repo.get("description", "") or "",
                "source": "github_trending",
            })
        return out
    except Exception as e:  # noqa: BLE001
        LOG.warning("github_trending unavailable: %s", e)
        return []


def _devto(*, limit: int) -> list[dict]:
    r = requests.get("https://dev.to/api/articles", params={"top": "1"}, timeout=20)
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
