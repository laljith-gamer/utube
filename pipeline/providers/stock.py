"""Stock B-roll router — Pexels + Pixabay, fully config-driven.

The find_video method takes an ordered list of queries; we try each in turn
until one matches. Caller passes the most-specific (key_subject) first and
broader fallbacks (broll_keywords) after.
"""
from __future__ import annotations

import logging
import random
from typing import Any

import requests

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.stock")


class StockRouter:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg.get_path("stock", {}) or {}
        self.chain: list[str] = self.cfg.get("chain", []) or []
        self.providers: dict[str, dict[str, Any]] = self.cfg.get("providers", {}) or {}
        self.timeout = self.cfg.get("request_timeout_sec", 120)

    def find_video(self, queries: list[str], *, orientation: str = "portrait") -> bytes | None:
        """Try each query in order, across all providers in chain. First hit wins."""
        # Filter empty / dedupe while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for q in queries:
            q = (q or "").strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                ordered.append(q)

        for q in ordered:
            for name in self.chain:
                p = self.providers.get(name)
                if not p:
                    continue
                api_key = env(p.get("api_key_env", ""))
                if not api_key:
                    continue
                try:
                    if name == "pexels":
                        url = self._pexels(p, q, orientation, api_key)
                    elif name == "pixabay":
                        url = self._pixabay(p, q, api_key)
                    else:
                        continue
                    if url:
                        return self._download(url)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("Stock provider %s failed for %r: %s", name, q, e)
        LOG.info("Stock: no match for any of %d queries", len(ordered))
        return None

    def _pexels(self, p: dict, query: str, orientation: str, api_key: str) -> str | None:
        params = {
            "query": query,
            "orientation": orientation,
            "per_page": p.get("per_page", 5),
            "size": p.get("size", "medium"),
        }
        r = requests.get(
            p["url"], params=params, headers={"Authorization": api_key}, timeout=self.timeout
        )
        r.raise_for_status()
        videos = r.json().get("videos", [])
        if not videos:
            return None
        v = random.choice(videos[: min(3, len(videos))])
        target_w = 1080 if orientation == "portrait" else 1920
        files = sorted(
            [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"],
            key=lambda f: abs((f.get("width") or 0) - target_w),
        )
        if files:
            LOG.info("B-roll via Pexels: %r → %s", query, files[0]["link"][:80])
            return files[0]["link"]
        return None

    def _pixabay(self, p: dict, query: str, api_key: str) -> str | None:
        params = {
            "key": api_key,
            "q": query,
            "per_page": p.get("per_page", 5),
            "safesearch": str(bool(p.get("safesearch", True))).lower(),
        }
        r = requests.get(p["url"], params=params, timeout=self.timeout)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        if not hits:
            return None
        v = random.choice(hits[: min(3, len(hits))])
        for size_key in ("medium", "large", "small", "tiny"):
            sized = v.get("videos", {}).get(size_key, {})
            if sized.get("url"):
                LOG.info("B-roll via Pixabay: %r → %s", query, sized["url"][:80])
                return sized["url"]
        return None

    def _download(self, url: str) -> bytes:
        r = requests.get(url, timeout=self.timeout, stream=True)
        r.raise_for_status()
        return r.content
