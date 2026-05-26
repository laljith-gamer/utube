"""Stock media (B-roll + music) router.

Primary:  Pexels API (free key)
Fallback: Pixabay API (free key)
"""
from __future__ import annotations

import logging
import random

import requests

from ..utils import env

LOG = logging.getLogger("utube.stock")


class StockRouter:
    def __init__(self) -> None:
        self.pexels_key = env("PEXELS_API_KEY")
        self.pixabay_key = env("PIXABAY_API_KEY")

    # ----- VIDEO -----
    def find_video(self, keywords: list[str], *, orientation: str = "portrait") -> bytes | None:
        for kw in keywords:
            url = self._pexels_video_url(kw, orientation)
            if url:
                return _download(url)
            url = self._pixabay_video_url(kw, orientation)
            if url:
                return _download(url)
        return None

    def _pexels_video_url(self, query: str, orientation: str) -> str | None:
        if not self.pexels_key:
            return None
        try:
            r = requests.get(
                "https://api.pexels.com/videos/search",
                params={"query": query, "orientation": orientation, "per_page": 5, "size": "medium"},
                headers={"Authorization": self.pexels_key},
                timeout=30,
            )
            r.raise_for_status()
            videos = r.json().get("videos", [])
            if not videos:
                return None
            v = random.choice(videos[: min(3, len(videos))])
            files = sorted(
                [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"],
                key=lambda f: abs((f.get("width") or 0) - (1080 if orientation == "portrait" else 1920)),
            )
            if files:
                LOG.info("B-roll via Pexels: %s", query)
                return files[0]["link"]
        except Exception as e:  # noqa: BLE001
            LOG.warning("Pexels search failed for %r: %s", query, e)
        return None

    def _pixabay_video_url(self, query: str, orientation: str) -> str | None:
        if not self.pixabay_key:
            return None
        try:
            r = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": self.pixabay_key, "q": query, "per_page": 5, "safesearch": "true"},
                timeout=30,
            )
            r.raise_for_status()
            hits = r.json().get("hits", [])
            if not hits:
                return None
            v = random.choice(hits[: min(3, len(hits))])
            sizes = v.get("videos", {})
            for key in ("medium", "large", "small", "tiny"):
                if key in sizes and sizes[key].get("url"):
                    LOG.info("B-roll via Pixabay: %s", query)
                    return sizes[key]["url"]
        except Exception as e:  # noqa: BLE001
            LOG.warning("Pixabay search failed for %r: %s", query, e)
        return None

    # ----- MUSIC -----
    def find_music(self, mood: str = "chill") -> bytes | None:
        if not self.pixabay_key:
            return None
        try:
            r = requests.get(
                "https://pixabay.com/api/",
                params={"key": self.pixabay_key, "q": mood, "per_page": 5},
                timeout=30,
            )
            # Pixabay /api/ is for images; Pixabay music API isn't public — fall back to local assets.
            # We just return None here so assemble.py picks a packaged track.
        except Exception:
            pass
        return None


def _download(url: str) -> bytes:
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    return r.content
