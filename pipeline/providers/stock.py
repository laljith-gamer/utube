"""Stock B-roll router — Pexels + Pixabay, fully config-driven."""
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
        self.used_ids: set[str] = set()

    def find_video(self, keywords: list[str], *, orientation: str = "portrait") -> bytes | None:
        for kw in keywords:
            for name in self.chain:
                p = self.providers.get(name)
                if not p:
                    continue
                api_key = env(p.get("api_key_env", ""))
                if not api_key:
                    continue
                try:
                    if name == "pexels":
                        url = self._pexels(p, kw, orientation, api_key)
                    elif name == "pixabay":
                        url = self._pixabay(p, kw, api_key)
                    else:
                        continue
                    if url:
                        return self._download(url)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("Stock provider %s failed for %r: %s", name, kw, e)
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
        videos = [v for v in videos if str(v.get("id")) not in self.used_ids]
        if not videos:
            return None
        v = random.choice(videos[: min(3, len(videos))])
        self.used_ids.add(str(v.get("id")))
        target_w = 1080 if orientation == "portrait" else 1920
        files = sorted(
            [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"],
            key=lambda f: abs((f.get("width") or 0) - target_w),
        )
        if files:
            LOG.info("B-roll via Pexels: %s", query)
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
        hits = [h for h in hits if str(h.get("id")) not in self.used_ids]
        if not hits:
            return None
        v = random.choice(hits[: min(3, len(hits))])
        self.used_ids.add(str(v.get("id")))
        for size_key in ("medium", "large", "small", "tiny"):
            sized = v.get("videos", {}).get(size_key, {})
            if sized.get("url"):
                LOG.info("B-roll via Pixabay: %s", query)
                return sized["url"]
        return None

    def _download(self, url: str) -> bytes:
        r = requests.get(url, timeout=self.timeout, stream=True)
        r.raise_for_status()
        return r.content
