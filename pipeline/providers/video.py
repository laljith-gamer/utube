"""SVD (Stable Video Diffusion) router — config-driven."""
from __future__ import annotations

import base64
import logging
import time
from typing import Any

import requests

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.video")


class VideoRouter:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg.get_path("video", {}) or {}
        # video config in providers.yaml is keyed `video:` at top level too —
        # we want the providers section, not pipeline.video. Re-resolve via path.
        # Convention: providers.yaml uses `video:` for SVD provider config and
        # pipeline.yaml uses `video:` for output specs. The deep-merge means both
        # end up under `video`. Provider chain lives under `video.chain`.
        self.chain: list[str] = self.cfg.get("chain", []) or []
        self.providers: dict[str, dict[str, Any]] = self.cfg.get("providers", {}) or {}
        self.timeout = self.cfg.get("request_timeout_sec", 300)

    def animate(self, image_png: bytes) -> bytes:
        """Return MP4 bytes of an ~4s clip animated from the given image."""
        errors: list[str] = []
        for name in self.chain:
            p = self.providers.get(name)
            if not p:
                continue
            try:
                if name == "nvidia_nim_svd":
                    return self._nim_svd(p, image_png)
                LOG.warning("Unknown video provider in chain: %s", name)
            except Exception as e:  # noqa: BLE001
                LOG.warning("Video provider %s failed: %s", name, e)
                errors.append(f"{name}:{e}")
        raise RuntimeError(f"All video providers failed: {errors}")

    def _nim_svd(self, p: dict, image_png: bytes) -> bytes:
        api_key = env(p.get("api_key_env", ""))
        if not api_key:
            raise RuntimeError("NIM SVD key not set")
        params = p.get("params", {}) or {}
        b64 = base64.b64encode(image_png).decode("ascii")
        payload = {
            "image": f"data:image/png;base64,{b64}",
            "cfg_scale": params.get("cfg_scale", 1.8),
            "seed": params.get("seed", 0),
            "motion_bucket_id": params.get("motion_bucket_id", 127),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = requests.post(p["url"], json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        video_b64 = data.get("video") or data.get("video_base64") or ""
        if not video_b64:
            artifacts = data.get("artifacts") or []
            if artifacts:
                video_b64 = artifacts[0].get("base64", "")
        if not video_b64:
            raise RuntimeError(f"NIM SVD returned no video: {str(data)[:200]}")
        LOG.info("SVD clip via NVIDIA NIM")
        return base64.b64decode(video_b64)


