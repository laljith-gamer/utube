"""SVD (Stable Video Diffusion) router.

Primary:  NVIDIA NIM stable-video-diffusion (~4s clip from one image)
Fallback: HuggingFace Inference (SVD-XT-1.1)

Both take a conditioning image and return MP4 bytes.
"""
from __future__ import annotations

import base64
import logging
import time

import requests

from ..utils import env

LOG = logging.getLogger("utube.video")


class VideoRouter:
    def __init__(self) -> None:
        self.nim_key = env("NVIDIA_NIM_API_KEY")
        self.hf_key = env("HUGGINGFACE_API_KEY")

    def animate(self, image_png: bytes, *, motion_bucket_id: int = 127, seed: int = 0) -> bytes:
        """Return MP4 bytes of an ~4s clip animated from the given image."""
        errors: list[str] = []

        if self.nim_key:
            try:
                return self._nim_svd(image_png, motion_bucket_id, seed)
            except Exception as e:  # noqa: BLE001
                LOG.warning("NIM SVD failed: %s", e)
                errors.append(f"nim:{e}")

        if self.hf_key:
            try:
                return self._hf_svd(image_png)
            except Exception as e:  # noqa: BLE001
                LOG.warning("HF SVD failed: %s", e)
                errors.append(f"hf:{e}")

        raise RuntimeError(f"All video providers failed: {errors}")

    def _nim_svd(self, image_png: bytes, motion_bucket_id: int, seed: int) -> bytes:
        url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-video-diffusion"
        headers = {
            "Authorization": f"Bearer {self.nim_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        b64 = base64.b64encode(image_png).decode("ascii")
        payload = {
            "image": f"data:image/png;base64,{b64}",
            "cfg_scale": 1.8,
            "seed": seed,
            "motion_bucket_id": motion_bucket_id,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=300)
        r.raise_for_status()
        data = r.json()
        # NIM SVD returns video in 'video' key as base64 mp4
        video_b64 = data.get("video") or data.get("video_base64") or ""
        if not video_b64:
            artifacts = data.get("artifacts") or []
            if artifacts:
                video_b64 = artifacts[0].get("base64", "")
        if not video_b64:
            raise RuntimeError(f"NIM SVD returned no video: {str(data)[:200]}")
        LOG.info("SVD clip via NVIDIA NIM")
        return base64.b64decode(video_b64)

    def _hf_svd(self, image_png: bytes) -> bytes:
        model = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {
            "Authorization": f"Bearer {self.hf_key}",
            "Content-Type": "image/png",
        }
        for attempt in range(2):
            r = requests.post(url, data=image_png, headers=headers, timeout=300)
            if r.status_code == 503 and attempt == 0:
                time.sleep(30)
                continue
            r.raise_for_status()
            LOG.info("SVD clip via HuggingFace")
            return r.content
        raise RuntimeError("HF SVD timed out")
