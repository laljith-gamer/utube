"""Image-generation router — config-driven, no hardcoded URLs/models/params."""
from __future__ import annotations

import base64
import logging
import time
from typing import Any
from urllib.parse import quote

import requests

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.image")


class ImageRouter:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg.get_path("image", {}) or {}
        self.chain: list[str] = self.cfg.get("chain", [])
        self.providers: dict[str, dict[str, Any]] = self.cfg.get("providers", {}) or {}
        self.timeout = self.cfg.get("request_timeout_sec", 120)
        self.default_negative = self.cfg.get("default_negative_prompt", "")

    def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        negative: str | None = None,
    ) -> bytes:
        negative = negative if negative is not None else self.default_negative
        errors: list[str] = []
        for name in self.chain:
            p = self.providers.get(name)
            if not p:
                continue
            try:
                if name == "nvidia_nim_sdxl":
                    return self._nim_sdxl(p, prompt, width, height, negative)
                if name == "pollinations":
                    return self._pollinations(p, prompt, width, height)
                LOG.warning("Unknown image provider in chain: %s", name)
            except Exception as e:  # noqa: BLE001
                LOG.warning("Image provider %s failed: %s", name, e)
                errors.append(f"{name}:{e}")
        raise RuntimeError(f"All image providers failed: {errors}")

    # ------- providers --------

    def _nim_sdxl(self, p: dict, prompt: str, w: int, h: int, neg: str) -> bytes:
        api_key = env(p.get("api_key_env", ""))
        if not api_key:
            raise RuntimeError("NIM SDXL key not set")
        params = p.get("params", {}) or {}
        payload = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": neg, "weight": -1.0},
            ],
            "cfg_scale": params.get("cfg_scale", 5),
            "sampler": params.get("sampler", "K_DPM_2_ANCESTRAL"),
            "seed": params.get("seed", 0),
            "steps": params.get("steps", 25),
            "width": w,
            "height": h,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = requests.post(p["url"], json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        artifacts = data.get("artifacts") or data.get("images") or []
        if not artifacts:
            raise RuntimeError(f"NIM returned no artifacts: {str(data)[:200]}")
        b64 = artifacts[0].get("base64") or artifacts[0].get("b64_json") or ""
        if not b64:
            raise RuntimeError("NIM artifact missing base64")
        LOG.info("Image via NVIDIA NIM SDXL")
        return base64.b64decode(b64)

    def _pollinations(self, p: dict, prompt: str, w: int, h: int) -> bytes:
        import time
        import random
        url_template = p.get("url_template", "")
        # Add random seed to avoid caching issues on Pollinations
        seed = random.randint(1, 99999)
        url = url_template.format(prompt=quote(prompt), w=w, h=h) + f"&seed={seed}"
        
        for attempt in range(4):
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 429 and attempt < 3:
                wait = 2 ** attempt + random.random()
                LOG.warning("Pollinations 429, backing off %.1fs", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            min_bytes = p.get("min_response_bytes", 1000)
            if not r.content or len(r.content) < min_bytes:
                raise RuntimeError(f"Pollinations returned tiny payload: {len(r.content)} bytes")
            LOG.info("Image via Pollinations.ai")
            return r.content
        raise RuntimeError("Pollinations failed after retries")


