"""Image-generation router.

Primary:  NVIDIA NIM SDXL
Fallback: Pollinations.ai  (no API key, unlimited)
Fallback: HuggingFace Inference (SDXL / FLUX-schnell)

Returns raw PNG bytes.
"""
from __future__ import annotations

import base64
import logging
import time
from urllib.parse import quote

import requests

from ..utils import env

LOG = logging.getLogger("utube.image")


class ImageRouter:
    def __init__(self) -> None:
        self.nim_key = env("NVIDIA_NIM_API_KEY")
        self.hf_key = env("HUGGINGFACE_API_KEY")

    def generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        negative: str = "low quality, blurry, watermark, text, signature",
    ) -> bytes:
        errors: list[str] = []

        if self.nim_key:
            try:
                return self._nim_sdxl(prompt, width, height, negative)
            except Exception as e:  # noqa: BLE001
                LOG.warning("NIM SDXL failed: %s", e)
                errors.append(f"nim:{e}")

        # Pollinations is keyless, always try
        try:
            return self._pollinations(prompt, width, height)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Pollinations failed: %s", e)
            errors.append(f"pollinations:{e}")

        if self.hf_key:
            try:
                return self._huggingface(prompt, width, height)
            except Exception as e:  # noqa: BLE001
                LOG.warning("HuggingFace failed: %s", e)
                errors.append(f"hf:{e}")

        raise RuntimeError(f"All image providers failed: {errors}")

    def _nim_sdxl(self, prompt: str, w: int, h: int, neg: str) -> bytes:
        url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"
        headers = {
            "Authorization": f"Bearer {self.nim_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": neg, "weight": -1.0},
            ],
            "cfg_scale": 5,
            "sampler": "K_DPM_2_ANCESTRAL",
            "seed": 0,
            "steps": 25,
            "width": w,
            "height": h,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        # NIM returns base64 in artifacts
        artifacts = data.get("artifacts") or data.get("images") or []
        if not artifacts:
            raise RuntimeError(f"NIM returned no artifacts: {str(data)[:200]}")
        b64 = artifacts[0].get("base64") or artifacts[0].get("b64_json") or ""
        if not b64:
            raise RuntimeError("NIM artifact missing base64")
        LOG.info("Image via NVIDIA NIM SDXL")
        return base64.b64decode(b64)

    def _pollinations(self, prompt: str, w: int, h: int) -> bytes:
        # No API key, returns image bytes directly
        url = (
            f"https://image.pollinations.ai/prompt/{quote(prompt)}"
            f"?width={w}&height={h}&nologo=true&enhance=true"
        )
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if not r.content or len(r.content) < 1000:
            raise RuntimeError(f"Pollinations returned tiny payload: {len(r.content)} bytes")
        LOG.info("Image via Pollinations.ai")
        return r.content

    def _huggingface(self, prompt: str, w: int, h: int) -> bytes:
        # FLUX-schnell is fast; SDXL also works
        model = "black-forest-labs/FLUX.1-schnell"
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self.hf_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {"width": w, "height": h, "num_inference_steps": 4},
        }
        # HF cold-starts; retry once
        for attempt in range(2):
            r = requests.post(url, json=payload, headers=headers, timeout=180)
            if r.status_code == 503 and attempt == 0:
                time.sleep(20)
                continue
            r.raise_for_status()
            LOG.info("Image via HuggingFace (%s)", model)
            return r.content
        raise RuntimeError("HuggingFace timed out")
