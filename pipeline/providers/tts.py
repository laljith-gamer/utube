"""TTS provider router.

Primary:  edge-tts  (unlimited, no key)
Fallback: NVIDIA NIM Magpie TTS (premium voices, trial credits)

Returns MP3 bytes.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import requests

from ..utils import env

LOG = logging.getLogger("utube.tts")


class TTSRouter:
    def __init__(self) -> None:
        self.nim_key = env("NVIDIA_NIM_API_KEY")

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "en-US-AriaNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        prefer_premium: bool = False,
    ) -> bytes:
        if prefer_premium and self.nim_key:
            try:
                return self._nim_magpie(text, voice)
            except Exception as e:  # noqa: BLE001
                LOG.warning("NIM Magpie failed, falling back to edge-tts: %s", e)

        try:
            return self._edge_tts(text, voice, rate, pitch)
        except Exception as e:  # noqa: BLE001
            LOG.warning("edge-tts failed: %s", e)
            if self.nim_key:
                return self._nim_magpie(text, voice)
            raise

    def _edge_tts(self, text: str, voice: str, rate: str, pitch: str) -> bytes:
        # Lazy import so the module loads even if edge-tts isn't installed yet
        import edge_tts

        async def _gen() -> bytes:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = Path(f.name)
            try:
                await communicate.save(str(tmp))
                return tmp.read_bytes()
            finally:
                tmp.unlink(missing_ok=True)

        LOG.info("TTS via edge-tts (%s)", voice)
        return asyncio.run(_gen())

    def _nim_magpie(self, text: str, voice: str) -> bytes:
        # Magpie multilingual via NIM. Voices are model-defined; pass through.
        url = "https://ai.api.nvidia.com/v1/genai/nvidia/magpie-tts-multilingual"
        headers = {
            "Authorization": f"Bearer {self.nim_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice": voice if not voice.startswith("en-US-") else "Magpie-Multilingual.EN-US.Ray",
            "encoding": "mp3",
            "sample_rate_hz": 22050,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        # NIM returns audio as base64 in 'audio' field
        import base64

        data = r.json()
        audio_b64 = data.get("audio") or data.get("audio_base64") or ""
        if not audio_b64:
            raise RuntimeError(f"NIM Magpie returned no audio: {str(data)[:200]}")
        LOG.info("TTS via NIM Magpie")
        return base64.b64decode(audio_b64)
