"""TTS router — config-driven (edge-tts primary, NIM Magpie premium)."""
from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

import requests

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.tts")


class TTSRouter:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg.get_path("tts", {}) or {}
        self.chain: list[str] = self.cfg.get("chain", []) or []
        self.providers: dict[str, dict[str, Any]] = self.cfg.get("providers", {}) or {}
        self.timeout = self.cfg.get("request_timeout_sec", 120)
        self.default_rate = self.cfg.get("default_rate", "+0%")
        self.default_pitch = self.cfg.get("default_pitch", "+0Hz")

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        rate: str | None = None,
        pitch: str | None = None,
        prefer_premium: bool = False,
    ) -> bytes:
        rate = rate or self.default_rate
        pitch = pitch or self.default_pitch

        # Optionally bias the chain to put premium first
        chain = list(self.chain)
        if prefer_premium and "nim_magpie" in chain:
            chain.remove("nim_magpie")
            chain.insert(0, "nim_magpie")

        last_err: Exception | None = None
        for name in chain:
            p = self.providers.get(name)
            if not p:
                continue
            try:
                if p.get("backend") == "edge_tts":
                    return self._edge_tts(text, voice, rate, pitch)
                if name == "nim_magpie":
                    return self._nim_magpie(p, text, voice)
                LOG.warning("Unknown TTS provider in chain: %s", name)
            except Exception as e:  # noqa: BLE001
                LOG.warning("TTS provider %s failed: %s", name, e)
                last_err = e
        raise RuntimeError(f"All TTS providers failed. Last error: {last_err}")

    def _edge_tts(self, text: str, voice: str, rate: str, pitch: str) -> bytes:
        import edge_tts  # lazy import

        async def _gen() -> bytes:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = Path(f.name)
            try:
                await comm.save(str(tmp))
                return tmp.read_bytes()
            finally:
                tmp.unlink(missing_ok=True)

        LOG.info("TTS via edge-tts (%s)", voice)
        return asyncio.run(_gen())

    def _nim_magpie(self, p: dict, text: str, voice: str) -> bytes:
        api_key = env(p.get("api_key_env", ""))
        if not api_key:
            raise RuntimeError("NIM Magpie key not set")
        params = p.get("params", {}) or {}
        # Edge-style voice ids don't apply to Magpie; use default unless caller passed a magpie voice
        magpie_voice = voice if voice and voice.startswith("Magpie-") else params.get("default_voice", "Magpie-Multilingual.EN-US.Ray")
        payload = {
            "text": text,
            "voice": magpie_voice,
            "encoding": params.get("encoding", "mp3"),
            "sample_rate_hz": params.get("sample_rate_hz", 22050),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        r = requests.post(p["url"], json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        audio_b64 = data.get("audio") or data.get("audio_base64") or ""
        if not audio_b64:
            raise RuntimeError(f"NIM Magpie returned no audio: {str(data)[:200]}")
        LOG.info("TTS via NIM Magpie")
        return base64.b64decode(audio_b64)
