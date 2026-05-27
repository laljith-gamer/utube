"""TTS router — fully config-driven.

Chain order from config/providers.yaml > tts.chain. Each provider tried in
order. edge-tts (no key, unlimited) is the primary; if Microsoft's TTS
server rejects the rate/pitch combination, we automatically retry with
safe defaults (+0%/+0Hz) so a quirky per-niche pitch never breaks the run.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import requests

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.tts")


# Edge-tts is strict about characters. These commonly cause "No audio was received".
_SMART_QUOTE_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00AB": '"', "\u00BB": '"',
    "\u2026": "...",
    "\u00A0": " ",
})


def _sanitize_text(text: str) -> str:
    """Make text safe for SSML transport."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_SMART_QUOTE_TABLE)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


_RATE_RE = re.compile(r"^[+-]\d{1,3}%$")
_PITCH_RE = re.compile(r"^[+-]\d{1,3}Hz$")


def _coerce_rate(rate: str | None, default: str = "+0%") -> str:
    if rate and _RATE_RE.match(rate):
        return rate
    return default


def _coerce_pitch(pitch: str | None, default: str = "+0Hz") -> str:
    if pitch and _PITCH_RE.match(pitch):
        return pitch
    return default


class TTSRouter:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg.get_path("tts", {}) or {}
        self.chain: list[str] = self.cfg.get("chain", []) or []
        self.providers: dict[str, dict[str, Any]] = self.cfg.get("providers", {}) or {}
        self.timeout = self.cfg.get("request_timeout_sec", 120)
        self.default_rate = _coerce_rate(self.cfg.get("default_rate"), "+0%")
        self.default_pitch = _coerce_pitch(self.cfg.get("default_pitch"), "+0Hz")

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        rate: str | None = None,
        pitch: str | None = None,
        prefer_premium: bool = False,
    ) -> bytes:
        text = _sanitize_text(text)
        if not text:
            raise ValueError("TTS text is empty after sanitization")
        rate = _coerce_rate(rate, self.default_rate)
        pitch = _coerce_pitch(pitch, self.default_pitch)

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
                    return self._edge_tts_with_retry(text, voice, rate, pitch)
                if name == "nim_magpie":
                    return self._nim_magpie(p, text, voice)
                LOG.warning("Unknown TTS provider in chain: %s", name)
            except Exception as e:  # noqa: BLE001
                LOG.warning("TTS provider %s failed: %s", name, e)
                last_err = e
        raise RuntimeError(f"All TTS providers failed. Last error: {last_err}")

    # ---------- edge-tts ----------

    def _edge_tts_with_retry(self, text: str, voice: str, rate: str, pitch: str) -> bytes:
        """Try with configured rate/pitch; on 'No audio was received' retry with safe defaults."""
        try:
            return self._edge_tts(text, voice, rate, pitch)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if ("no audio" in msg or "parameters" in msg or "empty" in msg) \
                    and (rate, pitch) != ("+0%", "+0Hz"):
                LOG.warning(
                    "edge-tts rejected rate=%s pitch=%s — retrying with safe defaults",
                    rate, pitch,
                )
                return self._edge_tts(text, voice, "+0%", "+0Hz")
            raise

    def _edge_tts(self, text: str, voice: str, rate: str, pitch: str) -> bytes:
        import edge_tts  # lazy import

        async def _gen() -> bytes:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = Path(f.name)
            try:
                await comm.save(str(tmp))
                data = tmp.read_bytes()
                if not data:
                    raise RuntimeError("edge-tts produced empty MP3")
                return data
            finally:
                tmp.unlink(missing_ok=True)

        LOG.info("TTS via edge-tts (%s rate=%s pitch=%s)", voice, rate, pitch)
        return asyncio.run(_gen())

    # ---------- NIM Magpie (best-effort) ----------

    def _nim_magpie(self, p: dict, text: str, voice: str) -> bytes:
        api_key = env(p.get("api_key_env", ""))
        if not api_key:
            raise RuntimeError("NIM Magpie key not set")
        params = p.get("params", {}) or {}
        magpie_voice = (
            voice if voice and voice.startswith("Magpie-")
            else params.get("default_voice", "Magpie-Multilingual.EN-US.Ray")
        )
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
        if r.status_code == 404:
            raise RuntimeError(
                f"NIM Magpie endpoint not found ({p['url']}); "
                "verify the URL in config/providers.yaml > tts.providers.nim_magpie.url"
            )
        r.raise_for_status()
        data = r.json()
        audio_b64 = data.get("audio") or data.get("audio_base64") or ""
        if not audio_b64:
            raise RuntimeError(f"NIM Magpie returned no audio: {str(data)[:200]}")
        LOG.info("TTS via NIM Magpie")
        return base64.b64decode(audio_b64)
