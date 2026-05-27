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


# Edge-tts is strict about characters in its SSML payload. Anything outside
# basic ASCII or any of the Unicode "lookalike" punctuation tends to make
# Microsoft's TTS service silently emit zero audio chunks ("No audio was
# received."). We aggressively normalize:
#
#   1. Map every known typographic dash / quote / space variant to its
#      ASCII equivalent. The big trap caught in production was U+2011
#      (NON-BREAKING HYPHEN) showing up in topic titles like "7‑Eleven".
#   2. After NFKD decomposition, drop any remaining non-ASCII byte. We
#      only synthesise English voices in this pipeline, so ASCII-only is
#      safe and bulletproof.
_SMART_QUOTE_TABLE = str.maketrans({
    # Quotes — single
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    # Quotes — double
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    # Guillemets
    "\u00AB": '"', "\u00BB": '"',
    # Hyphens / dashes (the U+2011 case is what bit us in run #14)
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2015": "-",  # HORIZONTAL BAR
    "\u2043": "-",  # HYPHEN BULLET
    "\u2212": "-",  # MINUS SIGN
    "\uFE58": "-",  # SMALL EM DASH
    "\uFE63": "-",  # SMALL HYPHEN-MINUS
    "\uFF0D": "-",  # FULLWIDTH HYPHEN-MINUS
    # Spaces / formatting
    "\u00A0": " ",  # NO-BREAK SPACE
    "\u2007": " ",  # FIGURE SPACE
    "\u2009": " ",  # THIN SPACE
    "\u200A": " ",  # HAIR SPACE
    "\u200B": "",   # ZERO WIDTH SPACE
    "\u200C": "",   # ZERO WIDTH NON-JOINER
    "\u200D": "",   # ZERO WIDTH JOINER
    "\uFEFF": "",   # BYTE ORDER MARK
    # Misc
    "\u2026": "...",
    "\u00B7": ".",
})


def _sanitize_text(text: str) -> str:
    """Make text safe for SSML transport.

    1. NFKC normalize compatibility forms (full-width digits, ligatures, etc.).
    2. Translate known typographic punctuation -> ASCII.
    3. Drop control characters.
    4. NFKD decompose accented chars and strip the resulting non-ASCII parts
       so what remains is plain ASCII (safe for edge-tts).
    5. Collapse whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_SMART_QUOTE_TABLE)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # NFKD decomposes "é" -> "e" + combining accent; ASCII-encode then drops
    # the combining accent (and anything else outside ASCII) cleanly.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Collapse runs of whitespace into single spaces.
    text = re.sub(r"\s+", " ", text)
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


_STRICT_KEEP = re.compile(r"[A-Za-z0-9\s.,!?'\"$%&\-:;()]")


def _strict_sanitize(text: str) -> str:
    """Last-resort sanitiser: keep ONLY a small ASCII subset. Used when
    edge-tts has already rejected the normally-sanitized text."""
    text = _sanitize_text(text)
    return re.sub(r"\s+", " ", "".join(ch if _STRICT_KEEP.match(ch) else " " for ch in text)).strip()


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
        """Try with configured rate/pitch; on 'No audio was received', escalate
        through two recovery steps before giving up:
            1. retry with rate=+0%, pitch=+0Hz (safe defaults)
            2. retry with text re-sanitized to a strict ASCII-letters/digits/
               basic-punct subset and rate/pitch defaults
        """
        try:
            return self._edge_tts(text, voice, rate, pitch)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            transient = ("no audio" in msg or "parameters" in msg or "empty" in msg)
            if not transient:
                raise
            # Step 1: same text, safe rate/pitch
            if (rate, pitch) != ("+0%", "+0Hz"):
                LOG.warning(
                    "edge-tts rejected rate=%s pitch=%s — retrying with safe defaults",
                    rate, pitch,
                )
                try:
                    return self._edge_tts(text, voice, "+0%", "+0Hz")
                except Exception as e2:  # noqa: BLE001
                    msg = str(e2).lower()
                    transient = ("no audio" in msg or "parameters" in msg or "empty" in msg)
                    if not transient:
                        raise
            # Step 2: stricter sanitization (drops anything outside [\w\s.,!?'"$%-])
            stricter = _strict_sanitize(text)
            if stricter and stricter != text:
                LOG.warning(
                    "edge-tts still rejecting; retrying with stricter ASCII text "
                    "(%d -> %d chars)", len(text), len(stricter),
                )
                return self._edge_tts(stricter, voice, "+0%", "+0Hz")
            raise RuntimeError(
                "edge-tts rejected the text even after defaults + strict sanitize. "
                "First 120 chars: " + repr(text[:120])
            )

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
