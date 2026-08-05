"""TTS router - config-driven multi-provider chain with full error visibility.

Providers
---------
camb   : Camb.ai streaming TTS. High-quality MARS voices with instruction-based
         delivery control. Requires CAMB_API_KEY.
edge   : Microsoft Edge TTS via the edge-tts python lib. No API key. High quality
         but flaky from cloud runners (~5-10% empty-response rate).
gtts   : Google Translate TTS via the gTTS python lib. No API key. Lower quality
         voice but extremely reliable; used as the no-network-key fallback.

Optional speed control is applied after synthesis via ffmpeg's `atempo` filter
(see pipeline/stages/audio.py). Camb should generally use native
`voice_settings.speaking_rate`; Edge and gTTS stay at neutral provider speed.

If you re-introduce another paid provider (NVIDIA NIM Magpie etc.), follow the
existing `_xxx()` pattern: synthesize at natural speed and return audio bytes.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from ..config import get_config
from ..utils import env

LOG = logging.getLogger("utube.tts")

# Per-request character ceiling enforced client-side. Some Camb.ai plans cap
# each request (e.g. the free tier at 500); we split the text ourselves and
# concatenate the resulting audio so a long script still synthesizes in one
# logical utterance without exceeding the per-request quota.
CAMB_CHUNK_CHAR_LIMIT = 480


class TTSRouter:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg.get_path("tts", {}) or {}
        self.chain: list[str] = self.cfg.get("chain", []) or []
        self.providers: dict[str, dict[str, Any]] = self.cfg.get("providers", {}) or {}
        self.timeout = self.cfg.get("request_timeout_sec", 120)

    def synthesize(self, text: str, *, voice: str, prefer_premium: bool = False) -> bytes:
        """Try each provider in order; return audio bytes from the first success.

        Providers should return natural, unmodified audio. The audio stage
        handles format normalization and optional post-processing.
        """
        chain = list(self.chain)
        if prefer_premium and "nim_magpie" in chain:
            chain.remove("nim_magpie")
            chain.insert(0, "nim_magpie")

        errors: list[str] = []
        for name in chain:
            p = self.providers.get(name) or {}
            try:
                backend = p.get("backend") or name
                if backend == "elevenlabs":
                    return self._elevenlabs(p, text, voice)
                if backend == "camb_tts" or name == "camb":
                    return self._camb_tts(p, text, voice)
                if backend == "edge_tts":
                    return self._edge_tts(text, voice)
                if backend == "gtts":
                    return self._gtts(text, voice, p.get("params", {}) or {})
                if name == "nim_magpie":
                    return self._nim_magpie(p, text, voice)
                LOG.warning("Unknown TTS provider in chain: %s (backend=%s)", name, backend)
                errors.append(f"{name}: unknown backend {backend!r}")
            except Exception as e:  # noqa: BLE001
                msg = f"{name}: {e}"
                LOG.warning("TTS provider %s failed: %s", name, e)
                errors.append(msg)

        # Surface every error so the next debugging session doesn't need
        # another round-trip through the workflow logs.
        raise RuntimeError("All TTS providers failed.\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------------ elevenlabs

    def _elevenlabs(self, p: dict, text: str, voice: str) -> bytes:
        """Synthesize with ElevenLabs TTS API."""
        api_key = env(p.get("api_key_env", ""))
        if not api_key:
            raise RuntimeError(f"ElevenLabs key not set for {p.get('api_key_env')}")

        params = p.get("params", {}) or {}
        # The slot config might pass an Edge/Azure TTS voice string (like en-US-AvaMultilingualNeural).
        # ElevenLabs voice IDs are exactly 20-character alphanumeric strings without hyphens.
        if not voice or "-" in voice or len(voice) != 20:
            voice_id = params.get("voice_id")
        else:
            voice_id = voice
        
        url = f"{p.get('url', 'https://api.elevenlabs.io/v1/text-to-speech')}/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        }

        payload = {
            "text": text,
            "model_id": params.get("model_id", "eleven_multilingual_v2"),
        }
        
        if "voice_settings" in params:
            payload["voice_settings"] = params["voice_settings"]

        LOG.info("TTS via ElevenLabs (voice_id=%s)", voice_id)
        
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        
        if r.status_code >= 400:
            body = (r.text or "").strip()
            # Status 401 usually means quota exceeded (or invalid key)
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {body[:500]}")
            
        data = r.content
        if not data:
            raise RuntimeError("ElevenLabs returned empty audio")
            
        return data

    # ------------------------------------------------------------------ camb

    def _camb_tts(self, p: dict, text: str, voice: str) -> bytes:
        """Synthesize with Camb.ai's streaming TTS endpoint.

        The narration script is passed as `text` exactly as received. Human
        delivery guidance belongs in Camb's `user_instructions`, which lets us
        shape pacing and emphasis without inserting tags or rewriting words.
        """
        api_key = env(p.get("api_key_env", ""))
        if not api_key:
            raise RuntimeError("Camb.ai key not set")

        params = p.get("params", {}) or {}
        speech_model = str(params.get("speech_model", "mars-instruct"))
        payload: dict[str, Any] = {
            "text": text,
            "language": str(params.get("language") or self._camb_language_from_voice(voice)).lower(),
            "voice_id": self._camb_voice_id(voice, params),
            "speech_model": speech_model,
        }

        user_instructions = params.get("user_instructions")
        if user_instructions and speech_model == "mars-instruct":
            payload["user_instructions"] = str(user_instructions)

        output_configuration = params.get("output_configuration")
        if not output_configuration and params.get("output_format"):
            output_configuration = {"format": params.get("output_format")}
        if output_configuration:
            payload["output_configuration"] = output_configuration

        for key in (
            "voice_settings",
            "inference_options",
            "enhance_named_entities_pronunciation",
        ):
            if key in params:
                payload[key] = params[key]

        payload = self._drop_none(payload)
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        url = p.get("url", "https://client.camb.ai/apis/tts-stream")

        # Split long text so each request stays under the plan's per-request
        # character ceiling (e.g. the free tier caps at 500). We split on
        # sentence/phrase boundaries so delivery stays natural across seams.
        text_chunks = self._camb_chunk_text(text, CAMB_CHUNK_CHAR_LIMIT)
        LOG.info(
            "TTS via Camb.ai (model=%s, voice_id=%s, chunks=%d, chars=%d)",
            payload.get("speech_model"),
            payload.get("voice_id"),
            len(text_chunks),
            len(text or ""),
        )

        audio_parts: list[bytes] = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_payload = dict(payload)
            chunk_payload["text"] = chunk_text
            audio_parts.append(self._camb_request(url, chunk_payload, headers, index=i))
            # Be gentle between requests so we don't trip rate limits.
            if i < len(text_chunks) - 1:
                time.sleep(1.0)

        data = b"".join(audio_parts)
        if not data:
            raise RuntimeError("Camb.ai returned empty audio")
        return data

    def _camb_request(
        self, url: str, payload: dict, headers: dict, *, index: int
    ) -> bytes:
        """One streaming Camb.ai request. Returns raw audio bytes."""
        with requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=self.timeout,
        ) as r:
            if r.status_code >= 400:
                body = (r.text or "").strip()
                raise RuntimeError(f"Camb.ai HTTP {r.status_code}: {body[:500]}")
            chunks = [chunk for chunk in r.iter_content(chunk_size=64 * 1024) if chunk]
        if not chunks:
            raise RuntimeError(f"Camb.ai returned empty audio for chunk {index}")
        return b"".join(chunks)

    @staticmethod
    def _camb_chunk_text(text: str, limit: int) -> list[str]:
        """Split text into <=limit-char pieces at sentence/phrase boundaries.

        Keeps punctuation with its sentence. A single sentence longer than
        `limit` is hard-split on whitespace so we never send an over-limit
        request regardless of input.
        """
        if not text or not text.strip():
            return [text] if text else []
        text = text.strip()
        if limit <= 0 or len(text) <= limit:
            return [text]

        # Greedy packing: accumulate sentences until adding the next would
        # exceed the limit, then flush. Sentence split on . ! ? followed by
        # optional quotes/space and keep the delimiter.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        cur = ""
        for sent in sentences:
            if not sent:
                continue
            # A single sentence over the limit gets hard-wrapped on words.
            if len(sent) > limit:
                if cur:
                    chunks.append(cur)
                    cur = ""
                words = sent.split(" ")
                buf = ""
                for w in words:
                    # A token with no breakable spaces (e.g. a long URL) still
                    # must be split so no request exceeds the ceiling.
                    if len(w) > limit:
                        if buf:
                            chunks.append(buf)
                            buf = ""
                        for i in range(0, len(w), limit):
                            chunks.append(w[i:i + limit])
                        continue
                    if buf and len(buf) + 1 + len(w) > limit:
                        chunks.append(buf)
                        buf = w
                    else:
                        buf = (buf + " " + w).strip()
                if buf:
                    chunks.append(buf)
                continue
            if cur and len(cur) + 1 + len(sent) > limit:
                chunks.append(cur)
                cur = sent
            else:
                cur = (cur + " " + sent).strip()
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def _camb_voice_id(voice: str, params: dict) -> int:
        for candidate in (voice, params.get("voice_id"), params.get("default_voice_id")):
            if candidate is None:
                continue
            s = str(candidate).strip()
            for prefix in ("camb:", "voice_id:"):
                if s.lower().startswith(prefix):
                    s = s[len(prefix):].strip()
            if s.isdigit():
                return int(s)
        raise RuntimeError("Camb.ai voice_id is not configured")

    @staticmethod
    def _camb_language_from_voice(voice: str) -> str:
        if not voice:
            return "en-us"
        parts = voice.split("-")
        if len(parts) >= 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
            return f"{parts[0]}-{parts[1]}".lower()
        return "en-us"

    @classmethod
    def _drop_none(cls, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for k, v in value.items():
                if v is None:
                    continue
                nested = cls._drop_none(v)
                if nested != {}:
                    cleaned[k] = nested
            return cleaned
        return value

    # ------------------------------------------------------------------ edge

    def _edge_tts(self, text: str, voice: str) -> bytes:
        """Synthesize with Microsoft Edge TTS at neutral rate/pitch.

        Sending non-zero rate/pitch over SSML triggers extra rejections from
        Microsoft's endpoint when called from cloud-runner IPs, so we always
        request neutral and let ffmpeg atempo apply the speed-up downstream.
        """
        import edge_tts  # lazy import

        async def _gen() -> bytes:
            comm = edge_tts.Communicate(text, voice, rate="+0%", pitch="+0Hz")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = Path(f.name)
            try:
                await comm.save(str(tmp))
                data = tmp.read_bytes()
                if not data:
                    raise RuntimeError("edge-tts returned empty audio")
                return data
            finally:
                tmp.unlink(missing_ok=True)

        LOG.info("TTS via edge-tts (%s)", voice)
        return asyncio.run(_gen())

    # ------------------------------------------------------------------ gtts

    def _gtts(self, text: str, voice: str, params: dict) -> bytes:
        """Google Translate TTS — robust no-key fallback.

        gTTS has no rate parameter; post-processing can apply a configured rate.
        Voice is mapped from Edge-style IDs to gtts (lang, tld) pairs:
            en-US-XxxNeural -> lang='en', tld='com'
            en-GB-XxxNeural -> lang='en', tld='co.uk'
            en-AU-XxxNeural -> lang='en', tld='com.au'
            en-IN-XxxNeural -> lang='en', tld='co.in'
        """
        from gtts import gTTS  # lazy import

        lang, tld = self._gtts_voice_map(voice)
        slow = bool(params.get("slow", False))

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = Path(f.name)
        try:
            tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
            tts.save(str(tmp))
            data = tmp.read_bytes()
            if not data:
                raise RuntimeError("gtts returned empty audio")
            LOG.info("TTS via gtts (lang=%s, tld=%s)", lang, tld)
            return data
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _gtts_voice_map(voice: str) -> tuple[str, str]:
        if not voice:
            return ("en", "com")
        v = voice.lower()
        for prefix, tld in (
            ("en-us", "com"),
            ("en-gb", "co.uk"),
            ("en-au", "com.au"),
            ("en-in", "co.in"),
            ("en-ca", "ca"),
            ("en-ie", "ie"),
        ):
            if v.startswith(prefix):
                return ("en", tld)
        # Fall through: take the language part (e.g. 'fr-FR-...' -> 'fr')
        lang = voice.split("-", 1)[0].lower() or "en"
        return (lang, "com")

    # ------------------------------------------------------------------ nim

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
        r.raise_for_status()
        data = r.json()
        audio_b64 = data.get("audio") or data.get("audio_base64") or ""
        if not audio_b64:
            raise RuntimeError(f"NIM Magpie returned no audio: {str(data)[:200]}")
        LOG.info("TTS via NIM Magpie")
        return base64.b64decode(audio_b64)
