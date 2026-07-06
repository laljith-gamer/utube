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

        LOG.info(
            "TTS via Camb.ai (model=%s, voice_id=%s)",
            payload.get("speech_model"),
            payload.get("voice_id"),
        )
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

        data = b"".join(chunks)
        if not data:
            raise RuntimeError("Camb.ai returned empty audio")
        return data

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
