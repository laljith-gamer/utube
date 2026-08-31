"""Audio validation — deterministic leakage and fidelity checks.

Compares the Whisper ASR transcript of the synthesized audio against
the intended script and the TTS reference text to detect hallucinated leakage.
"""
from __future__ import annotations

import logging
import re
from typing import Any

LOG = logging.getLogger("utube.audio_validation")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def validate_audio(script: dict[str, Any], asr_text: str, ref_text: str) -> None:
    """Verify that the ASR text matches the script and doesn't leak the reference.

    Raises:
        ValueError if the audio fails fidelity or leakage bounds.
    """
    if not asr_text or not asr_text.strip():
        raise ValueError("ASR returned empty text. Audio synthesis may have failed.")

    # 1. Build the full intended script text
    parts = []
    if script.get("hook"):
        parts.append(str(script["hook"]))
    for scene in script.get("scenes", []):
        parts.append(str(scene.get("narration", "")))
    if script.get("cta"):
        parts.append(str(script["cta"]))
    script_text = " ".join(parts)

    norm_script = _normalize(script_text)
    norm_asr = _normalize(asr_text)
    norm_ref = _normalize(ref_text)

    script_words = norm_script.split()
    asr_words = norm_asr.split()

    if not script_words:
        return  # Nothing to validate

    # 2. Check fidelity (word coverage)
    # Ensure at least 70% of the script words appear in the ASR output.
    # Whisper might misspell some, but 70% is a safe lower bound for a match.
    script_vocab = set(script_words)
    asr_vocab = set(asr_words)
    coverage = len(script_vocab & asr_vocab) / max(1, len(script_vocab))
    
    LOG.info("Audio ASR fidelity: %.1f%% word coverage", coverage * 100)
    if coverage < 0.70:
        raise ValueError(f"Audio validation failed: ASR fidelity too low ({coverage*100:.1f}%). Expected script was not spoken.")

    # 3. Check for reference text leakage (Hallucination)
    # F5-TTS sometimes loops or injects the reference text into the output.
    # We look for 4-grams from the reference text that appear in the ASR,
    # but which are NOT part of the intended script.
    ref_words = norm_ref.split()
    if len(ref_words) >= 4:
        ref_4grams = _ngrams(ref_words, 4)
        asr_4grams = _ngrams(asr_words, 4)
        script_4grams = _ngrams(script_words, 4)

        # What 4-grams leaked from ref to asr?
        leaked = (ref_4grams & asr_4grams) - script_4grams

        if leaked:
            leaked_phrases = [" ".join(g) for g in leaked]
            LOG.error("Audio validation FAILED. Leaked reference text: %r", leaked_phrases)
            raise ValueError(f"Audio validation failed: TTS leaked reference text into output: {leaked_phrases}")

    LOG.info("Audio validation PASSED.")
