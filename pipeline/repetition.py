"""Deterministic repetition-control layer.

Checks for stylistic repetition both within a single script (intra-video)
and across the recent production history (cross-video).  This is NOT an LLM
call — it is a fast, deterministic text-analysis pass that runs after script
generation and produces a structured report the orchestrator can act on.

Cross-video checking uses the narration archive
(``data/narration_history.json``) to compare against the last N scripts.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

LOG = logging.getLogger("utube.repetition")

# Minimum n-gram length to flag as "distinctive phrase"
MIN_NGRAM = 5
# Top N recent scripts to compare against
DEFAULT_HISTORY_DEPTH = 20
# Ignore generic n-grams shorter than this character count
MIN_PHRASE_CHARS = 20


@dataclass
class RepetitionReport:
    """Result of a repetition check."""

    passed: bool
    intra_issues: list[str] = field(default_factory=list)
    cross_issues: list[str] = field(default_factory=list)
    flagged_phrases: list[str] = field(default_factory=list)

    @property
    def all_issues(self) -> list[str]:
        return self.intra_issues + self.cross_issues


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _word_ngrams(text: str, n: int) -> list[str]:
    """Extract word-level n-grams from normalized text."""
    words = _normalize(text).split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _extract_all_ngrams(text: str, min_n: int = MIN_NGRAM, max_n: int = 8) -> set[str]:
    """Extract distinctive n-grams of various sizes."""
    ngrams: set[str] = set()
    for n in range(min_n, max_n + 1):
        for gram in _word_ngrams(text, n):
            if len(gram) >= MIN_PHRASE_CHARS:
                ngrams.add(gram)
    return ngrams


def _first_word(sentence: str) -> str:
    """Return the first substantive word of a sentence."""
    words = _normalize(sentence).split()
    return words[0] if words else ""


# ── Intra-video checks ──────────────────────────────────────────────────────

def _check_intra_duplicates(
    hook: str, scenes: list[str], cta: str
) -> list[str]:
    """Detect exact or near-duplicate narration across hook/scenes/CTA."""
    issues: list[str] = []
    all_lines = [("hook", hook)] + [(f"scene_{i}", s) for i, s in enumerate(scenes)] + [("cta", cta)]
    norm_map: dict[str, list[str]] = {}
    for label, text in all_lines:
        if not text.strip():
            continue
        key = _normalize(text)
        norm_map.setdefault(key, []).append(label)
    for key, labels in norm_map.items():
        if len(labels) > 1:
            issues.append(f"Exact duplicate narration across {', '.join(labels)}: '{key[:60]}…'")
    return issues


def _check_same_word_starts(scenes: list[str]) -> list[str]:
    """Flag consecutive scenes that start with the same word."""
    issues: list[str] = []
    prev_word = ""
    for i, scene in enumerate(scenes):
        word = _first_word(scene)
        if word and word == prev_word:
            issues.append(
                f"Consecutive scenes {i-1} and {i} both start with '{word}'"
            )
        prev_word = word
    return issues


def _check_intra_shared_phrases(
    hook: str, scenes: list[str], cta: str
) -> list[str]:
    """Flag distinctive phrases shared across different parts of the script."""
    issues: list[str] = []
    parts = {"hook": hook, "cta": cta}
    for i, s in enumerate(scenes):
        parts[f"scene_{i}"] = s

    # Build per-part n-gram sets
    part_ngrams: dict[str, set[str]] = {}
    for label, text in parts.items():
        if text.strip():
            part_ngrams[label] = _extract_all_ngrams(text)

    # Find shared phrases between different parts
    labels = list(part_ngrams.keys())
    seen: set[str] = set()
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1 :]:
            shared = part_ngrams[l1] & part_ngrams[l2]
            for phrase in shared:
                if phrase not in seen:
                    issues.append(
                        f"Shared phrase between {l1} and {l2}: '{phrase[:60]}'"
                    )
                    seen.add(phrase)
    return issues


# ── Cross-video checks ──────────────────────────────────────────────────────

def _check_cross_video_phrases(
    script_text: str,
    history_texts: list[str],
) -> tuple[list[str], list[str]]:
    """Flag distinctive phrases reused from recent production history.

    Returns (issues, flagged_phrases).
    """
    if not history_texts:
        return [], []

    current_ngrams = _extract_all_ngrams(script_text)
    if not current_ngrams:
        return [], []

    # Build history n-gram set
    history_ngrams: set[str] = set()
    for ht in history_texts:
        history_ngrams |= _extract_all_ngrams(ht)

    overlap = current_ngrams & history_ngrams
    if not overlap:
        return [], []

    issues = [f"Reused phrase from recent history: '{p[:60]}'" for p in sorted(overlap)[:10]]
    flagged = sorted(overlap)
    return issues, flagged


def _check_cross_video_openings(
    hook: str,
    history_hooks: list[str],
) -> list[str]:
    """Flag if the hook opening pattern matches recent hooks."""
    issues: list[str] = []
    if not hook or not history_hooks:
        return issues

    # Check first 3 words
    current_opening = " ".join(_normalize(hook).split()[:3])
    if not current_opening:
        return issues

    for i, h in enumerate(history_hooks):
        hist_opening = " ".join(_normalize(h).split()[:3])
        if current_opening == hist_opening:
            issues.append(
                f"Hook opening '{current_opening}' matches recent hook #{i+1}: '{h[:50]}…'"
            )
            break  # One match is enough to flag
    return issues


def _check_cross_video_cta(
    cta: str,
    history_ctas: list[str],
) -> list[str]:
    """Flag if the CTA structure matches recent CTAs."""
    issues: list[str] = []
    if not cta or not history_ctas:
        return issues

    current_norm = _normalize(cta)
    for i, h_cta in enumerate(history_ctas):
        hist_norm = _normalize(h_cta)
        # Check for identical CTA
        if current_norm == hist_norm:
            issues.append(f"Exact CTA duplicate with recent video #{i+1}")
            break
        # Check for same opening 4 words
        cur_words = current_norm.split()[:4]
        hist_words = hist_norm.split()[:4]
        if len(cur_words) >= 4 and cur_words == hist_words:
            issues.append(
                f"CTA opening pattern matches recent video #{i+1}: '{' '.join(cur_words)}…'"
            )
            break
    return issues


# ── Public API ───────────────────────────────────────────────────────────────

class RepetitionChecker:
    """Deterministic repetition checker for narration scripts.

    Parameters
    ----------
    history_depth : int
        How many recent scripts to compare against (default 20).
    """

    def __init__(self, history_depth: int = DEFAULT_HISTORY_DEPTH) -> None:
        self.history_depth = history_depth

    def check(
        self,
        script: dict,
        *,
        history: list[dict] | None = None,
    ) -> RepetitionReport:
        """Run all repetition checks on a script.

        Parameters
        ----------
        script : dict
            The generated script JSON with ``hook``, ``scenes``, ``cta``.
        history : list[dict] | None
            Recent narration entries from the narration archive.
            Each entry has ``hook``, ``scenes`` (list[str]), ``cta``.
        """
        hook = str(script.get("hook", ""))
        scenes = [str(s.get("narration", "")) for s in script.get("scenes", [])]
        cta = str(script.get("cta", ""))

        # ── Intra-video ──
        intra: list[str] = []
        intra.extend(_check_intra_duplicates(hook, scenes, cta))
        intra.extend(_check_same_word_starts(scenes))
        intra.extend(_check_intra_shared_phrases(hook, scenes, cta))

        # ── Cross-video ──
        cross: list[str] = []
        flagged: list[str] = []
        history = (history or [])[:self.history_depth]

        if history:
            # Full script text for n-gram comparison
            script_text = " ".join([hook] + scenes + [cta])
            history_texts = []
            history_hooks = []
            history_ctas = []
            for entry in history:
                h_hook = str(entry.get("hook", ""))
                h_scenes = entry.get("scenes", [])
                h_cta = str(entry.get("cta", ""))
                history_texts.append(
                    " ".join([h_hook] + [str(s) for s in h_scenes] + [h_cta])
                )
                history_hooks.append(h_hook)
                history_ctas.append(h_cta)

            phrase_issues, phrase_flags = _check_cross_video_phrases(
                script_text, history_texts
            )
            cross.extend(phrase_issues)
            flagged.extend(phrase_flags)
            cross.extend(_check_cross_video_openings(hook, history_hooks))
            cross.extend(_check_cross_video_cta(cta, history_ctas))

        passed = not intra and not cross
        if not passed:
            LOG.warning(
                "Repetition check FAILED: %d intra-video, %d cross-video issues",
                len(intra),
                len(cross),
            )
        else:
            LOG.info("Repetition check passed")

        return RepetitionReport(
            passed=passed,
            intra_issues=intra,
            cross_issues=cross,
            flagged_phrases=flagged[:20],  # Cap to avoid prompt bloat
        )
