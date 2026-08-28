"""Content memory: learn which content patterns perform and expose safe priors."""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from ..utils import repo_root
from .analytics import update_performance_records

LOG = logging.getLogger("utube.content_memory")
VERSION = 3


def _default() -> dict:
    return {"version": VERSION, "winning_patterns": {}, "weak_patterns": {}, "recent_hashes": [], "updated_at": None}


def _bucket_duration(seconds: Any) -> str:
    try: s = float(seconds)
    except (TypeError, ValueError): return "unknown"
    if s < 30: return "25-29"
    if s < 35: return "30-34"
    if s < 40: return "35-39"
    return "40-42"


def _title_pattern(title: str) -> str:
    t = (title or "").lower()
    if "?" in t or t.startswith(("why ", "how ")): return "question"
    if any(x in t for x in (" vs ", "versus", "compared")): return "comparison"
    if any(x in t for x in ("secret", "hidden", "nobody knows", "you don't know")): return "secret"
    if any(x in t for x in ("danger", "dangerous", "scam", "warning")): return "danger"
    if any(ch.isdigit() for ch in t): return "number"
    if any(x in t for x in ("real reason", "truth", "actually")): return "contradiction"
    return "statement"


def _topic_family(video: dict) -> str:
    explicit = str(video.get("topic_family") or "").strip()
    if explicit: return explicit.lower()
    text = f"{video.get('title', '')} {video.get('chosen_angle', '')}".lower()
    rules = {
        "ai scams": ("voice cloning", "deepfake", "impersonat", "ai scam", "ai fraud"),
        "ai": ("artificial intelligence", " ai ", "chatgpt", "agent", "machine learning"),
        "cybersecurity": ("hack", "malware", "phishing", "password", "cyber", "security"),
        "privacy": ("privacy", "tracking", "location", "spy", "permission", "data"),
        "consumer technology": ("phone", "iphone", "android", "laptop", "browser", "wifi"),
        "internet mechanics": ("internet", "dns", "server", "cookie", "cloud"),
    }
    for family, needles in rules.items():
        if any(n in text for n in needles): return family
    return "other"


def _visual_source(video: dict) -> str:
    sources = video.get("visual_sources")
    if isinstance(sources, dict) and sources:
        nonzero = [(k, float(v or 0)) for k, v in sources.items() if float(v or 0) > 0]
        if nonzero: return max(nonzero, key=lambda x: x[1])[0]
    return str(video.get("visual_source") or "unknown").lower()


def _weight(age_days: Any) -> float:
    try: age = max(0.0, float(age_days))
    except (TypeError, ValueError): age = 30.0
    return max(0.15, math.exp(-age / 90.0))


def _posterior(wins: float, samples: float) -> tuple[float, float]:
    """Beta(2,2) shrinkage plus bounded evidence strength."""
    mean = (wins + 2.0) / (samples + 4.0)
    strength = min(1.0, samples / 10.0)
    return mean, strength


def _accumulate(stats: dict, key: str, win: bool, weight: float) -> None:
    if not key or key == "unknown": return
    item = stats.setdefault(key, {"samples": 0.0, "wins": 0.0})
    item["samples"] += weight
    if win: item["wins"] += weight


class ContentMemory:
    def __init__(self):
        root = repo_root()
        self.memory_path = root / "data" / "content_memory.json"
        self.perf_path = root / "data" / "performance.json"
        self.data = self._load()

    def _load(self) -> dict:
        if not self.memory_path.exists(): return _default()
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if data.get("version") != VERSION: return _default()
            return data
        except (json.JSONDecodeError, OSError): return _default()

    def save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def refresh_from_performance(self) -> None:
        if not self.perf_path.exists(): return
        try: perf = json.loads(self.perf_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.error("Cannot read performance data: %s", exc); return
        videos = perf.get("videos", [])
        if not videos: return

        categories = ("topic_families", "hook_types", "emotional_drivers", "duration_buckets", "title_patterns", "visual_sources", "topic_hook", "topic_emotion", "hook_emotion", "topic_hook_emotion")
        stats = {cat: {} for cat in categories}
        for v in videos:
            label = v.get("performance_label")
            if label in ("too_new", "unknown", None): continue
            win = label in ("winner", "above_average")
            weight = _weight(v.get("age_days", 30))
            family = _topic_family(v); hook = str(v.get("hook_type") or "unknown").lower(); emotion = str(v.get("emotional_driver") or "unknown").lower()
            duration = str(v.get("duration_bucket") or _bucket_duration(v.get("duration_seconds"))).lower()
            title_pat = str(v.get("title_pattern") or _title_pattern(v.get("title", ""))).lower(); visual = _visual_source(v)
            _accumulate(stats["topic_families"], family, win, weight); _accumulate(stats["hook_types"], hook, win, weight)
            _accumulate(stats["emotional_drivers"], emotion, win, weight); _accumulate(stats["duration_buckets"], duration, win, weight)
            _accumulate(stats["title_patterns"], title_pat, win, weight); _accumulate(stats["visual_sources"], visual, win, weight)
            if hook != "unknown": _accumulate(stats["topic_hook"], f"{family}|{hook}", win, weight)
            if emotion != "unknown": _accumulate(stats["topic_emotion"], f"{family}|{emotion}", win, weight)
            if hook != "unknown" and emotion != "unknown":
                _accumulate(stats["hook_emotion"], f"{hook}|{emotion}", win, weight)
                _accumulate(stats["topic_hook_emotion"], f"{family}|{hook}|{emotion}", win, weight)

        min_samples = 5.0
        try:
            import yaml
            cfg = yaml.safe_load((repo_root() / "config" / "quality.yaml").read_text(encoding="utf-8")) or {}
            min_samples = float(cfg.get("content_memory", {}).get("min_samples_for_pattern", 5))
        except Exception: pass

        winning, weak = {}, {}
        for cat, entries in stats.items():
            winning[cat], weak[cat] = {}, {}
            for key, st in entries.items():
                samples, wins = st["samples"], st["wins"]
                mean, strength = _posterior(wins, samples)
                pattern = {"samples": round(samples, 2), "wins": round(wins, 2), "win_rate": round(wins / samples, 3) if samples else 0.0, "posterior_mean": round(mean, 3), "evidence_strength": round(strength, 3)}
                if samples >= min_samples and mean >= 0.55: winning[cat][key] = pattern
                elif samples >= min_samples and mean <= 0.40: weak[cat][key] = pattern
        self.data.update({"version": VERSION, "winning_patterns": winning, "weak_patterns": weak, "recent_hashes": [v.get("topic_hash") for v in sorted(videos, key=lambda x: x.get("published_at", ""), reverse=True)[:30] if v.get("topic_hash")], "updated_at": datetime.now(timezone.utc).isoformat()})
        self.save()
        LOG.info("Content memory V3 refreshed from %d videos", len(videos))

    def get_context_for_scoring(self) -> dict:
        win, weak = self.data.get("winning_patterns", {}), self.data.get("weak_patterns", {})
        def ranked(cat: str, source: dict) -> list[dict]:
            vals = [{"key": k, **v} for k, v in source.get(cat, {}).items()]
            return sorted(vals, key=lambda x: (x.get("posterior_mean", 0), x.get("evidence_strength", 0)), reverse=True)[:8]
        return {
            "winning_patterns": win,
            "weak_patterns": weak,
            "strong_topic_families": ranked("topic_families", win),
            "weak_topic_families": ranked("topic_families", weak),
            "strong_hooks": ranked("hook_types", win),
            "weak_hooks": ranked("hook_types", weak),
            "strong_emotions": ranked("emotional_drivers", win),
            "strong_durations": ranked("duration_buckets", win),
            "strong_title_patterns": ranked("title_patterns", win),
            "strong_visual_sources": ranked("visual_sources", win),
            "strong_combinations": ranked("topic_hook_emotion", win),
            "recent_hashes": self.data.get("recent_hashes", []),
        }


def refresh_memory(ledger_entries: list[dict]) -> None:
    update_performance_records(ledger_entries)
    ContentMemory().refresh_from_performance()
