"""Content Memory — learning from performance data.

Aggregates performance data into winning/losing patterns.
These patterns feed back into the topic_scoring stage to prioritize or penalize candidates.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from ..utils import repo_root
from .analytics import update_performance_records

LOG = logging.getLogger("utube.content_memory")


class ContentMemory:
    def __init__(self):
        self.memory_path = repo_root() / "data" / "content_memory.json"
        self.perf_path = repo_root() / "data" / "performance.json"
        self.data = self._load()

    def _load(self) -> dict:
        default_data = {
            "version": 3,
            "winning_patterns": {},
            "weak_patterns": {},
            "recent_hashes": [],
        }
        if not self.memory_path.exists():
            return default_data
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("version") != 3:
                    return default_data
                return data
        except json.JSONDecodeError:
            return default_data

    def save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def refresh_from_performance(self) -> None:
        """Analyze performance.json and extract patterns (V3)."""
        if not self.perf_path.exists():
            return
            
        try:
            with open(self.perf_path, "r", encoding="utf-8") as f:
                perf_data = json.load(f)
        except json.JSONDecodeError:
            return
            
        videos = perf_data.get("videos", [])
        if not videos:
            return
            
        self.data["version"] = 3
        stats = {
            "hook_types": defaultdict(lambda: {"samples": 0.0, "wins": 0.0}),
            "emotional_drivers": defaultdict(lambda: {"samples": 0.0, "wins": 0.0}),
            "combinations": defaultdict(lambda: {"samples": 0.0, "wins": 0.0})
        }

        for v in videos:
            is_win = v.get("performance_label") in ("winner", "above_average")
            age_days = v.get("age_days", 30.0)
            
            # Recency weighting: linearly drops from 1.0 down to 0.1 over 90 days
            weight = max(0.1, 1.0 - (age_days / 90.0))
            
            hook = v.get("hook_type")
            driver = v.get("emotional_driver")
            
            if hook and hook != "unknown":
                stats["hook_types"][hook]["samples"] += weight
                if is_win:
                    stats["hook_types"][hook]["wins"] += weight
                    
            if driver and driver != "unknown":
                stats["emotional_drivers"][driver]["samples"] += weight
                if is_win:
                    stats["emotional_drivers"][driver]["wins"] += weight
                    
            if hook and hook != "unknown" and driver and driver != "unknown":
                comb = f"{hook}|{driver}"
                stats["combinations"][comb]["samples"] += weight
                if is_win:
                    stats["combinations"][comb]["wins"] += weight

        winning_patterns = {"hook_types": {}, "emotional_drivers": {}, "combinations": {}}
        weak_patterns = {"hook_types": {}, "emotional_drivers": {}, "combinations": {}}
        
        def _wilson_score_lower_bound(wins: float, n: float, z: float = 1.96) -> float:
            if n <= 0: return 0.0
            phat = min(1.0, max(0.0, wins / n))
            return (phat + z*z/(2*n) - z * ((phat*(1-phat)+z*z/(4*n))/n)**0.5) / (1+z*z/n)

        for cat in ("hook_types", "emotional_drivers", "combinations"):
            for key, st in stats[cat].items():
                samples = st["samples"]
                wins = st["wins"]
                win_rate = wins / samples if samples > 0 else 0
                confidence = _wilson_score_lower_bound(wins, samples)
                
                pattern = {
                    "samples": round(samples, 2),
                    "wins": round(wins, 2),
                    "win_rate": round(win_rate, 2),
                    "confidence": round(confidence, 2)
                }
                
                # ~2 recent videos equivalent threshold
                if samples >= 1.5:
                    if win_rate >= 0.5:
                        winning_patterns[cat][key] = pattern
                    elif win_rate <= 0.3:
                        weak_patterns[cat][key] = pattern

        self.data["winning_patterns"] = winning_patterns
        self.data["weak_patterns"] = weak_patterns
        
        # Track recent hashes to prevent immediate repetition (keep last 30)
        recent = sorted(videos, key=lambda x: x.get("published_at", ""), reverse=True)[:30]
        self.data["recent_hashes"] = [v["topic_hash"] for v in recent if v.get("topic_hash")]
        
        self.save()
        LOG.info("Content memory version 3 refreshed")

    def get_context_for_scoring(self) -> dict:
        """Returns a summarized context dict for the LLM and heuristic algorithms."""
        win_hooks = self.data.get("winning_patterns", {}).get("hook_types", {})
        lose_hooks = self.data.get("weak_patterns", {}).get("hook_types", {})
        win_combs = self.data.get("winning_patterns", {}).get("combinations", {})
        
        strong_hooks = []
        for h, st in win_hooks.items():
            if st.get("confidence", 0) >= 0.2:
                strong_hooks.append({"hook": h, "confidence": st["confidence"]})
        strong_hooks.sort(key=lambda x: x["confidence"], reverse=True)
                
        weak_hooks = []
        for h, st in lose_hooks.items():
            if st.get("win_rate", 1) <= 0.3:
                weak_hooks.append(h)
                
        strong_combinations = []
        for c, st in win_combs.items():
            if st.get("confidence", 0) >= 0.2:
                strong_combinations.append({"combination": c, "confidence": st["confidence"]})
        strong_combinations.sort(key=lambda x: x["confidence"], reverse=True)
                
        return {
            "strong_hooks": [h["hook"] for h in strong_hooks],
            "weak_hooks": weak_hooks,
            "strong_combinations": strong_combinations,
            "recent_hashes": self.data.get("recent_hashes", []),
        }

def refresh_memory(ledger_entries: list[dict]) -> None:
    """Helper to update performance and then refresh memory."""
    update_performance_records(ledger_entries)
    mem = ContentMemory()
    mem.refresh_from_performance()
