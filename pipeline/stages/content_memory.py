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
        if not self.memory_path.exists():
            return {
                "winning_patterns": {},
                "weak_patterns": {},
                "recent_hashes": [],
            }
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("version") != 2:
                    data = {
                        "version": 2,
                        "winning_patterns": {},
                        "weak_patterns": {},
                        "recent_hashes": [],
                    }
                return data
        except json.JSONDecodeError:
            return {
                "version": 2,
                "winning_patterns": {},
                "weak_patterns": {},
                "recent_hashes": [],
            }

    def save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def refresh_from_performance(self) -> None:
        """Analyze performance.json and extract patterns."""
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
            
        self.data["version"] = 2
        stats = {
            "hook_types": defaultdict(lambda: {"samples": 0, "wins": 0}),
            "emotional_drivers": defaultdict(lambda: {"samples": 0, "wins": 0})
        }

        for v in videos:
            is_win = v.get("performance_label") in ("winner", "above_average")
            hook = v.get("hook_type")
            if hook and hook != "unknown":
                stats["hook_types"][hook]["samples"] += 1
                if is_win:
                    stats["hook_types"][hook]["wins"] += 1
                    
            driver = v.get("emotional_driver")
            if driver and driver != "unknown":
                stats["emotional_drivers"][driver]["samples"] += 1
                if is_win:
                    stats["emotional_drivers"][driver]["wins"] += 1

        winning_patterns = {"hook_types": {}, "emotional_drivers": {}}
        weak_patterns = {"hook_types": {}, "emotional_drivers": {}}

        for cat in ("hook_types", "emotional_drivers"):
            for key, st in stats[cat].items():
                samples = st["samples"]
                wins = st["wins"]
                win_rate = wins / samples if samples > 0 else 0
                
                pattern = {
                    "samples": samples,
                    "wins": wins,
                    "win_rate": round(win_rate, 2)
                }
                
                if samples >= 2:
                    if win_rate >= 0.5:
                        winning_patterns[cat][key] = pattern
                    elif win_rate <= 0.3:
                        weak_patterns[cat][key] = pattern

        self.data["winning_patterns"] = winning_patterns
        self.data["weak_patterns"] = weak_patterns
        
        # Track recent hashes to prevent immediate repetition
        # Only keep last 30
        recent = sorted(videos, key=lambda x: x.get("published_at", ""), reverse=True)[:30]
        self.data["recent_hashes"] = [v["topic_hash"] for v in recent if v.get("topic_hash")]
        
        self.save()
        LOG.info("Content memory version 2 refreshed")

    def get_context_for_scoring(self) -> dict:
        """Returns a summarized context dict for the LLM to use during scoring."""
        # Find hook types that are strongly winning
        win_hooks = self.data.get("winning_patterns", {}).get("hook_types", {})
        lose_hooks = self.data.get("weak_patterns", {}).get("hook_types", {})
        
        strong_hooks = []
        for h, st in win_hooks.items():
            if st.get("win_rate", 0) >= 0.5:
                strong_hooks.append(h)
                
        weak_hooks = []
        for h, st in lose_hooks.items():
            if st.get("win_rate", 1) <= 0.3:
                weak_hooks.append(h)
                
        return {
            "strong_hooks": strong_hooks,
            "weak_hooks": weak_hooks,
            "recent_hashes": self.data.get("recent_hashes", []),
        }

def refresh_memory(ledger_entries: list[dict]) -> None:
    """Helper to update performance and then refresh memory."""
    update_performance_records(ledger_entries)
    mem = ContentMemory()
    mem.refresh_from_performance()
