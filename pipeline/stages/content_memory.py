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
                return json.load(f)
        except json.JSONDecodeError:
            return {
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
            
        # Extract patterns
        winners = [v for v in videos if v.get("performance_label") in ("winner", "above_average")]
        losers = [v for v in videos if v.get("performance_label") in ("failure", "below_average")]
        
        self.data["winning_patterns"] = self._extract_patterns(winners)
        self.data["weak_patterns"] = self._extract_patterns(losers)
        
        # Track recent hashes to prevent immediate repetition
        # Only keep last 30
        recent = sorted(videos, key=lambda x: x.get("published_at", ""), reverse=True)[:30]
        self.data["recent_hashes"] = [v["topic_hash"] for v in recent if v.get("topic_hash")]
        
        self.save()
        LOG.info("Content memory refreshed: %d winners, %d losers analyzed", len(winners), len(losers))

    def _extract_patterns(self, video_list: list[dict]) -> dict:
        patterns = {
            "hook_types": defaultdict(int),
            "emotional_drivers": defaultdict(int),
        }
        
        for v in video_list:
            hook = v.get("hook_type")
            if hook and hook != "unknown":
                patterns["hook_types"][hook] += 1
                
            driver = v.get("emotional_driver")
            if driver and driver != "unknown":
                patterns["emotional_drivers"][driver] += 1
                
        return {
            "hook_types": dict(patterns["hook_types"]),
            "emotional_drivers": dict(patterns["emotional_drivers"])
        }

    def get_context_for_scoring(self) -> dict:
        """Returns a summarized context dict for the LLM to use during scoring."""
        # Find hook types that are strongly winning
        win_hooks = self.data.get("winning_patterns", {}).get("hook_types", {})
        lose_hooks = self.data.get("weak_patterns", {}).get("hook_types", {})
        
        strong_hooks = []
        for h, count in win_hooks.items():
            if count >= 3 and count > lose_hooks.get(h, 0) * 2:
                strong_hooks.append(h)
                
        weak_hooks = []
        for h, count in lose_hooks.items():
            if count >= 3 and count > win_hooks.get(h, 0) * 2:
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
