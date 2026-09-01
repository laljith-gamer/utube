"""Lightweight quota / topic-history ledger persisted to disk.

Tracks:
- recent topic hashes per niche (to avoid 30-day repeats)
- recent topic families (to avoid similar semantic topics)
- per-day usage counters per provider (best-effort, advisory)
- full performance records
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG = logging.getLogger("utube.ledger")


@dataclass
class Ledger:
    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        if path.exists():
            try:
                return cls(path=path, data=json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:  # noqa: BLE001
                LOG.warning("Ledger %s corrupt (%s); starting fresh", path, e)
        return cls(path=path, data={"topics": {}, "families": {}, "themes": {}, "usage": {}, "runs": []})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    # ----- topics -----

    def recent_hashes(self, slot_id: str, *, days: int) -> list[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out = []
        for h, ts in self.data.get("topics", {}).get(slot_id, {}).items():
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    out.append(h)
            except ValueError:
                continue
        return out

    def record_topic(self, slot_id: str, topic_hash: str) -> None:
        topics = self.data.setdefault("topics", {}).setdefault(slot_id, {})
        topics[topic_hash] = datetime.now(timezone.utc).isoformat()
        if len(topics) > 200:
            for k in list(topics.keys())[:-200]:
                topics.pop(k, None)

    # ----- families -----

    def recent_families(self, *, days: int) -> list[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out = []
        for f, ts in self.data.get("families", {}).items():
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    out.append(f)
            except ValueError:
                continue
        return out

    def record_family(self, family: str) -> None:
        if not family:
            return
        families = self.data.setdefault("families", {})
        families[family] = datetime.now(timezone.utc).isoformat()
        if len(families) > 500:
            for k in list(families.keys())[:-500]:
                families.pop(k, None)

    # ----- themes (global, not per-lane) -----

    def recent_theme_ids(self, *, days: int) -> set[str]:
        """Return theme ids used in the last `days`. Used to skip-pick repeats."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out: set[str] = set()
        for tid, ts in self.data.get("themes", {}).items():
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    out.add(tid)
            except ValueError:
                continue
        return out

    def record_theme(self, theme_id: str) -> None:
        themes = self.data.setdefault("themes", {})
        themes[theme_id] = datetime.now(timezone.utc).isoformat()
        if len(themes) > 2000:
            for k in list(themes.keys())[:-2000]:
                themes.pop(k, None)

    # ----- source health -----

    def record_source_health(self, source: str, success: bool, failure_type: str = "") -> None:
        health = self.data.setdefault("source_health", {}).setdefault(source, {"successes": 0, "failures": 0, "history": []})
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "failure_type": failure_type
        }
        health["history"].append(event)
        if success:
            health["successes"] += 1
        else:
            health["failures"] += 1
        
        # Keep rolling window of 50
        if len(health["history"]) > 50:
            oldest = health["history"].pop(0)
            if oldest["success"]:
                health["successes"] -= 1
            else:
                health["failures"] -= 1

    def get_source_health(self, source: str) -> float:
        """Returns a score between 0.0 and 1.0. 1.0 means perfect health."""
        health = self.data.get("source_health", {}).get(source)
        if not health or not health["history"]:
            return 1.0
        total = health["successes"] + health["failures"]
        if total == 0:
            return 1.0
        return health["successes"] / total

    # ----- usage -----

    def bump(self, provider: str, count: int = 1) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        u = self.data.setdefault("usage", {}).setdefault(today, {})
        u[provider] = u.get(provider, 0) + count

    # ----- full run records -----
    
    def record_run(self, metadata: dict) -> None:
        runs = self.data.setdefault("runs", [])
        runs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata
        })
        if len(runs) > 1000:
            self.data["runs"] = runs[-1000:]
