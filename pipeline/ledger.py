"""Lightweight quota / topic-history ledger persisted to disk.

Tracks:
- recent topic hashes per niche (to avoid 30-day repeats)
- per-day usage counters per provider (best-effort, advisory)
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
        return cls(path=path, data={"topics": {}, "themes": {}, "usage": {}})

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
        # Trim to last 200 per slot
        if len(topics) > 200:
            for k in list(topics.keys())[:-200]:
                topics.pop(k, None)

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
        # Trim to last 2000 (covers ~3 years at 2 vids/day)
        if len(themes) > 2000:
            for k in list(themes.keys())[:-2000]:
                themes.pop(k, None)

    # ----- usage -----

    def bump(self, provider: str, count: int = 1) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        u = self.data.setdefault("usage", {}).setdefault(today, {})
        u[provider] = u.get(provider, 0) + count
