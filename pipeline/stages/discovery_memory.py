"""Persistent memory for discovery events to track velocity and novelty."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from ..utils import repo_root

LOG = logging.getLogger("utube.discovery_memory")

class DiscoveryMemory:
    def __init__(self):
        self.path = repo_root() / "data" / "discovery_memory.json"
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"events": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Discovery memory corrupt (%s), starting fresh", exc)
            return {"events": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Prune events older than 7 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        to_remove = []
        for event_id, event_data in self.data.get("events", {}).items():
            try:
                last_seen = datetime.fromisoformat(event_data.get("last_seen_at", ""))
                if last_seen < cutoff:
                    to_remove.append(event_id)
            except ValueError:
                to_remove.append(event_id)
        
        for k in to_remove:
            self.data["events"].pop(k, None)

        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def record_event(self, event_id: str, current_mentions: int, traffic_volume: int = 0) -> dict:
        """Records an event and returns its historical metrics (velocity, etc)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        events = self.data.setdefault("events", {})
        
        if event_id not in events:
            events[event_id] = {
                "first_seen_at": now_iso,
                "last_seen_at": now_iso,
                "history": [{"timestamp": now_iso, "mentions": current_mentions, "traffic_volume": traffic_volume}],
                "max_mentions": current_mentions,
                "max_traffic": traffic_volume
            }
            return {"novelty": 1.0, "velocity": current_mentions, "acceleration": 0.0, "age_hours": 0.0, "prior_mentions": 0, "trend_momentum": traffic_volume}
        
        ev = events[event_id]
        ev["last_seen_at"] = now_iso
        
        # Calculate age
        try:
            first_seen = datetime.fromisoformat(ev["first_seen_at"])
            age_hours = (datetime.now(timezone.utc) - first_seen).total_seconds() / 3600.0
        except ValueError:
            age_hours = 0.0

        # Calculate velocity and acceleration
        history = ev.setdefault("history", [])
        history.append({"timestamp": now_iso, "mentions": current_mentions, "traffic_volume": traffic_volume})
        
        # Prune history to last 10 records for this event
        if len(history) > 10:
            ev["history"] = history[-10:]
            
        prior_mentions = history[-2]["mentions"] if len(history) > 1 else current_mentions
        velocity = current_mentions - prior_mentions
        
        prior_traffic = history[-2].get("traffic_volume", 0) if len(history) > 1 else traffic_volume
        trend_momentum = traffic_volume - prior_traffic
        
        acceleration = 0.0
        if len(history) >= 3:
            prev_velocity = history[-2]["mentions"] - history[-3]["mentions"]
            acceleration = velocity - prev_velocity

        ev["max_mentions"] = max(ev.get("max_mentions", 0), current_mentions)
        ev["max_traffic"] = max(ev.get("max_traffic", 0), traffic_volume)
        
        return {
            "novelty": max(0.0, 1.0 - (age_hours / 168.0)), # 168 hours = 7 days
            "velocity": velocity,
            "acceleration": acceleration,
            "age_hours": age_hours,
            "prior_mentions": prior_mentions,
            "trend_momentum": trend_momentum
        }
