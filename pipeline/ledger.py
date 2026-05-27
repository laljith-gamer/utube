"""History + de-duplication ledger persisted to ``assets/history.json``.

This single file is the source of truth for:

  - Topic-hash dedup per niche (don't re-pick the same topic in N days).
  - Source-URL dedup per niche (don't re-cover the same article URL even
    if the topic hash drifts).
  - Per-provider quota counters (advisory).
  - Append-only **video records** — full metadata for every successful run.
  - Append-only **error records** — every failure with stage + message.

Schema::

    {
      "version": 1,
      "topics":  {<slot>: {<topic_hash>: <iso_ts>}},
      "sources": {<slot>: {<url_hash>:   <iso_ts>}},
      "usage":   {<date>: {<provider>:   <count>}},
      "videos":  [<video_record>, ...],     # capped to DEFAULT_VIDEOS_TRIM
      "errors":  [<error_record>, ...]      # capped to DEFAULT_ERRORS_TRIM
    }

The file is committed back to the repo by the daily workflow so the
history persists across runs without a database.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("utube.history")

CURRENT_VERSION = 1
DEFAULT_TOPICS_TRIM = 200
DEFAULT_VIDEOS_TRIM = 1000
DEFAULT_ERRORS_TRIM = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_url(url: str) -> str:
    return hashlib.sha1((url or "").lower().strip().encode("utf-8")).hexdigest()[:16]


@dataclass
class Ledger:
    """Replaces the old ``ledger.json``. Class name preserved so existing
    call-sites (``Ledger.load``, ``record_topic``, ``recent_hashes``) keep
    working; new methods are additive."""

    path: Path
    data: dict = field(default_factory=dict)

    # ----- load / save -----

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        # 1. Try the requested path
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls._from_dict(path, data)
            except Exception as e:  # noqa: BLE001
                LOG.warning("History %s corrupt (%s); starting fresh", path, e)

        # 2. One-time migration from the old repo-root ledger.json.
        # If the new history.json doesn't exist but the old ledger.json does,
        # adopt its contents as the seed.
        if path.name == "history.json":
            legacy = path.parent.parent / "ledger.json"
            if legacy.exists():
                try:
                    data = json.loads(legacy.read_text(encoding="utf-8"))
                    LOG.info("Migrating legacy ledger %s -> %s", legacy, path)
                    return cls._from_dict(path, data)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("Legacy ledger migration failed (%s); starting fresh", e)

        return cls._from_dict(path, {})

    @classmethod
    def _from_dict(cls, path: Path, data: dict) -> "Ledger":
        data = dict(data) if isinstance(data, dict) else {}
        data.setdefault("version", CURRENT_VERSION)
        data.setdefault("topics", {})
        data.setdefault("sources", {})
        data.setdefault("usage", {})
        data.setdefault("videos", [])
        data.setdefault("errors", [])
        return cls(path=path, data=data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write to avoid corrupting the file if the process is killed
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # ----- topics (existing API, still works) -----

    def recent_hashes(self, slot_id: str, *, days: int) -> list[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out: list[str] = []
        for h, ts in self.data.get("topics", {}).get(slot_id, {}).items():
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    out.append(h)
            except ValueError:
                continue
        return out

    def record_topic(self, slot_id: str, topic_hash: str) -> None:
        topics = self.data.setdefault("topics", {}).setdefault(slot_id, {})
        topics[topic_hash] = _now_iso()
        if len(topics) > DEFAULT_TOPICS_TRIM:
            for k in list(topics.keys())[:-DEFAULT_TOPICS_TRIM]:
                topics.pop(k, None)

    # ----- source URL dedup -----

    def recent_source_url_hashes(self, slot_id: str, *, days: int) -> set[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out: set[str] = set()
        for h, ts in self.data.get("sources", {}).get(slot_id, {}).items():
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    out.add(h)
            except ValueError:
                continue
        return out

    def is_duplicate_url(self, slot_id: str, url: str, *, days: int) -> bool:
        if not url:
            return False
        return _hash_url(url) in self.recent_source_url_hashes(slot_id, days=days)

    def record_source_url(self, slot_id: str, url: str) -> None:
        if not url:
            return
        sources = self.data.setdefault("sources", {}).setdefault(slot_id, {})
        sources[_hash_url(url)] = _now_iso()
        if len(sources) > DEFAULT_TOPICS_TRIM:
            for k in list(sources.keys())[:-DEFAULT_TOPICS_TRIM]:
                sources.pop(k, None)

    # ----- usage (advisory provider counters) -----

    def bump(self, provider: str, count: int = 1) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        u = self.data.setdefault("usage", {}).setdefault(today, {})
        u[provider] = u.get(provider, 0) + count

    # ----- video records -----

    def record_video(self, record: dict) -> None:
        """Append a full per-video record. Caller supplies whatever fields
        it has; we attach a timestamp and trim the tail."""
        record = dict(record)
        record.setdefault("recorded_at", _now_iso())
        videos = self.data.setdefault("videos", [])
        videos.append(record)
        if len(videos) > DEFAULT_VIDEOS_TRIM:
            del videos[:-DEFAULT_VIDEOS_TRIM]

    # ----- error records -----

    def record_error(self, *, slot_id: str, stage: str, error: str,
                     traceback: str | None = None) -> None:
        errs = self.data.setdefault("errors", [])
        errs.append({
            "slot": slot_id,
            "stage": stage,
            "error": str(error)[:500],
            "traceback": (traceback or "")[:2000],
            "at": _now_iso(),
        })
        if len(errs) > DEFAULT_ERRORS_TRIM:
            del errs[:-DEFAULT_ERRORS_TRIM]

    # ----- read-only helpers -----

    def stats(self) -> dict[str, Any]:
        return {
            "videos_total":     len(self.data.get("videos", [])),
            "errors_total":     len(self.data.get("errors", [])),
            "topics_per_slot":  {k: len(v) for k, v in self.data.get("topics", {}).items()},
            "sources_per_slot": {k: len(v) for k, v in self.data.get("sources", {}).items()},
        }
