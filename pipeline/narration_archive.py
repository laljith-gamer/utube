"""Lightweight narration archive for cross-video repetition checking.

Stores recent narration texts in ``data/narration_history.json`` so the
repetition checker can compare new scripts against recent production output
without scanning the filesystem.

The archive is append-only with a configurable cap (default 50 entries).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .utils import repo_root

LOG = logging.getLogger("utube.narration_archive")

DEFAULT_MAX_ENTRIES = 50


def _archive_path() -> Path:
    return repo_root() / "data" / "narration_history.json"


def load_recent(n: int = DEFAULT_MAX_ENTRIES) -> list[dict]:
    """Load the most recent N narration entries.

    Each entry has: ``hook``, ``scenes`` (list[str]), ``cta``, ``title``,
    ``run_id``, ``timestamp``.
    """
    path = _archive_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("narrations", [])
        return entries[-n:]
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("Could not load narration archive: %s", exc)
        return []


def append(script: dict, *, run_id: str = "", timestamp: str = "") -> None:
    """Append a script's narration to the archive.

    Parameters
    ----------
    script : dict
        The script JSON with ``hook``, ``scenes``, ``cta``, ``title``.
    run_id : str
        Optional run identifier.
    timestamp : str
        Optional ISO timestamp.
    """
    path = _archive_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        data = {}

    entries: list[dict[str, Any]] = data.get("narrations", [])
    entry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "hook": str(script.get("hook", "")),
        "scenes": [str(s.get("narration", "")) for s in script.get("scenes", [])],
        "cta": str(script.get("cta", "")),
        "title": str(script.get("title", "")),
    }
    entries.append(entry)

    # Cap at max entries
    max_entries = DEFAULT_MAX_ENTRIES
    if len(entries) > max_entries:
        entries = entries[-max_entries:]

    data["narrations"] = entries
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Narration archived (%d total entries)", len(entries))
