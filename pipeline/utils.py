"""Shared utilities."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("utube")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_date() -> str:
    """Return RUN_DATE env var or today UTC as YYYY-MM-DD."""
    override = os.getenv("RUN_DATE", "").strip()
    if override:
        return override
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_dir(slot_id: str | None = None) -> Path:
    base = repo_root() / "runs" / run_date()
    if slot_id:
        base = base / slot_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len] or "topic"


def topic_hash(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()[:12]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default
