"""Single source of truth for ALL configuration.

Loads and deep-merges:
  config/goal.yaml          channel mission, persona, format
  config/providers.yaml     external service URLs, models, defaults
  config/pipeline.yaml      per-stage tunables
  config/niches.yaml        per-slot definitions
  config/schedule.yaml      cron + dedup window

Any code that needs a value must call get_config() and read from the resulting
dict — there are no hardcoded constants outside this module's defaults.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .utils import repo_root

LOG = logging.getLogger("utube.config")


CONFIG_FILES = [
    "goal.yaml",
    "providers.yaml",
    "pipeline.yaml",
    "lanes.yaml",
    "schedule.yaml",
]


class Config(dict):
    """Plain dict with attribute access and a `get_path` helper for nested keys."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        if isinstance(v, dict) and not isinstance(v, Config):
            return Config(v)
        return v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """`cfg.get_path('llm.providers.nvidia_nim.model')`"""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


@lru_cache(maxsize=1)
def get_config() -> Config:
    merged: dict[str, Any] = {}
    cfg_dir = repo_root() / "config"
    for name in CONFIG_FILES:
        path = cfg_dir / name
        if not path.exists():
            LOG.warning("Config file missing: %s", path)
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid YAML in {path}: {e}") from e
        if not isinstance(data, dict):
            raise RuntimeError(f"{path} did not parse as a mapping")
        _deep_merge(merged, data)
    LOG.debug("Loaded config keys: %s", list(merged.keys()))
    return Config(merged)


def reset_config_cache() -> None:
    """Useful for tests."""
    get_config.cache_clear()


# ---- helpers commonly used by the pipeline ----

def goal_summary() -> str:
    """Compact text summary of the channel goal — injected into LLM prompts."""
    cfg = get_config()
    ch = cfg.get_path("channel", {}) or {}
    parts = []
    if ch.get("name"):
        parts.append(f"Channel: {ch['name']}")
    if ch.get("audience"):
        parts.append(f"Audience: {ch['audience']}")
    if ch.get("goal"):
        parts.append(f"Goal: {ch['goal'].strip()}")
    if ch.get("tone"):
        parts.append(f"Tone: {', '.join(ch['tone'])}")
    if ch.get("language"):
        parts.append(f"Language: {ch['language']}")
    return "\n".join(parts)
