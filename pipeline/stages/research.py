"""Topic selection + research brief — token budgets and reasoning effort from pipeline.yaml."""
from __future__ import annotations

import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root, topic_hash

LOG = logging.getLogger("utube.research")


def _research_cfg() -> dict:
    return get_config().get_path("research", {}) or {}


def select_topic(
    llm: LLMRouter,
    candidates: list[dict],
    *,
    niche_title: str,
    sources_label: str,
    recent_hashes: list[str],
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No candidates to choose from")

    cfg = get_config()
    rcfg = _research_cfg().get("topic_select", {}) or {}
    template = (repo_root() / "prompts" / "topic_select.txt").read_text(encoding="utf-8")

    rendered = "\n".join(
        f"[{i}] ({c.get('source','')}, score={c.get('score',0)}) {c.get('title','')}\n    URL: {c.get('url','')}"
        for i, c in enumerate(candidates)
    )
    prompt = template.format(
        goal=goal_summary(),
        niche_title=niche_title,
        n_candidates=len(candidates),
        sources=sources_label,
        recent_hashes=", ".join(recent_hashes[-30:]) or "(none)",
        candidates=rendered,
        target_duration=cfg.get_path("video.target_duration_sec", 35),
        format_label=cfg.get_path("channel.format", "shorts"),
        hook_max_seconds=cfg.get_path("script.hook_max_seconds", 3),
    )

    out = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=int(rcfg.get("max_tokens", 2000)),
        temperature=float(rcfg.get("temperature", 0.5)),
        reasoning_effort=rcfg.get("reasoning_effort"),
    )

    idx = int(out.get("chosen_index", 0))
    idx = max(0, min(idx, len(candidates) - 1))
    chosen = dict(candidates[idx])
    chosen["angle"] = out.get("angle", "")
    chosen["reason"] = out.get("reason", "")
    chosen["open_loop"] = out.get("open_loop", "")
    chosen["payoff"] = out.get("payoff", "")
    chosen["topic_hash"] = out.get("topic_hash") or topic_hash(chosen["title"])
    LOG.info("Selected topic: %s", chosen["title"])
    return chosen


def fetch_source_text(url: str, *, max_chars: int | None = None) -> str:
    if max_chars is None:
        max_chars = int(_research_cfg().get("source_text_max_chars", 6000))
    if not url:
        return ""
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (utube-bot)"},
            allow_redirects=True,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        node = soup.find("article") or soup.find("main") or soup.body or soup
        text = " ".join(node.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as e:  # noqa: BLE001
        LOG.warning("fetch_source_text(%s) failed: %s", url, e)
        return ""


def build_research_brief(llm: LLMRouter, topic: dict) -> dict[str, Any]:
    cfg = get_config()
    bcfg = _research_cfg().get("brief", {}) or {}
    template = (repo_root() / "prompts" / "research.txt").read_text(encoding="utf-8")

    source_url = topic.get("external_url") or topic.get("url", "")
    source_text = topic.get("summary") or ""
    if len(source_text) < 300 and source_url and not source_url.startswith("https://reddit.com"):
        fetched = fetch_source_text(source_url)
        if fetched:
            source_text = fetched

    prompt = template.format(
        goal=goal_summary(),
        topic_title=topic.get("title", ""),
        angle=topic.get("angle", ""),
        source_url=source_url,
        source_text=source_text or "(no source text available)",
        target_duration=cfg.get_path("video.target_duration_sec", 35),
        format_label=cfg.get_path("channel.format", "shorts"),
    )

    brief = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=int(bcfg.get("max_tokens", 2500)),
        temperature=float(bcfg.get("temperature", 0.4)),
        reasoning_effort=bcfg.get("reasoning_effort"),
    )
    LOG.info("Research brief: %d facts, %d gaps",
             len(brief.get("key_facts", [])), len(brief.get("open_questions", [])))
    return brief
