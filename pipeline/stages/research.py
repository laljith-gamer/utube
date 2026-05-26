"""Pick best topic + enrich with research brief via LLM."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from ..providers.llm import LLMRouter
from ..utils import repo_root, topic_hash

LOG = logging.getLogger("utube.research")


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

    template = (repo_root() / "prompts" / "topic_select.txt").read_text(encoding="utf-8")

    rendered = "\n".join(
        f"[{i}] ({c.get('source','')}, score={c.get('score',0)}) {c.get('title','')}\n    URL: {c.get('url','')}"
        for i, c in enumerate(candidates)
    )
    prompt = template.format(
        niche_title=niche_title,
        n_candidates=len(candidates),
        sources=sources_label,
        recent_hashes=", ".join(recent_hashes[-30:]) or "(none)",
        candidates=rendered,
    )

    out = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.5,
        reasoning_effort="medium",
    )

    idx = int(out.get("chosen_index", 0))
    idx = max(0, min(idx, len(candidates) - 1))
    chosen = dict(candidates[idx])
    chosen["angle"] = out.get("angle", "")
    chosen["reason"] = out.get("reason", "")
    chosen["topic_hash"] = out.get("topic_hash") or topic_hash(chosen["title"])
    LOG.info("Selected topic: %s", chosen["title"])
    return chosen


def fetch_source_text(url: str, *, max_chars: int = 6000) -> str:
    """Fetch and reduce a source page to readable text. Resilient to failures."""
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
        # Prefer <article>, then <main>, then body
        node = soup.find("article") or soup.find("main") or soup.body or soup
        text = " ".join(node.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as e:  # noqa: BLE001
        LOG.warning("fetch_source_text(%s) failed: %s", url, e)
        return ""


def build_research_brief(llm: LLMRouter, topic: dict) -> dict[str, Any]:
    template = (repo_root() / "prompts" / "research.txt").read_text(encoding="utf-8")

    source_url = topic.get("external_url") or topic.get("url", "")
    source_text = topic.get("summary") or ""
    if len(source_text) < 300 and source_url and not source_url.startswith("https://reddit.com"):
        fetched = fetch_source_text(source_url)
        if fetched:
            source_text = fetched

    prompt = template.format(
        topic_title=topic.get("title", ""),
        angle=topic.get("angle", ""),
        source_url=source_url,
        source_text=source_text or "(no source text available)",
    )

    brief = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.4,
    )
    LOG.info("Research brief: %d facts, %d gaps",
             len(brief.get("key_facts", [])), len(brief.get("open_questions", [])))
    return brief
