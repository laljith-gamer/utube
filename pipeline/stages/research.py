"""Research stage — topic selection + research brief generation.

The old `select_topic()` is kept for backward compatibility but the new pipeline
uses `build_research_brief()` with a concept object instead of a raw topic.
Token budgets and reasoning effort come from pipeline.yaml > research.
"""
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
    """Legacy topic selection via LLM. Kept for backward compatibility.

    The new pipeline uses topic_scoring + concept stages instead.
    """
    if not candidates:
        raise RuntimeError("No candidates to choose from")

    cfg = get_config()
    rcfg = _research_cfg().get("topic_select", {}) or {}
    template = (repo_root() / "prompts" / "topic_select.txt").read_text(encoding="utf-8")

    candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

    rendered = "\n".join(
        f"[{i}] ({c.get('source','')}, normalized hotness: {c.get('score',0)}/100) {c.get('title','')}\n    URL: {c.get('url','')}"
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
    """Fetch and extract readable text from a URL."""
    if max_chars is None:
        max_chars = int(_research_cfg().get("source_text_max_chars", 6000))
    if not url:
        return ""
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
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


def build_research_brief(
    llm: LLMRouter,
    topic: dict,
    *,
    concept: dict | None = None,
) -> dict[str, Any]:
    """Build a research brief for the selected topic/concept.

    If a concept is provided, uses the concept's angle and curiosity gap
    to focus the research on specific facts needed for that framing.
    """
    cfg = get_config()
    bcfg = _research_cfg().get("brief", {}) or {}
    template = (repo_root() / "prompts" / "research.txt").read_text(encoding="utf-8")

    source_url = topic.get("external_url") or topic.get("url", "")
    source_text = topic.get("summary") or ""
    if len(source_text) < 300 and source_url and not source_url.startswith("https://reddit.com"):
        fetched = fetch_source_text(source_url)
        if fetched:
            source_text = fetched

    # Use concept angle if available, otherwise fall back to topic angle
    angle = ""
    if concept:
        angle = concept.get("chosen_angle", "")
    if not angle:
        angle = topic.get("angle", "")

    evidence_text = "\n\nEVENT EVIDENCE PACKET:\n" + str(topic.get("evidence_packet", {})) if topic.get("evidence_packet") else ""
    
    # Do more research using Brave to fetch extra context and grounded AI answers
    try:
        from ..providers.brave import BraveProvider
        extra_news = BraveProvider.search_news(topic.get("title", ""), count=3)
        if extra_news:
            evidence_text += "\n\nADDITIONAL BRAVE SEARCH CONTEXT:\n"
            for n in extra_news:
                evidence_text += f"- {n.get('title')}: {n.get('summary')}\n"
        
        # Use the newly available Answers plan for deep grounded context
        grounded_answer = BraveProvider.get_answer(topic.get("title", ""))
        if grounded_answer:
            evidence_text += f"\n\nDEEP AI-GROUNDED RESEARCH ANSWER:\n{grounded_answer}\n"
            
    except Exception as e:
        LOG.warning("Failed to fetch extra Brave context: %s", e)
    
    prompt = template.format(
        goal=goal_summary(),
        topic_title=topic.get("title", ""),
        angle=angle,
        source_url=source_url,
        source_text=(source_text or "(no source text available)") + evidence_text,
        target_duration=cfg.get_path("video.target_duration_sec", 35),
        format_label=cfg.get_path("channel.format", "shorts"),
    )

    brief = llm.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=int(bcfg.get("max_tokens", 3000)),
        temperature=float(bcfg.get("temperature", 0.4)),
        reasoning_effort=bcfg.get("reasoning_effort"),
    )

    # Enrich brief with concept metadata if available
    if concept:
        brief["concept_angle"] = concept.get("chosen_angle", "")
        brief["concept_hook_type"] = concept.get("hook_type", "")
        brief["concept_curiosity_gap"] = concept.get("curiosity_gap", "")
        brief["concept_payoff"] = concept.get("payoff", "")

    LOG.info("Research brief: %d facts, %d gaps",
             len(brief.get("key_facts", [])), len(brief.get("open_questions", [])))
    return brief
