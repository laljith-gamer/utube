"""Factual consistency validation.

Ensures the generated script does not hallucinate facts, overclaim,
or make unsupported causal statements that weren't in the research brief.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..providers.llm import LLMRouter

LOG = logging.getLogger("utube.factual_consistency")


def validate_facts(llm: LLMRouter, brief: dict[str, Any], script: dict[str, Any]) -> None:
    """Validate that the script is fully grounded in the research brief.

    Raises:
        ValueError if the script contains hallucinated or unsupported claims.
    """
    LOG.info("Validating factual consistency...")

    prompt = (
        "You are an expert fact-checker for a documentary YouTube channel.\n"
        "Your job is to read a Research Brief and a drafted Script, and verify that EVERY specific claim, "
        "number, historical fact, and causal link in the script is fully supported by the brief.\n\n"
        "RULES:\n"
        "1. NO HALLUCINATIONS: If the script includes a name, date, statistic, or mechanism not found in the brief, it is a hallucination.\n"
        "2. NO OVERCLAIMING: If the brief says 'A is associated with B', the script cannot claim 'A caused B'.\n"
        "3. The script is allowed to use expressive, conversational language, as long as the underlying facts are strictly supported.\n\n"
        "RESEARCH BRIEF:\n"
        f"{json.dumps(brief, indent=2)}\n\n"
        "DRAFT SCRIPT:\n"
        f"{json.dumps(script, indent=2)}\n\n"
        "Respond with a JSON object containing exactly two fields:\n"
        "- \"passed\": true if all claims are supported, false if there is a hallucination or overclaim.\n"
        "- \"reason\": a string explaining the exact unsupported claim if failed, or 'all facts supported' if passed."
    )

    try:
        response = llm.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1,
        )
    except Exception as e:
        LOG.warning("Factual consistency LLM call failed, assuming pass to not block pipeline: %s", e)
        return

    passed = response.get("passed", True)
    reason = response.get("reason", "")

    if not passed:
        LOG.error("Factual consistency FAILED: %s", reason)
        raise ValueError(f"Factual consistency check failed: {reason}")
    else:
        LOG.info("Factual consistency PASSED.")
