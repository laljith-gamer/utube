"""Premium Rewrite Stage - Integrates Puter (Claude Opus) as a high-quality fallback/enhancer."""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_config, goal_summary
from ..providers.llm import LLMRouter
from ..utils import repo_root

LOG = logging.getLogger("utube.premium_rewrite")


def evaluate_and_rewrite(
    script: dict[str, Any],
    qc_result: dict[str, Any],
    topic: dict | None = None,
    concept: dict | None = None,
) -> dict[str, Any]:
    """Check if the script warrants a premium rewrite, and execute it if so."""
    cfg = get_config()
    
    # Check if Puter provider is configured and token is present
    try:
        router = LLMRouter("llm_premium_rewrite")
        if not router.active:
            LOG.info("Premium rewrite skipped (no active provider).")
            return script
    except Exception as e:
        LOG.warning("Could not initialize premium rewrite router: %s", e)
        return script
        
    scores = qc_result.get("scores", {})
    overall_score = qc_result.get("overall_score", 0)
    passed = qc_result.get("passed", False)
    
    # We trigger a premium rewrite if:
    # 1. The script failed QC
    # 2. Or the score is below the premium threshold (e.g. 80)
    target_score = 80
    
    if passed and overall_score >= target_score:
        LOG.info("Premium rewrite skipped (script score %s >= %s).", overall_score, target_score)
        return script
        
    LOG.info("Triggering premium rewrite. Passed: %s, Score: %s", passed, overall_score)
    
    template = (repo_root() / "prompts" / "premium_rewrite.txt").read_text(encoding="utf-8") if (repo_root() / "prompts" / "premium_rewrite.txt").exists() else _DEFAULT_REWRITE_PROMPT
    
    prompt = template.format(
        goal=goal_summary(),
        topic_title=topic.get("title", "") if topic else "",
        angle=concept.get("chosen_angle", "") if concept else "",
        script_json=json.dumps(script, indent=2),
        qc_feedback=qc_result.get("feedback", ""),
        qc_issues=json.dumps(qc_result.get("issues", [])),
    )
    
    try:
        new_script = router.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.2,
        )
        
        # Verify required fields
        required = ["hook", "scenes", "title", "description", "hashtags", "thumbnail_prompt"]
        missing = [k for k in required if k not in new_script]
        if missing:
            raise ValueError(f"Premium rewrite JSON missing fields: {missing}")
            
        LOG.info("Premium rewrite successful.")
        
        # Keep original learning and concept metadata
        if "_learning" in script:
            new_script["_learning"] = script["_learning"]
        if "_concept" in script:
            new_script["_concept"] = script["_concept"]
            
        return new_script
        
    except Exception as e:
        LOG.error("Premium rewrite failed, falling back to original script. Error: %s", e)
        return script


_DEFAULT_REWRITE_PROMPT = """{goal}

You are an elite, premium copywriter. We have a YouTube Shorts script that needs polish.
The script currently has some quality control issues or didn't meet our highest standards.

Topic: {topic_title}
Angle: {angle}

QC Feedback:
{qc_feedback}
QC Issues:
{qc_issues}

Original Script:
{script_json}

YOUR TASK:
Rewrite the script to dramatically improve its quality while preserving the following strict constraints:
1. DO NOT invent facts, numbers, quotes, or change dates/names.
2. DO NOT change the causal relationships or introduce unsupported claims.
3. Remove all fluff, AI buzzwords (e.g. "delve", "explore", "testament").
4. Ensure the spoken units (hook, scenes, cta) do not repeat 5-word phrases or concepts.
5. Make it sound completely natural and continuous, like a real person talking briskly but comfortably.
6. The hook MUST NOT be paraphrased in the scenes.
7. Keep the exact same JSON structure.
8. DO NOT change the number of scenes. You must keep the exact same number of scenes as the original script.

Return ONLY the rewritten JSON object.
"""
