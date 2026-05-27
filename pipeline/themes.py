"""Theme pool — 1000+ video theme seeds, generated from curated angle x seed templates.

Why a generator and not a 1000-line YAML?
    - Reproducible: the same lane + angle + seed always yields the same theme id,
      so dedup across runs is trivial.
    - Reviewable: a small, structured Python file is easier to skim than 1000 YAML rows.
    - Extensible: add an angle or a seed and the pool grows by ~20-100 themes immediately.

Public API
----------
    all_themes(lanes_cfg)           -> list[dict]   every theme across every lane
    themes_for_lane(lane_id, ...)   -> list[dict]   filter to one lane
    pick_themes(n, ...)             -> list[dict]   random pick honouring dedup
    materialize_slot(theme, lanes)  -> dict         theme + lane preset -> orchestrator slot

Each theme is::

    {
        "id":         "did_you_know__the-surprising-origin-of-seed__wifi",
        "lane":       "did_you_know",
        "title_seed": "the surprising origin of WiFi",
        "keywords":   ["WiFi", "history", "fact", ...],
    }
"""
from __future__ import annotations

import logging
import random
from typing import Iterable

from .utils import slugify

LOG = logging.getLogger("utube.themes")


# ----------------------------------------------------------------------------
# Curated building blocks
# ----------------------------------------------------------------------------
# Each lane gets a list of "angles" (the framing the script will take) and a
# list of "seeds" (the concrete topic the angle is applied to).
# Theme count per lane = len(angles) * len(seeds).

THEME_TEMPLATES: dict[str, dict[str, list[str]]] = {
    # ------------------------------------------------------------------ 264
    "did_you_know": {
        "angles": [
            "the surprising origin of {seed}",
            "the dark history behind {seed}",
            "{seed}: a fact that will change how you see it",
            "what nobody tells you about {seed}",
            "the truth about {seed} hidden in plain sight",
            "how {seed} accidentally changed the world",
            "the forgotten story of {seed}",
            "{seed} explained in 30 seconds",
            "the science nobody teaches you about {seed}",
            "the bizarre invention behind {seed}",
            "the accident that created {seed}",
            "why {seed} works the way it does",
        ],
        "seeds": [
            "WiFi", "the QWERTY keyboard", "USB", "captchas", "the @ symbol",
            "RAM", "the computer mouse", "hard drives", "fiber optics", "JPEG",
            "GIF", "the first computer virus", "the first email", "the first website",
            "the smartphone", "GPS", "Bluetooth", "microchips", "the floppy disk",
            "the SIM card", "MP3", "video calls",
        ],
    },
    # ------------------------------------------------------------------ 220
    "tech_news": {
        "angles": [
            "what {seed} means for you this week",
            "why {seed} is making headlines right now",
            "{seed} just changed the game — here is how",
            "the breaking news about {seed} you missed",
            "everyone is talking about {seed} — here is why",
            "the {seed} announcement explained in 30s",
            "what is really happening with {seed}",
            "the silent shift in {seed} this month",
            "is {seed} the next big thing?",
            "the {seed} story everyone got wrong",
        ],
        "seeds": [
            "Apple's next move", "Google's AI strategy", "Meta's VR push",
            "Tesla's robotaxi plans", "Microsoft and OpenAI", "NVIDIA's monopoly",
            "Amazon's cloud war", "Samsung foldables", "the EU AI Act",
            "US chip export bans", "TikTok's ban saga", "X/Twitter changes",
            "the Rabbit R1", "the Apple Vision Pro", "ChatGPT updates",
            "Perplexity vs Google", "Anthropic's Claude", "Stripe's IPO",
            "Reddit going public", "Adobe AI lawsuits", "GitHub Copilot pricing",
            "the death of Google Search",
        ],
    },
    # ------------------------------------------------------------------ 220
    "ai_breakthrough": {
        "angles": [
            "the new {seed} model that just dropped",
            "{seed}: the AI breakthrough you missed",
            "how {seed} is rewriting AI overnight",
            "the secret behind {seed}'s sudden jump",
            "what {seed} actually does — explained",
            "why {seed} matters for the future",
            "the {seed} paper everyone is reading",
            "{seed} just beat humans at this task",
            "the open-source {seed} that scares big tech",
            "the surprising limits of {seed}",
        ],
        "seeds": [
            "Llama 4", "GPT-5", "Claude 4", "Gemini Ultra", "Mistral models",
            "DeepSeek", "Stable Diffusion 4", "FLUX", "video diffusion models",
            "agent frameworks", "RAG pipelines", "multimodal models",
            "on-device LLMs", "AI music generation", "AI voice cloning",
            "robotics foundation models", "AlphaFold-style biology AI",
            "reasoning models", "mixture of experts", "long-context windows",
            "tool-use agents", "world models",
        ],
    },
    # ------------------------------------------------------------------ 180
    "cybersecurity": {
        "angles": [
            "the {seed} hack you need to know about",
            "how attackers exploit {seed}",
            "{seed}: a security flaw hiding in your daily life",
            "the {seed} breach explained in 30 seconds",
            "why {seed} is the next battleground",
            "the simple fix for {seed} most people skip",
            "what hackers really do with {seed}",
            "the {seed} threat your antivirus won't catch",
            "how {seed} ended up everywhere",
            "the rise and fall of {seed}",
        ],
        "seeds": [
            "phishing", "ransomware", "deepfakes", "password leaks",
            "SIM swapping", "supply-chain attacks", "zero-day vulnerabilities",
            "public WiFi risks", "smart-home cameras", "biometric spoofing",
            "social engineering", "USB-drop attacks", "MFA bypasses",
            "browser extensions", "cloud misconfigurations", "OAuth abuse",
            "Bluetooth attacks", "IoT botnets",
        ],
    },
    # ------------------------------------------------------------------ 220
    "viral_science": {
        "angles": [
            "the wild science of {seed}",
            "{seed}: the experiment that broke physics",
            "how {seed} is closer than you think",
            "the breakthrough in {seed} no one is covering",
            "what {seed} could mean for humanity",
            "the dark possibility of {seed}",
            "the day science cracked {seed}",
            "the story behind {seed} explained fast",
            "the future where {seed} is normal",
            "why {seed} just changed everything",
        ],
        "seeds": [
            "fusion energy", "quantum computing", "brain-computer interfaces",
            "gene editing", "longevity science", "room-temperature superconductors",
            "Mars colonization", "asteroid mining", "lab-grown meat",
            "lab-grown organs", "exoskeletons", "autonomous vehicles",
            "supersonic flight", "space tourism", "neutrino physics",
            "dark matter", "the multiverse", "time crystals",
            "anti-aging drugs", "mind uploading", "holographic displays",
            "James Webb telescope discoveries",
        ],
    },
}


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _build_themes_for_lane(lane: dict) -> list[dict]:
    lane_id = lane["id"]
    bundle = THEME_TEMPLATES.get(lane_id)
    if not bundle:
        LOG.warning("No theme templates for lane %r — lane will produce 0 themes", lane_id)
        return []

    default_kw = list(lane.get("default_keywords", []) or [])
    out: list[dict] = []
    for angle in bundle["angles"]:
        angle_slug = slugify(angle.replace("{seed}", "X"), max_len=40)
        for seed in bundle["seeds"]:
            seed_slug = slugify(seed, max_len=40)
            theme_id = f"{lane_id}__{angle_slug}__{seed_slug}"
            title_seed = angle.replace("{seed}", seed)
            out.append({
                "id": theme_id,
                "lane": lane_id,
                "title_seed": title_seed,
                "keywords": [seed] + default_kw,
            })
    return out


def all_themes(lanes_cfg: Iterable[dict]) -> list[dict]:
    """Return every theme across every lane in `lanes_cfg`."""
    out: list[dict] = []
    for lane in lanes_cfg:
        out.extend(_build_themes_for_lane(lane))
    return out


def themes_for_lane(lane_id: str, lanes_cfg: Iterable[dict]) -> list[dict]:
    for lane in lanes_cfg:
        if lane["id"] == lane_id:
            return _build_themes_for_lane(lane)
    return []


# ----------------------------------------------------------------------------
# Picking
# ----------------------------------------------------------------------------

def pick_themes(
    n: int,
    *,
    lanes_cfg: Iterable[dict],
    exclude_ids: set[str] | None = None,
    only_lane: str | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """Pick `n` random distinct themes, skipping any in `exclude_ids`.

    If the filtered pool is smaller than n (e.g. very long dedup window) we fall
    back to ignoring the exclusion to avoid producing fewer videos than asked.
    """
    rng = rng or random
    exclude = exclude_ids or set()

    pool = themes_for_lane(only_lane, lanes_cfg) if only_lane else all_themes(lanes_cfg)
    fresh = [t for t in pool if t["id"] not in exclude]
    if len(fresh) >= n:
        return rng.sample(fresh, n)

    LOG.warning(
        "Theme pool exhausted under dedup (%d fresh < %d needed); falling back to full pool",
        len(fresh), n,
    )
    if len(pool) >= n:
        return rng.sample(pool, n)
    # extreme edge case (lane mis-configured); allow repeats
    return [rng.choice(pool) for _ in range(n)] if pool else []


def find_theme(theme_id: str, lanes_cfg: Iterable[dict]) -> dict | None:
    for t in all_themes(lanes_cfg):
        if t["id"] == theme_id:
            return t
    return None


# ----------------------------------------------------------------------------
# Materialisation: theme -> orchestrator-shaped "slot"
# ----------------------------------------------------------------------------

def materialize_slot(theme: dict, lanes_cfg: Iterable[dict]) -> dict:
    """Merge lane preset + theme to produce a slot dict the orchestrator can run.

    The orchestrator's `produce_one(slot)` only reads:
        id, title, sources, voice, voice_style, style, palette, music_mood
    Plus our additions:
        lane            -> used for ledger topic dedup (per-lane history)
        title_seed      -> injected as a high-priority candidate in discover
        theme_keywords  -> available to downstream stages if needed
    """
    lanes_by_id = {l["id"]: l for l in lanes_cfg}
    lane = lanes_by_id.get(theme["lane"])
    if not lane:
        raise KeyError(f"Theme {theme['id']} references unknown lane {theme['lane']}")

    return {
        # lane preset (visual + audio + sources stay identical to the old slot shape)
        "sources":     lane.get("sources", []),
        "voice":       lane.get("voice"),
        "voice_style": lane.get("voice_style"),
        "style":       lane.get("style"),
        "palette":     lane.get("palette"),
        "music_mood":  lane.get("music_mood"),
        # theme overrides
        "id":             theme["id"],            # unique per video -> unique run dir
        "lane":           theme["lane"],          # for ledger topic dedup
        "title":          theme["title_seed"],    # used as niche_title in prompts
        "title_seed":     theme["title_seed"],    # for orchestrator's seed-candidate inject
        "theme_keywords": theme.get("keywords", []),
    }


# ----------------------------------------------------------------------------
# CLI: `python -m pipeline.themes` for local sanity checks
# ----------------------------------------------------------------------------

def _cli() -> int:
    import argparse, json, sys
    from .config import get_config

    parser = argparse.ArgumentParser(description="Inspect the theme pool")
    parser.add_argument("--list", action="store_true", help="Print every theme id")
    parser.add_argument("--count", action="store_true", help="Print theme counts per lane and total")
    parser.add_argument("--lane", help="Filter to one lane id")
    parser.add_argument("--pick", type=int, help="Pick N random themes and print them")
    args = parser.parse_args()

    lanes = get_config().get("lanes", []) or []
    pool = themes_for_lane(args.lane, lanes) if args.lane else all_themes(lanes)

    if args.count or not (args.list or args.pick):
        per_lane: dict[str, int] = {}
        for t in pool:
            per_lane[t["lane"]] = per_lane.get(t["lane"], 0) + 1
        for k, v in per_lane.items():
            print(f"{k:24s} {v}")
        print(f"{'TOTAL':24s} {len(pool)}")

    if args.list:
        for t in pool:
            print(t["id"])

    if args.pick:
        picked = pick_themes(args.pick, lanes_cfg=lanes, only_lane=args.lane)
        print(json.dumps(picked, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
