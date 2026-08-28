"""Theme pool — a pool of seed ideas that feed into the candidate pool.

Provides a list of topic ideas aligned with the channel identity:
"surprising technology that matters to ordinary people".
These are used to inject strong fallback candidates into the discovery stage.
"""
from __future__ import annotations

import random
from typing import Iterable

# ----------------------------------------------------------------------------
# Curated building blocks
# ----------------------------------------------------------------------------

SEED_IDEAS: list[str] = [
    # consumer_tech_hidden
    "The surprising origin of WiFi",
    "Why USB drives have that logo",
    "How captchas secretly train AI",
    "The dark history behind the QWERTY keyboard",
    "What nobody tells you about Bluetooth",
    "The bizarre accident that created fiber optics",
    # ai_surprises
    "AI voice cloning scams",
    "How AI is reading ancient scrolls",
    "The open-source AI that scares big tech",
    "Why ChatGPT cannot draw hands",
    "On-device LLMs replacing Siri",
    # digital_scams & tech_mistakes
    "The SIM swapping nightmare",
    "How public WiFi actually steals data",
    "The MFA bypass attack",
    "Why you should never use public USB chargers",
    "The truth about password managers",
    # phone_internet_mechanics
    "How your phone tracks you while turned off",
    "The undersea cables that power the internet",
    "Why 5G drained your battery",
    "The hidden mechanics of the App Store",
    # consumer_myths
    "Does closing background apps save battery?",
    "The incognito mode myth",
    "Why more megapixels don't mean better photos",
    "The truth about fast charging your phone",
]


def pick_seeds(n: int, rng: random.Random | None = None) -> list[str]:
    """Pick `n` random seed ideas from the pool."""
    rng = rng or random
    if len(SEED_IDEAS) >= n:
        return rng.sample(SEED_IDEAS, n)
    return SEED_IDEAS[:]


def _cli() -> int:
    import argparse, json

    parser = argparse.ArgumentParser(description="Inspect the theme pool")
    parser.add_argument("--list", action="store_true", help="Print every seed idea")
    parser.add_argument("--pick", type=int, help="Pick N random seeds and print them")
    args = parser.parse_args()

    if args.list:
        for t in SEED_IDEAS:
            print(t)

    if args.pick:
        picked = pick_seeds(args.pick)
        print(json.dumps(picked, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
