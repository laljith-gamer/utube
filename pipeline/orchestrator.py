"""Top-level orchestrator. ALL configuration via config/*.yaml — no constants here.

Modes
-----
    # Default (used by the daily workflow): random batch from the global theme pool.
    python -m pipeline.orchestrator --random-batch

    # Pick within a single lane.
    python -m pipeline.orchestrator --random-batch --lane tech_news

    # Force a specific theme (manual debugging / regenerating one video).
    python -m pipeline.orchestrator --theme tech_news__why-seed-is-making-headlines-right-now__chatgpt-updates

    # Build but skip YouTube upload.
    python -m pipeline.orchestrator --random-batch --no-upload

    # Skip SVD animation for a faster build.
    python -m pipeline.orchestrator --random-batch --skip-svd
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import themes as themes_mod
from .config import get_config
from .ledger import Ledger
from .providers.image import ImageRouter
from .providers.llm import LLMRouter
from .providers.stock import StockRouter
from .providers.tts import TTSRouter
from .providers.video import VideoRouter
from .providers.youtube import upload_video
from .stages import assemble, audio, captions, discover, research, script, thumbnail, visuals
from .utils import env_bool, repo_root, run_date, run_dir, setup_logging, slugify, write_json

LOG = logging.getLogger("utube.orchestrator")


def produce_one(slot: dict, *, upload: bool, skip_svd: bool, script_only: bool, ledger: Ledger) -> dict:
    cfg = get_config()
    slot_id = slot["id"]                        # unique per video (theme id)
    lane_id = slot.get("lane", slot_id)         # for per-lane topic dedup
    out = run_dir(slot_id)
    LOG.info("=" * 72)
    LOG.info("THEME %s (lane=%s)", slot_id, lane_id)
    LOG.info("Title seed: %s", slot.get("title_seed", slot.get("title", "")))
    LOG.info("Output dir: %s", out)
    LOG.info("=" * 72)

    llm_research = LLMRouter("llm_research")
    llm_script = LLMRouter("llm_script")
    img = ImageRouter()
    vid = VideoRouter()
    tts = TTSRouter()
    stock = StockRouter()

    result: dict = {"slot": slot_id, "lane": lane_id, "ok": False, "out_dir": str(out)}
    try:
        # 1. Discover
        candidates = discover.discover_for_niche(slot)
        # Seed-inject the chosen theme so the LLM topic-picker has a strong on-brand option.
        # Stays additive — the LLM is still free to pick a fresher discovered candidate.
        seed_title = slot.get("title_seed")
        if seed_title:
            candidates = [{
                "title":   seed_title,
                "url":     "",
                "score":   9999,
                "summary": seed_title,
                "source":  "theme_seed",
            }] + candidates
        write_json(out / "candidates.json", candidates)
        if not candidates:
            raise RuntimeError("No candidates discovered for this niche")

        # 2. Research / pick
        topic = research.select_topic(
            llm_research,
            candidates,
            niche_title=slot.get("title", lane_id),
            sources_label=", ".join(s.get("type", "") for s in slot.get("sources", [])),
            recent_hashes=ledger.recent_hashes(lane_id, days=int(cfg.get_path("dedup_days", 30))),
        )
        write_json(out / "topic.json", topic)
        ledger.record_topic(lane_id, topic["topic_hash"])

        brief = research.build_research_brief(llm_research, topic)
        write_json(out / "research.json", brief)

        # 3. Script
        sc = script.generate_script(llm_script, slot=slot, topic=topic, research=brief)
        write_json(out / "script.json", sc)

        if script_only:
            LOG.info("--script-only set; stopping after script generation")
            result["ok"] = True
            result["title"] = sc.get("title")
            result["script"] = sc
            write_json(out / "result.json", result)
            return result

        import concurrent.futures

        # We can parallelize the generation of audio/captions, visuals, and the thumbnail.
        # Captions depend on audio, so they are grouped together.
        def _do_audio_and_captions():
            a_sum = audio.synthesize_narration(tts, script=sc, slot=slot, out_dir=out)
            srt_p = out / "captions.srt"
            captions.transcribe_to_srt(out / a_sum["master"], srt_p)
            return a_sum, srt_p

        def _do_visuals():
            if skip_svd:
                cfg["visuals"] = {**cfg.get("visuals", {}), "skip_svd": True}
            v = visuals.generate_visuals(image=img, video=vid, stock=stock, script=sc, out_dir=out)
            write_json(out / "visuals.json", v)
            return v

        def _do_thumbnail():
            t_path = out / "thumbnail.jpg"
            thumbnail.make_thumbnail(
                image=img,
                prompt=sc.get("thumbnail_prompt", sc.get("title", "")),
                text=sc.get("thumbnail_text", sc.get("title", "")[:30]),
                out_path=t_path,
                palette=slot.get("palette"),
            )
            return t_path

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_audio = executor.submit(_do_audio_and_captions)
            future_visuals = executor.submit(_do_visuals)
            future_thumbnail = executor.submit(_do_thumbnail)

            audio_summary, srt = future_audio.result()
            vis = future_visuals.result()
            thumb_path = future_thumbnail.result()

        # 8. Assemble
        video_out = out / f"{slugify(sc['title'])}.mp4"
        assemble.assemble_video(
            visuals=vis,
            audio_summary=audio_summary,
            srt_path=srt,
            out_dir=out,
            output_path=video_out,
            music_path=_pick_music(slot.get("music_mood")),
        )
        result["video_path"] = str(video_out)
        result["thumbnail_path"] = str(thumb_path)
        result["title"] = sc.get("title")
        result["description"] = sc.get("description")
        result["hashtags"] = sc.get("hashtags", [])

        # 9. Upload
        if upload:
            publish_strategy = (cfg.get_path("publish_strategy", "immediate") or "immediate").lower()
            tags_max = int(cfg.get_path("youtube.tags_max", 30))
            hashtags_list: list[str] = sc.get("hashtags", [])
            tags = [h.lstrip("#") for h in hashtags_list][:tags_max]

            # Build description: append hashtags as clickable #tags if not already present
            desc = sc.get("description", "")
            hashtag_str = " ".join(
                h if h.startswith("#") else f"#{h}" for h in hashtags_list
            )
            if hashtag_str and hashtag_str not in desc:
                desc = f"{desc.rstrip()}\n\n{hashtag_str}"
            desc_max = int(cfg.get_path("youtube.description_max_chars", 5000))
            desc = desc[:desc_max]

            # Pre-upload validation: ensure video file exists and is non-empty
            if not video_out.exists() or video_out.stat().st_size < 1024:
                raise RuntimeError(
                    f"Output video missing or too small ({video_out.stat().st_size if video_out.exists() else 0} bytes): {video_out}"
                )
            LOG.info("Pre-upload check OK: %s (%.1f MB, captions=%s)",
                     video_out.name,
                     video_out.stat().st_size / 1_048_576,
                     "yes" if srt.exists() and srt.stat().st_size > 0 else "NO — captions missing!")

            if publish_strategy == "scheduled":
                publish_at = _publish_at_for_slot(slot.get("schedule_utc"))
                privacy = cfg.get_path("privacy_status_for_scheduled", "private")
            else:
                publish_at = None
                privacy = cfg.get_path("default_privacy", "public")

            up = upload_video(
                video_out,
                title=sc["title"],
                description=desc,
                tags=tags,
                publish_at_iso=publish_at,
                thumbnail_path=thumb_path,
                privacy_status=privacy,
            )
            result["upload"] = up
        else:
            LOG.info("--no-upload set; skipping YouTube upload")

        result["ok"] = True
        # Mark this theme as used only once the run actually produced a video.
        ledger.record_theme(slot_id)
        write_json(out / "result.json", result)
        return result
    except Exception as e:  # noqa: BLE001
        LOG.error("THEME %s failed: %s\n%s", slot_id, e, traceback.format_exc())
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        write_json(out / "result.json", result)
        return result


# ---------- helpers ----------

def _publish_at_for_slot(hhmm: str | None) -> str | None:
    if not hhmm or ":" not in hhmm:
        return None
    h, m = (int(x) for x in hhmm.split(":"))
    now = datetime.now(timezone.utc)
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _pick_music(mood: str | None) -> Path | None:
    if not mood:
        return None
    music_dir = repo_root() / "assets" / "music" / mood
    if music_dir.is_dir():
        tracks = sorted(p for p in music_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a"))
        if tracks:
            return random.choice(tracks)
    return None


# ---------- theme selection ----------

def _select_slots(cfg, ledger: Ledger, args) -> list[dict]:
    """Resolve CLI flags into a concrete list of slot dicts the orchestrator will run."""
    lanes = cfg.get("lanes", []) or []
    if not lanes:
        raise RuntimeError("No lanes defined. Check config/lanes.yaml.")

    # 1. --theme: force a single theme
    if args.theme:
        theme = themes_mod.find_theme(args.theme, lanes)
        if not theme:
            raise SystemExit(f"Unknown --theme {args.theme!r}")
        return [themes_mod.materialize_slot(theme, lanes)]

    # 2. --random-batch: pick N themes (optionally filtered to one lane)
    dedup_days = int(cfg.get_path("dedup_days", 30))
    used = ledger.recent_theme_ids(days=dedup_days)

    if args.random_batch:
        vmin = args.videos_min if args.videos_min is not None else int(cfg.get_path("videos_min", 1))
        vmax = args.videos_max if args.videos_max is not None else int(cfg.get_path("videos_max", 2))
        vmin, vmax = max(1, vmin), max(vmin, vmax)
        n = random.randint(vmin, vmax)
        LOG.info("Random batch: picking %d theme(s) from pool (min=%d max=%d)", n, vmin, vmax)
        picked = themes_mod.pick_themes(n, lanes_cfg=lanes, exclude_ids=used, only_lane=args.lane)
        return [themes_mod.materialize_slot(t, lanes) for t in picked]

    # 3. --lane only: produce one random theme inside that lane
    if args.lane:
        picked = themes_mod.pick_themes(1, lanes_cfg=lanes, exclude_ids=used, only_lane=args.lane)
        return [themes_mod.materialize_slot(t, lanes) for t in picked]

    # 4. No flags: produce one random theme from anywhere.
    picked = themes_mod.pick_themes(1, lanes_cfg=lanes, exclude_ids=used)
    return [themes_mod.materialize_slot(t, lanes) for t in picked]


# ---------- entry point ----------

def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="utube — produce daily videos")
    parser.add_argument("--random-batch", action="store_true",
                        help="Pick a random N (videos_min..videos_max) themes from the pool")
    parser.add_argument("--videos-min", type=int, default=None,
                        help="Override videos_min from schedule.yaml")
    parser.add_argument("--videos-max", type=int, default=None,
                        help="Override videos_max from schedule.yaml")
    parser.add_argument("--lane", default=None,
                        help="Restrict random pick to this lane id (e.g. tech_news)")
    parser.add_argument("--theme", default=None,
                        help="Force a specific theme id (skips random pick)")
    parser.add_argument("--no-upload", action="store_true", help="Skip YouTube upload")
    parser.add_argument("--skip-svd", action="store_true",
                        help="Skip SDXL+SVD; visuals stage uses only stock video and motion filler")
    parser.add_argument("--script-only", action="store_true",
                        help="Stop immediately after generating the JSON script (for testing)")
    args = parser.parse_args(argv)

    upload = not args.no_upload and not env_bool("DRY_RUN")

    cfg = get_config()
    ledger = Ledger.load(repo_root() / "ledger.json")

    slots = _select_slots(cfg, ledger, args)

    LOG.info("Run date: %s", run_date())
    LOG.info("Channel: %s", cfg.get_path("channel.name", "?"))
    LOG.info("Themes to run: %s", [s["id"] for s in slots])
    LOG.info("Upload: %s", upload)
    LOG.info("Publish strategy: %s",
             cfg.get_path("publish_strategy", "immediate"))

    results = []
    for slot in slots:
        results.append(produce_one(slot, upload=upload, skip_svd=args.skip_svd, script_only=args.script_only, ledger=ledger))
        ledger.save()

    write_json(repo_root() / "runs" / run_date() / "batch_summary.json", results)

    n_ok = sum(1 for r in results if r.get("ok"))
    LOG.info("=" * 72)
    LOG.info("BATCH SUMMARY: %d / %d themes ok", n_ok, len(results))
    for r in results:
        marker = "OK " if r.get("ok") else "FAIL"
        LOG.info("  [%s] %s — %s", marker, r["slot"], r.get("title") or r.get("error", ""))
    LOG.info("=" * 72)

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
