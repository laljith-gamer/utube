"""Top-level orchestrator. ALL configuration via config/*.yaml — no constants here.

Modes:
  - python -m pipeline.orchestrator                        # all niche slots, full pipeline
  - python -m pipeline.orchestrator --slot tech_news       # one slot only
  - python -m pipeline.orchestrator --no-upload            # build but skip upload
  - python -m pipeline.orchestrator --slot tech_news --no-upload --skip-svd
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def produce_one(slot: dict, *, upload: bool, skip_svd: bool, ledger: Ledger) -> dict:
    cfg = get_config()
    slot_id = slot["id"]
    out = run_dir(slot_id)
    LOG.info("=" * 72)
    LOG.info("SLOT %s — %s", slot_id, slot.get("title", ""))
    LOG.info("Output dir: %s", out)
    LOG.info("=" * 72)

    llm = LLMRouter()
    img = ImageRouter()
    vid = VideoRouter()
    tts = TTSRouter()
    stock = StockRouter()

    result: dict = {"slot": slot_id, "ok": False, "out_dir": str(out)}
    try:
        # 1. Discover
        candidates = discover.discover_for_niche(slot)
        write_json(out / "candidates.json", candidates)
        if not candidates:
            raise RuntimeError("No candidates discovered for this niche")

        # 2. Research / pick
        topic = research.select_topic(
            llm,
            candidates,
            niche_title=slot.get("title", slot_id),
            sources_label=", ".join(s.get("type", "") for s in slot.get("sources", [])),
            recent_hashes=ledger.recent_hashes(slot_id, days=int(cfg.get_path("dedup_days", 30))),
            slot_angle_hint=slot.get("angle_hint", ""),
        )
        write_json(out / "topic.json", topic)
        ledger.record_topic(slot_id, topic["topic_hash"])

        brief = research.build_research_brief(llm, topic)
        write_json(out / "research.json", brief)

        # 3. Script
        sc = script.generate_script(llm, slot=slot, topic=topic, research=brief)
        write_json(out / "script.json", sc)

        # 4. Audio
        audio_summary = audio.synthesize_narration(tts, script=sc, slot=slot, out_dir=out)

        # 5. Visuals
        if skip_svd:
            # Disable SVD by zeroing its allocation just for this run
            cfg["video"] = {**cfg.get("video", {}), "use_svd_for_n_scenes": 0}
        vis = visuals.generate_visuals(image=img, video=vid, stock=stock,
                                       script=sc, slot=slot, out_dir=out)
        write_json(out / "visuals.json", vis)

        # 6. Captions — extension follows captions.format (ass | srt)
        sub_ext = (cfg.get_path("captions.format", "srt") or "srt").lower()
        srt = out / f"captions.{sub_ext}"
        captions.transcribe_to_srt(out / audio_summary["master"], srt)

        # 7. Thumbnail
        thumb_path = out / "thumbnail.jpg"
        thumbnail.make_thumbnail(
            image=img,
            prompt=sc.get("thumbnail_prompt", sc.get("title", "")),
            text=sc.get("thumbnail_text", sc.get("title", "")[:30]),
            out_path=thumb_path,
            palette=slot.get("palette"),
        )

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
            publish_at = _publish_at_for_slot(slot.get("schedule_utc"))
            tags = [h.lstrip("#") for h in sc.get("hashtags", [])][:30]
            up = upload_video(
                video_out,
                title=sc["title"],
                description=sc.get("description", ""),
                tags=tags,
                publish_at_iso=publish_at,
                thumbnail_path=thumb_path,
                privacy_status=cfg.get_path("privacy_status_for_scheduled", "private"),
            )
            result["upload"] = up
        else:
            LOG.info("--no-upload set; skipping YouTube upload")

        result["ok"] = True
        write_json(out / "result.json", result)
        return result
    except Exception as e:  # noqa: BLE001
        LOG.error("SLOT %s failed: %s\n%s", slot_id, e, traceback.format_exc())
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
            import random
            return random.choice(tracks)
    return None


# ---------- entry point ----------

def main(argv: list[str] | None = None) -> int:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="utube — produce daily videos")
    parser.add_argument("--slot", help="Run only one slot id (e.g. tech_news). Omit to run all.")
    parser.add_argument("--no-upload", action="store_true", help="Skip YouTube upload")
    parser.add_argument("--skip-svd", action="store_true", help="Skip SVD animation (Ken Burns only)")
    args = parser.parse_args(argv)

    upload = not args.no_upload and not env_bool("DRY_RUN")

    cfg = get_config()
    slots = cfg.get("slots", []) or []
    if args.slot:
        slots = [s for s in slots if s["id"] == args.slot]
        if not slots:
            LOG.error("Unknown --slot %r. Available: %s",
                      args.slot, [s["id"] for s in cfg.get("slots", [])])
            return 2

    ledger = Ledger.load(repo_root() / "ledger.json")

    LOG.info("Run date: %s", run_date())
    LOG.info("Channel: %s", cfg.get_path("channel.name", "?"))
    LOG.info("Slots to run: %s", [s["id"] for s in slots])
    LOG.info("Upload: %s", upload)

    results = []
    for slot in slots:
        results.append(produce_one(slot, upload=upload, skip_svd=args.skip_svd, ledger=ledger))
        ledger.save()

    write_json(repo_root() / "runs" / run_date() / "batch_summary.json", results)

    n_ok = sum(1 for r in results if r.get("ok"))
    LOG.info("=" * 72)
    LOG.info("BATCH SUMMARY: %d / %d slots ok", n_ok, len(results))
    for r in results:
        marker = "OK " if r.get("ok") else "FAIL"
        LOG.info("  [%s] %s — %s", marker, r["slot"], r.get("title") or r.get("error", ""))
    LOG.info("=" * 72)

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
