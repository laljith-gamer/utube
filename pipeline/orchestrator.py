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


def _history_path() -> Path:
    """Single source of truth for the history file location."""
    return repo_root() / "assets" / "history.json"


def produce_one(slot: dict, *, upload: bool, skip_svd: bool, ledger: Ledger) -> dict:
    cfg = get_config()
    slot_id = slot["id"]
    out = run_dir(slot_id)
    started_at = datetime.now(timezone.utc).isoformat()
    LOG.info("=" * 72)
    LOG.info("SLOT %s — %s", slot_id, slot.get("title", ""))
    LOG.info("Output dir: %s", out)
    LOG.info("=" * 72)

    llm = LLMRouter()
    img = ImageRouter()
    vid = VideoRouter()
    tts = TTSRouter()
    stock = StockRouter()

    result: dict = {
        "slot": slot_id,
        "ok": False,
        "out_dir": str(out),
        "started_at": started_at,
        "run_date": run_date(),
    }
    current_stage = "init"
    try:
        dedup_days = int(cfg.get_path("dedup_days", 30))

        # 1. Discover
        current_stage = "discover"
        candidates = discover.discover_for_niche(slot)
        write_json(out / "candidates.json", candidates)
        if not candidates:
            raise RuntimeError("No candidates discovered for this niche")

        # Filter out already-covered source URLs (no video repetition)
        before_n = len(candidates)
        candidates = [
            c for c in candidates
            if not (
                ledger.is_duplicate_url(slot_id, c.get("url", ""), days=dedup_days)
                or ledger.is_duplicate_url(slot_id, c.get("external_url") or "", days=dedup_days)
            )
        ]
        if len(candidates) < before_n:
            LOG.info("URL dedup: %d -> %d candidates after history filter",
                     before_n, len(candidates))
        if not candidates:
            raise RuntimeError(
                "No fresh candidates after URL-history filter (all seen in last "
                f"{dedup_days} days)"
            )

        # 2. Research / pick
        current_stage = "research"
        topic = research.select_topic(
            llm,
            candidates,
            niche_title=slot.get("title", slot_id),
            sources_label=", ".join(s.get("type", "") for s in slot.get("sources", [])),
            recent_hashes=ledger.recent_hashes(slot_id, days=dedup_days),
            slot_angle_hint=slot.get("angle_hint", ""),
        )
        write_json(out / "topic.json", topic)
        ledger.record_topic(slot_id, topic["topic_hash"])
        ledger.record_source_url(slot_id, topic.get("url", ""))
        ledger.record_source_url(slot_id, topic.get("external_url") or "")

        brief = research.build_research_brief(llm, topic)
        write_json(out / "research.json", brief)

        # 3. Script
        current_stage = "script"
        sc = script.generate_script(llm, slot=slot, topic=topic, research=brief)
        write_json(out / "script.json", sc)

        # 4. Audio
        current_stage = "audio"
        audio_summary = audio.synthesize_narration(tts, script=sc, slot=slot, out_dir=out)

        # 5. Visuals
        current_stage = "visuals"
        if skip_svd:
            cfg["video"] = {**cfg.get("video", {}), "use_svd_for_n_scenes": 0}
        vis = visuals.generate_visuals(image=img, video=vid, stock=stock,
                                       script=sc, slot=slot, out_dir=out)
        write_json(out / "visuals.json", vis)

        # 6. Captions — extension follows captions.format (ass | srt)
        current_stage = "captions"
        sub_ext = (cfg.get_path("captions.format", "srt") or "srt").lower()
        srt = out / f"captions.{sub_ext}"
        captions.transcribe_to_srt(out / audio_summary["master"], srt)

        # 7. Thumbnail
        current_stage = "thumbnail"
        thumb_path = out / "thumbnail.jpg"
        thumbnail.make_thumbnail(
            image=img,
            prompt=sc.get("thumbnail_prompt", sc.get("title", "")),
            text=sc.get("thumbnail_text", sc.get("title", "")[:30]),
            out_path=thumb_path,
            palette=slot.get("palette"),
        )

        # 8. Assemble
        current_stage = "assemble"
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
        current_stage = "upload"
        upload_info = None
        if upload:
            publish_at = _publish_at_for_slot(slot.get("schedule_utc"))
            tags = [h.lstrip("#") for h in sc.get("hashtags", [])][:30]
            upload_info = upload_video(
                video_out,
                title=sc["title"],
                description=sc.get("description", ""),
                tags=tags,
                publish_at_iso=publish_at,
                thumbnail_path=thumb_path,
                privacy_status=cfg.get_path("privacy_status_for_scheduled", "private"),
            )
            result["upload"] = upload_info
        else:
            LOG.info("--no-upload set; skipping YouTube upload")

        # ------ record full video metadata to history ------
        completed_at = datetime.now(timezone.utc).isoformat()
        ledger.record_video({
            "run_date":       run_date(),
            "slot":           slot_id,
            "topic_hash":     topic["topic_hash"],
            "topic_title":    topic.get("title"),
            "source_url":     topic.get("url"),
            "external_url":   topic.get("external_url"),
            "video_title":    sc.get("title"),
            "video_slug":     slugify(sc.get("title", "")),
            "video_path":     str(video_out.relative_to(repo_root())),
            "thumbnail_path": str(thumb_path.relative_to(repo_root())),
            "scenes_count":   len(sc.get("scenes", []) or []),
            "real_video_clips": sum(1 for v in vis if "video" in v),
            "still_clips":      sum(1 for v in vis if "image" in v and "video" not in v),
            "duration_sec":   audio_summary.get("master_duration"),
            "voice":          slot.get("voice"),
            "youtube_id":     (upload_info or {}).get("id"),
            "youtube_url":    (upload_info or {}).get("url"),
            "started_at":     started_at,
            "completed_at":   completed_at,
            "ok":             True,
        })

        result["ok"] = True
        result["completed_at"] = completed_at
        write_json(out / "result.json", result)
        return result

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        LOG.error("SLOT %s failed at stage %r: %s\n%s", slot_id, current_stage, e, tb)
        ledger.record_error(slot_id=slot_id, stage=current_stage,
                            error=str(e), traceback=tb)
        result["error"] = str(e)
        result["stage"] = current_stage
        result["traceback"] = tb
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
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
        tracks = sorted(p for p in music_dir.iterdir()
                        if p.suffix.lower() in (".mp3", ".wav", ".m4a"))
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

    history_file = _history_path()
    ledger = Ledger.load(history_file)

    LOG.info("Run date: %s", run_date())
    LOG.info("Channel:  %s", cfg.get_path("channel.name", "?"))
    LOG.info("History:  %s", history_file)
    LOG.info("Stats:    %s", ledger.stats())
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
        info = r.get("title") or f"{r.get('stage', '?')}: {r.get('error', '')}"
        LOG.info("  [%s] %s — %s", marker, r["slot"], info)
    LOG.info("=" * 72)

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
