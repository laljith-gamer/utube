"""Top-level orchestrator. ALL configuration via config/*.yaml — no constants here.

Modes
-----
    # Default (used by the daily workflow): single best Short.
    python -m pipeline.orchestrator

    # Build but skip YouTube upload.
    python -m pipeline.orchestrator --no-upload

    # Skip SVD animation for a faster build.
    python -m pipeline.orchestrator --skip-svd
    
    # Generate script only.
    python -m pipeline.orchestrator --script-only
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
from .stages import (
    assemble,
    audio,
    captions,
    concept,
    content_memory,
    discover,
    research,
    script,
    script_qc,
    thumbnail,
    topic_scoring,
    visual_qc,
    visuals,
)
from .utils import env_bool, repo_root, run_date, run_dir, setup_logging, slugify, write_json

LOG = logging.getLogger("utube.orchestrator")


def produce_one(upload: bool, skip_svd: bool, script_only: bool, ledger: Ledger) -> dict:
    """The core pipeline logic to produce one high-quality Short."""
    cfg = get_config()
    qual_cfg = cfg.get("quality", {}) or {}
    
    # We will pick a run ID based on the timestamp to ensure uniqueness.
    run_id = f"run_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    out = run_dir(run_id)
    LOG.info("=" * 72)
    LOG.info("STARTING DATA-DRIVEN PIPELINE (run=%s)", run_id)
    LOG.info("Output dir: %s", out)
    LOG.info("=" * 72)

    llm_research = LLMRouter("llm_research")
    llm_script = LLMRouter("llm_script")
    llm_concept = LLMRouter("llm_concept")
    llm_qc = LLMRouter("llm_qc")
    
    img = ImageRouter()
    vid = VideoRouter()
    tts = TTSRouter()
    stock = StockRouter()
    
    mem = content_memory.ContentMemory()
    mem_ctx = mem.get_context_for_scoring()

    result: dict = {"run_id": run_id, "ok": False, "out_dir": str(out)}
    
    try:
        # 1. Discover
        LOG.info("--- Stage 1: Discover ---")
        candidates = discover.discover_all()
        
        # Inject seed ideas
        seed_ideas = themes_mod.pick_seeds(5)
        for seed in seed_ideas:
            candidates.append({
                "title": seed,
                "url": "",
                "source": "theme_seed",
                "summary": seed,
                "source_score": 1.0,
                "freshness_score": 1.0,
                "source_quality_score": 1.0,
                "keywords": []
            })
            
        write_json(out / "1_candidates.json", candidates)
        if not candidates:
            raise RuntimeError("No candidates discovered.")

        # 2. Score Candidates
        LOG.info("--- Stage 2: Topic Scoring ---")
        scored_candidates = topic_scoring.score_candidates(
            llm_research, candidates, mem_ctx
        )
        write_json(out / "2_scored_candidates.json", [c.__dict__ for c in scored_candidates])
        
        exploration_ratio = float(qual_cfg.get("exploration_ratio", 0.25))
        best_candidate = topic_scoring.select_best(scored_candidates, exploration_ratio)
        
        if not best_candidate:
            LOG.warning("Pipeline rejected all candidates. No good topic today.")
            result["ok"] = True
            result["reason"] = "No candidate passed minimum quality threshold."
            write_json(out / "result.json", result)
            return result
            
        # Record topic usage
        topic_hash = best_candidate.topic_hash
        ledger.record_topic("global", topic_hash)
        
        write_json(out / "2_best_topic.json", best_candidate.__dict__)
        
        # 3. Concept Generation
        LOG.info("--- Stage 3: Concept Generation ---")
        top_concept = concept.generate_concepts(llm_concept, best_candidate, mem_ctx)
        if not top_concept:
            LOG.warning("Failed to generate a valid concept.")
            result["reason"] = "Concept generation failed."
            write_json(out / "result.json", result)
            return result
            
        write_json(out / "3_concept.json", top_concept)

        # 4. Deep Research
        LOG.info("--- Stage 4: Deep Research ---")
        brief = research.deep_research(llm_research, best_candidate, top_concept)
        write_json(out / "4_research.json", brief)
        
        if brief.get("confidence", 0) < float(qual_cfg.get("min_fact_confidence", 90)):
            LOG.warning("Fact confidence too low (%s). Aborting.", brief.get("confidence"))
            result["reason"] = "Fact check failed."
            write_json(out / "result.json", result)
            return result

        # 5. Script & QC
        LOG.info("--- Stage 5: Script & QC ---")
        max_regen = int(qual_cfg.get("max_regenerations", 2))
        
        sc = None
        qc_result = None
        
        for attempt in range(max_regen + 1):
            LOG.info("Script generation attempt %d/%d", attempt + 1, max_regen + 1)
            sc = script.generate_script(
                llm_script, 
                topic=best_candidate, 
                concept=top_concept, 
                research=brief,
                previous_qc=qc_result
            )
            write_json(out / f"5_script_v{attempt+1}.json", sc)
            
            qc_result = script_qc.evaluate_script(llm_qc, sc, top_concept)
            write_json(out / f"5_qc_v{attempt+1}.json", qc_result)
            
            if qc_result["passed"]:
                break
                
        if not qc_result or not qc_result["passed"]:
            LOG.warning("Script failed QC after maximum regenerations.")
            result["reason"] = "Failed script QC."
            write_json(out / "result.json", result)
            return result

        if script_only:
            LOG.info("--script-only set; stopping after script generation")
            result["ok"] = True
            result["title"] = sc.get("title")
            result["script"] = sc
            write_json(out / "result.json", result)
            return result

        # 6. Visuals, Audio, Captions
        LOG.info("--- Stage 6: Visuals, Audio, Captions ---")
        import concurrent.futures

        def _do_audio_and_captions():
            a_sum = audio.synthesize_narration(tts, script=sc, slot={}, out_dir=out)
            captions_path = out / "captions.ass"
            captions_path = captions.transcribe_to_srt(out / a_sum["master"], captions_path)
            return a_sum, captions_path

        def _do_visuals():
            v = visuals.generate_visuals(image=img, video=vid, stock=stock, script=sc, out_dir=out)
            write_json(out / "6_visuals.json", v)
            return v

        def _do_thumbnail():
            t_path = out / "thumbnail.jpg"
            thumbnail.make_thumbnail(
                image=img,
                prompt=sc.get("thumbnail_prompt", sc.get("title", "")),
                text=sc.get("thumbnail_text", sc.get("title", "")[:30]),
                out_path=t_path,
                palette="vibrant", # Using default palette
            )
            return t_path

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_audio = executor.submit(_do_audio_and_captions)
            future_visuals = executor.submit(_do_visuals)
            future_thumbnail = executor.submit(_do_thumbnail)

            audio_summary, captions_file = future_audio.result()
            vis = future_visuals.result()
            thumb_path = future_thumbnail.result()
            
        # 7. Visual QC
        v_qc = visual_qc.evaluate_visuals(vis)
        write_json(out / "7_visual_qc.json", v_qc)
        if not v_qc["passed"]:
            LOG.warning("Visual QC failed. Aborting.")
            result["reason"] = "Failed visual QC."
            write_json(out / "result.json", result)
            return result

        # 8. Assemble
        LOG.info("--- Stage 8: Assemble ---")
        video_out = out / f"{slugify(sc['title'])}.mp4"
        assemble.assemble_video(
            visuals=vis,
            audio_summary=audio_summary,
            srt_path=captions_file,
            out_dir=out,
            output_path=video_out,
            music_path=_pick_music(sc.get("music_mood", "suspense")),
        )
        
        result["video_path"] = str(video_out)
        result["thumbnail_path"] = str(thumb_path)
        result["title"] = sc.get("title")
        result["description"] = sc.get("description")
        result["hashtags"] = sc.get("hashtags", [])

        # 9. Upload
        if upload:
            LOG.info("--- Stage 9: Upload ---")
            publish_strategy = (cfg.get_path("publish_strategy", "immediate") or "immediate").lower()
            tags_max = int(cfg.get_path("youtube.tags_max", 30))
            hashtags_list: list[str] = sc.get("hashtags", [])
            tags = [h.lstrip("#") for h in hashtags_list][:tags_max]

            desc = sc.get("description", "")
            hashtag_str = " ".join(
                h if h.startswith("#") else f"#{h}" for h in hashtags_list
            )
            if hashtag_str and hashtag_str not in desc:
                desc = f"{desc.rstrip()}\n\n{hashtag_str}"
            desc_max = int(cfg.get_path("youtube.description_max_chars", 5000))
            desc = desc[:desc_max]

            if publish_strategy == "scheduled":
                publish_at = _publish_at_for_slot(None) # Can be refined later
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
        
        # 10. Record metadata
        ledger.record_run({
            "run_id": run_id,
            "topic": best_candidate.__dict__,
            "concept": top_concept,
            "script": sc,
            "visual_qc": v_qc,
            "upload": result.get("upload", {}),
        })
        
        write_json(out / "result.json", result)
        return result
    except Exception as e:  # noqa: BLE001
        LOG.error("PIPELINE FAILED: %s\n%s", e, traceback.format_exc())
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


# ---------- entry point ----------

def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="utube — produce data-driven daily shorts")
    parser.add_argument("--no-upload", action="store_true", help="Skip YouTube upload")
    parser.add_argument("--skip-svd", action="store_true",
                        help="Skip SDXL+SVD; visuals stage uses only stock video and motion filler")
    parser.add_argument("--script-only", action="store_true",
                        help="Stop immediately after generating the JSON script (for testing)")
    # Keep some old args so github actions doesn't crash before being updated
    parser.add_argument("--random-batch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--videos-min", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--videos-max", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lane", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--theme", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    upload = not args.no_upload and not env_bool("DRY_RUN")

    cfg = get_config()
    
    if args.skip_svd:
        cfg["visuals"] = {**cfg.get("visuals", {}), "skip_svd": True}

    ledger = Ledger.load(repo_root() / "ledger.json")

    LOG.info("Run date: %s", run_date())
    LOG.info("Channel: %s", cfg.get_path("channel.name", "?"))
    LOG.info("Upload: %s", upload)

    result = produce_one(
        upload=upload,
        skip_svd=args.skip_svd,
        script_only=args.script_only,
        ledger=ledger
    )
    
    ledger.save()
    
    # Generate machine-readable daily summary
    summary_path = repo_root() / "runs" / run_date() / "daily_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, result)

    if result.get("ok"):
        if "reason" in result:
            LOG.info("PIPELINE COMPLETED GRACEFULLY: %s", result["reason"])
        else:
            LOG.info("PIPELINE SUCCESS: Generated %s", result.get("title", ""))
        return 0
    else:
        LOG.error("PIPELINE FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
