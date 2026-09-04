"""Top-level orchestrator for the production Shorts pipeline."""
from __future__ import annotations

import argparse
import logging
import random
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import narration_archive, repetition, themes as themes_mod
from .config import get_config
from .ledger import Ledger
from .providers.image import ImageRouter
from .providers.llm import LLMRouter
from .providers.stock import StockRouter
from .providers.tts import TTSRouter
from .providers.video import VideoRouter
from .providers.youtube import upload_video
from .stages import assemble, audio, captions, concept, content_memory, discover, research, script, script_qc, thumbnail, topic_scoring, visual_qc, visuals
from .utils import env, env_bool, repo_root, run_date, run_dir, setup_logging, slugify, write_json

LOG = logging.getLogger("utube.orchestrator")


def _apply_dynamic_templates(cfg: dict) -> None:
    """Inject randomized dynamic settings into the pipeline configuration."""
    # 1. Script Archetype
    archetypes = ["The Standard Loop", "The Mythbuster", "The Deep Dive", "The Comparison"]
    if "script" not in cfg:
        cfg["script"] = {}
    cfg["script"]["archetype"] = random.choice(archetypes)
    LOG.info("Dynamic Template: Selected Archetype '%s'", cfg["script"]["archetype"])

    # 2. Caption Styling
    if "assemble" not in cfg:
        cfg["assemble"] = {}
    if "cinematic_caption_style" not in cfg["assemble"]:
        cfg["assemble"]["cinematic_caption_style"] = {}
        
    style = cfg["assemble"]["cinematic_caption_style"]
    
    fonts = ["DejaVu Sans", "Arial", "Impact", "Trebuchet MS", "Verdana"]
    style["fontname"] = random.choice(fonts)
    
    if "captions" not in cfg:
        cfg["captions"] = {}
        
    # ASS format override colors: \c&H<BB><GG><RR>&
    highlights = [
        "&H00FFFF&", # Yellow
        "&H00FF00&", # Green
        "&HFFFF00&", # Cyan
        "&H00A5FF&", # Orange
        "&H0000FF&", # Red
        "&HFF00FF&", # Magenta
    ]
    cfg["captions"]["highlight_color"] = random.choice(highlights)
    
    # 5=center, 2=bottom-center
    style["alignment"] = random.choice([2, 5])
    if style["alignment"] == 2:
        style["margin_v"] = random.randint(150, 300)
    else:
        style["margin_v"] = 0
        
    cfg["captions"]["fade_ms"] = random.choice([0, 50, 100, 150])
    cfg["captions"]["words_per_chunk"] = random.choice([1, 2, 3])
    
    LOG.info("Dynamic Template: Caption font='%s', highlight='%s', align=%d, chunk=%d", 
             style["fontname"], cfg["captions"]["highlight_color"], style["alignment"], cfg["captions"]["words_per_chunk"])


def produce_one(upload: bool, skip_svd: bool, script_only: bool, ledger: Ledger) -> dict:
    cfg = get_config()
    _apply_dynamic_templates(cfg)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    out = run_dir(run_id)
    llm_research = LLMRouter("llm_research")
    llm_script = LLMRouter("llm_script")
    img, vid, tts, stock = ImageRouter(), VideoRouter(), TTSRouter(), StockRouter()
    llm_vision = LLMRouter("llm_vision")
    mem_ctx = content_memory.ContentMemory().get_context_for_scoring()
    result: dict = {"run_id": run_id, "ok": False, "out_dir": str(out)}

    try:
        LOG.info("Running LLM preflight check to ensure models are available...")
        llm_research.chat([{"role": "user", "content": "Reply with the word 'pong'"}], max_tokens=5, temperature=0.1)
        LOG.info("LLM preflight check passed. Models are online.")
    except Exception as e:
        msg = f"LLM preflight check failed! All configured providers are down or returning errors. Details: {e}"
        LOG.error(msg)
        result.update({"error": msg, "traceback": traceback.format_exc()})
        write_json(out / "result.json", result)
        return result

    try:
        candidates = discover.discover_candidates()
        for seed in themes_mod.pick_seeds(5):
            candidates.append({"title": seed, "url": "", "source": "theme_seed", "summary": seed, "source_score": 1.0, "freshness_score": 1.0, "source_quality_score": 1.0, "keywords": []})
        write_json(out / "1_candidates.json", candidates)
        if not candidates:
            raise RuntimeError("No candidates discovered.")

        memory_days = int(cfg.get_path("memory_days", 30))
        recent_hashes = ledger.recent_hashes("global", days=memory_days)
        scored = topic_scoring.score_candidates(candidates, content_memory=mem_ctx, recent_hashes=recent_hashes)
        write_json(out / "2_scored_candidates.json", scored)
        
        max_candidate_attempts = 3
        blacklisted_hashes = set()
        best = None
        top_concept = None
        brief = None
        evaluated_candidates = []
        
        for candidate_attempt in range(max_candidate_attempts):
            available_scored = [c for c in scored if c.get("content_hash") not in blacklisted_hashes]
            best = topic_scoring.select_best(available_scored, exploration_ratio=float(cfg.get_path("topic_scoring.exploration_ratio", 0.25)))
            if not best:
                result.update({"ok": True, "reason": "No candidate passed minimum quality threshold."})
                write_json(out / "result.json", result)
                return result
                
            topic_hash = best.get("topic_hash", best.get("content_hash"))
            if topic_hash:
                ledger.record_topic("global", topic_hash)
            family = best.get("topic_family", best.get("family", ""))
            if family:
                ledger.record_family(family)
            write_json(out / f"2_best_topic_v{candidate_attempt+1}.json", best)

            top_concept = concept.generate_concept(best, content_memory=mem_ctx, out_dir=out)
            if not top_concept:
                LOG.warning("Concept generation rejected due to low score. Blacklisting candidate %s and retrying.", topic_hash)
                blacklisted_hashes.add(topic_hash)
                continue
            write_json(out / f"3_concept_v{candidate_attempt+1}.json", top_concept)

            brief = research.build_research_brief(llm_research, best, concept=top_concept)
            write_json(out / f"4_research_v{candidate_attempt+1}.json", brief)
            conf_val = brief.get("confidence", 0)
            try:
                conf_val = float(conf_val)
            except (ValueError, TypeError):
                conf_val = 0.0

            if conf_val < float(cfg.get_path("min_fact_confidence", 70)):
                LOG.warning("Fact confidence too low (%s) for candidate %s. Blacklisting and retrying.", conf_val, topic_hash)
                blacklisted_hashes.add(topic_hash)
                evaluated_candidates.append((conf_val, best, top_concept, brief))
                continue
                
            # If we reach here, we have a valid concept and research brief
            break
        else:
            if evaluated_candidates:
                evaluated_candidates.sort(key=lambda x: x[0], reverse=True)
                conf_val, best, top_concept, brief = evaluated_candidates[0]
                LOG.warning("Exhausted candidate attempts. Falling back to highest confidence candidate (%s) with confidence %s", best.get("topic_hash", best.get("content_hash", "")), conf_val)
            else:
                result.update({"ok": True, "reason": f"Failed to find a candidate that passes concept and fact checks after {max_candidate_attempts} attempts."})
                write_json(out / "result.json", result)
                return result

        lane = cfg.get_path("lane", {}) or {}
        sc = None
        qc_result = None
        rep_result = None
        max_attempts = int(cfg.get_path("script_qc.max_regenerations", 2)) + 1
        for attempt in range(max_attempts):
            sc = script.generate_script(
                llm_script, slot=lane, topic=best, concept=top_concept,
                research=brief, previous_qc=qc_result,
                previous_repetition=rep_result,
            )
            write_json(out / f"5_script_v{attempt+1}.json", sc)

            # ── Repetition check (deterministic, no LLM call) ──
            rep_result = repetition.RepetitionChecker().check_script(sc, history=narration_archive.load_recent())
            write_json(out / f"5_repetition_v{attempt+1}.json", {
                "passed": rep_result.passed,
                "intra_issues": rep_result.intra_issues,
                "cross_issues": rep_result.cross_issues,
                "flagged_phrases": rep_result.flagged_phrases[:10],
            })

            # ── Factual Consistency Check ──
            from .stages.factual_consistency import validate_facts
            try:
                validate_facts(LLMRouter("llm_qc"), brief, sc)
                fact_passed = True
                fact_reason = ""
            except ValueError as e:
                fact_passed = False
                fact_reason = str(e)
            
            write_json(out / f"5_factual_v{attempt+1}.json", {
                "passed": fact_passed,
                "reason": fact_reason,
            })
            
            if not fact_passed:
                LOG.warning("Script failed factual consistency on attempt %d: %s", attempt + 1, fact_reason)
                qc_result = {"passed": False, "feedback": fact_reason, "issues": ["hallucination_or_overclaim"]}
                continue # Force a retry by bypassing the break condition

            # ── Script QC (LLM-based quality evaluation) ──
            qc_result = script_qc.evaluate_script(sc, topic=best, concept=top_concept)
            write_json(out / f"5_qc_v{attempt+1}.json", qc_result)

            if qc_result.get("passed") and rep_result.passed:
                break

        if not qc_result or not qc_result.get("passed"):
            # Before failing outright, give the premium rewrite a chance
            from .stages.premium_rewrite import evaluate_and_rewrite
            
            for premium_attempt in range(3):
                sc = evaluate_and_rewrite(sc, qc_result, topic=best, concept=top_concept)
                # Re-run QC on the rewritten script
                qc_result = script_qc.evaluate_script(sc, topic=best, concept=top_concept)
                write_json(out / f"5_premium_qc_v{premium_attempt+1}.json", qc_result)
                
                if qc_result.get("passed"):
                    break
            
            if not qc_result.get("passed"):
                LOG.warning("Failed script QC (even after premium rewrite), but proceeding anyway to guarantee output.")
        else:
            # Script passed QC, but might still qualify for premium enhancement
            from .stages.premium_rewrite import evaluate_and_rewrite
            sc_enhanced = evaluate_and_rewrite(sc, qc_result, topic=best, concept=top_concept)
            if sc_enhanced is not sc:
                sc = sc_enhanced
                write_json(out / "5_premium_script.json", sc)

        # Repetition issues are advisory after exhausting attempts — proceed
        # with the best version but log a warning.
        if rep_result and not rep_result.passed:
            LOG.warning("Proceeding with repetition issues after %d attempts: %s",
                        max_attempts, rep_result.all_issues[:3])
        if script_only:
            result.update({"ok": True, "title": sc.get("title"), "script": sc})
            write_json(out / "result.json", result)
            return result

        # ── TTS synthesis + Audio Validation (ASR), with retry on a bad take ──
        # F5-TTS occasionally leaks fragments of the voice-clone reference audio
        # into the narration (or mistranscribes badly). Re-synthesizing usually
        # produces a clean take since inference isn't seed-pinned, so we retry
        # the whole narration before giving up.
        max_audio_attempts = int(cfg.get_path("audio_validation.max_retries", 2)) + 1
        audio_summary: dict = {}
        asr_text, asr_segments, asr_info = "", [], None
        for audio_attempt in range(max_audio_attempts):
            audio_summary = audio.synthesize_narration(tts, script=sc, slot=lane, out_dir=out)

            master_audio = out / audio_summary["master"]
            asr_text, asr_segments, asr_info = captions.transcribe_audio(master_audio)
            if not asr_text:
                break  # Whisper unavailable/disabled — nothing to validate against

            from .stages.audio_validation import validate_audio
            ref_text = cfg.get_path("tts.providers.f5_tts.params.ref_text", env("F5_REF_TEXT", ""))
            try:
                validate_audio(sc, asr_text, ref_text)
                write_json(out / f"audio_validation_v{audio_attempt + 1}.json", {"passed": True})
                break
            except ValueError as e:
                write_json(out / f"audio_validation_v{audio_attempt + 1}.json", {"passed": False, "reason": str(e)})
                if audio_attempt >= max_audio_attempts - 1:
                    raise
                LOG.warning(
                    "Audio validation failed on attempt %d/%d, re-synthesizing narration: %s",
                    audio_attempt + 1, max_audio_attempts, e,
                )

        # ── Write ASS ──
        captions_file = out / "captions.ass"
        captions.write_ass(asr_segments, asr_info, captions_file)
        vis = visuals.generate_visuals(image=img, video=vid, stock=stock, llm_vision=llm_vision, script=sc, out_dir=out)
        write_json(out / "6_visuals.json", vis)
        thumb_path = out / "thumbnail.jpg"
        thumb_palette = cfg.get_path("thumbnail.palette", None)
        if not isinstance(thumb_palette, list) or not thumb_palette:
            thumb_palette = ["#FFFFFF", "#000000", "#FFDD00"]
        thumbnail.make_thumbnail(image=img, prompt=sc.get("thumbnail_prompt", sc.get("title", "")), text=sc.get("thumbnail_text", sc.get("title", "")[:30]), out_path=thumb_path, palette=thumb_palette)

        v_qc = visual_qc.evaluate_visuals(vis)
        write_json(out / "7_visual_qc.json", v_qc)
        if not v_qc.get("passed"):
            LOG.warning("Failed visual QC, but proceeding anyway to guarantee output.")

        video_out = out / f"{slugify(sc['title'])}.mp4"
        assemble.assemble_video(visuals=vis, audio_summary=audio_summary, srt_path=captions_file, out_dir=out, output_path=video_out, music_path=_pick_music(sc.get("music_mood", "suspense")))
        result.update({"video_path": str(video_out), "thumbnail_path": str(thumb_path), "title": sc.get("title"), "description": sc.get("description"), "hashtags": sc.get("hashtags", [])})

        if upload:
            publish_strategy = (cfg.get_path("publish_strategy", "immediate") or "immediate").lower()
            tags = [h.lstrip("#") for h in sc.get("hashtags", [])][:int(cfg.get_path("youtube.tags_max", 30))]
            desc = sc.get("description", "")
            hashtag_str = " ".join(h if h.startswith("#") else f"#{h}" for h in sc.get("hashtags", []))
            if hashtag_str and hashtag_str not in desc:
                desc = f"{desc.rstrip()}\n\n{hashtag_str}"
            desc = desc[:int(cfg.get_path("youtube.description_max_chars", 5000))]
            publish_at = _publish_at_for_slot(None) if publish_strategy == "scheduled" else None
            privacy = cfg.get_path("privacy_status_for_scheduled", "private") if publish_strategy == "scheduled" else cfg.get_path("default_privacy", "public")
            result["upload"] = upload_video(video_out, title=sc["title"], description=desc, tags=tags, publish_at_iso=publish_at, thumbnail_path=thumb_path, privacy_status=privacy)

        result["ok"] = True
        ledger.record_run({"run_id": run_id, "topic": best, "concept": top_concept, "script": sc, "visual_qc": v_qc, "upload": result.get("upload", {})})

        # Archive narration for future cross-video repetition checking
        narration_archive.append(sc, run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat())
        write_json(out / "result.json", result)
        return result
    except Exception as exc:
        LOG.error("PIPELINE FAILED: %s\n%s", exc, traceback.format_exc())
        result.update({"error": str(exc), "traceback": traceback.format_exc()})
        write_json(out / "result.json", result)
        return result


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


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="utube — produce data-driven daily shorts")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--skip-svd", action="store_true")
    parser.add_argument("--script-only", action="store_true")
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
    result = produce_one(upload=upload, skip_svd=args.skip_svd, script_only=args.script_only, ledger=ledger)
    ledger.save()
    summary_path = repo_root() / "runs" / run_date() / "daily_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())