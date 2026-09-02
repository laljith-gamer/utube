"""TTS narration: continuous master MP3, audio bitrate/codec from config.

The default path synthesizes the whole script in one pass so sentence-to-sentence
delivery stays smooth and conversational. We still store estimated per-section
durations for video assembly, but the audible narration is continuous.

An optional segmented mode remains available for debugging and provider fallback
experiments. Post-processing normalizes output to MP3, trims edge silence, and
can apply a small `atempo` speed adjustment when configured.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from ..config import get_config
from ..providers.tts import TTSRouter

LOG = logging.getLogger("utube.audio")
FFMPEG_TIMEOUT_SEC = 180
FFPROBE_TIMEOUT_SEC = 30


def synthesize_narration(
    tts: TTSRouter,
    *,
    script: dict,
    slot: dict,
    out_dir: Path,
) -> dict:
    full_cfg = get_config()
    cfg = full_cfg.get_path("audio", {}) or {}
    tts_cfg = full_cfg.get_path("tts", {}) or {}
    trim = bool(tts_cfg.get("trim_silence", True))
    silence_db = int(tts_cfg.get("silence_threshold_db", -45))
    keep_sec = float(tts_cfg.get("silence_keep_sec", 0.05))
    rate_pct_str = str(tts_cfg.get("default_rate", "+0%"))
    atempo_factor = _atempo_factor(rate_pct_str)
    continuous = bool(tts_cfg.get("continuous_narration", True))
    normalize = bool(tts_cfg.get("normalize_output", True))

    voice = slot.get("voice", "en-US-AriaNeural")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    segments = _script_segments(script)
    if not segments:
        raise RuntimeError("No narration text found in script")

    if continuous:
        return _synthesize_continuous(
            tts,
            segments=segments,
            voice=voice,
            audio_dir=audio_dir,
            out_dir=out_dir,
            trim=trim,
            atempo_factor=atempo_factor,
            silence_db=silence_db,
            keep_sec=keep_sec,
            codec=cfg.get("codec", "libmp3lame"),
            bitrate=cfg.get("bitrate", "128k"),
            normalize=normalize,
            rate_label=rate_pct_str,
        )

    return _synthesize_segmented(
        tts,
        segments=segments,
        voice=voice,
        audio_dir=audio_dir,
        out_dir=out_dir,
        trim=trim,
        atempo_factor=atempo_factor,
        silence_db=silence_db,
        keep_sec=keep_sec,
        codec=cfg.get("codec", "libmp3lame"),
        bitrate=cfg.get("bitrate", "128k"),
        normalize=normalize,
        rate_label=rate_pct_str,
    )


# ---------- synthesis modes ----------

def _synthesize_continuous(
    tts: TTSRouter,
    *,
    segments: list[tuple[str, str]],
    voice: str,
    audio_dir: Path,
    out_dir: Path,
    trim: bool,
    atempo_factor: float,
    silence_db: int,
    keep_sec: float,
    codec: str,
    bitrate: str,
    normalize: bool,
    rate_label: str,
) -> dict:
    master = audio_dir / "narration.mp3"
    full_text = " ".join(text.strip() for _, text in segments if text.strip())
    full_text = _sanitize_for_tts(full_text)
    data = tts.synthesize(full_text, voice=voice)
    master.write_bytes(data)
    _postprocess_segment(
        master,
        trim=trim,
        atempo_factor=atempo_factor,
        silence_db=silence_db,
        keep_sec=keep_sec,
        codec=codec,
        bitrate=bitrate,
        force_reencode=normalize,
    )
    master_dur = _probe_duration(master)
    per_scene = _estimate_segment_timings(segments, master_dur)

    summary = {
        "voice": voice,
        "mode": "continuous",
        "atempo": atempo_factor,
        "master": str(master.relative_to(out_dir)),
        "master_duration": master_dur,
        "segments": per_scene,
    }
    (out_dir / "audio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Narration: %.2fs continuous across %d sections (rate=%s)",
             master_dur, len(segments), rate_label)
    return summary


def _synthesize_segmented(
    tts: TTSRouter,
    *,
    segments: list[tuple[str, str]],
    voice: str,
    audio_dir: Path,
    out_dir: Path,
    trim: bool,
    atempo_factor: float,
    silence_db: int,
    keep_sec: float,
    codec: str,
    bitrate: str,
    normalize: bool,
    rate_label: str,
) -> dict:
    import concurrent.futures

    def _generate_segment(name: str, text: str) -> dict:
        mp3 = audio_dir / f"{name}.mp3"
        clean_text = _sanitize_for_tts(text)
        data = tts.synthesize(clean_text, voice=voice)
        mp3.write_bytes(data)
        _postprocess_segment(
            mp3,
            trim=trim,
            atempo_factor=atempo_factor,
            silence_db=silence_db,
            keep_sec=keep_sec,
            codec=codec,
            bitrate=bitrate,
            force_reencode=normalize,
        )
        dur = _probe_duration(mp3)
        LOG.info("  TTS %s -> %.2fs (atempo=%.3f)", name, dur, atempo_factor)
        return {
            "name": name,
            "text": text,
            "file": str(mp3.relative_to(out_dir)),
            "duration": dur,
            "_path": mp3,
        }

    per_scene: list[dict] = [{}] * len(segments)
    seg_files: list[Path] = [Path()] * len(segments)
    
    # PyTorch already multithreads internally. Running parallel inferences on CPU
    # causes severe thread contention and memory spikes. Run sequentially.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(_generate_segment, name, text): i for i, (name, text) in enumerate(segments)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            res = future.result()
            seg_files[i] = res.pop("_path")
            per_scene[i] = res

    master = audio_dir / "narration.mp3"
    _ffmpeg_concat(seg_files, master, codec=codec, bitrate=bitrate)
    master_dur = _probe_duration(master)

    summary = {
        "voice": voice,
        "mode": "segmented",
        "atempo": atempo_factor,
        "master": str(master.relative_to(out_dir)),
        "master_duration": master_dur,
        "segments": per_scene,
    }
    (out_dir / "audio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Narration: %.2fs total across %d segments (rate=%s)",
             master_dur, len(seg_files), rate_label)
    return summary


# ---------- helpers ----------

def _script_segments(script: dict) -> list[tuple[str, str]]:
    segments = [("hook", script.get("hook", ""))]
    for i, sc in enumerate(script["scenes"]):
        segments.append((f"scene_{i:02d}", sc.get("narration", "")))
    if script.get("cta"):
        segments.append(("cta", script["cta"]))
    return [(name, str(text)) for name, text in segments if str(text).strip()]


def _sanitize_for_tts(text: str) -> str:
    """Strip characters that break F5-TTS chunking (quotes, parens, symbols)."""
    t = text
    t = t.replace('"', '').replace("'", "")
    t = t.replace("(", ", ").replace(")", ", ")
    t = t.replace("$", " dollars ").replace("%", " percent ")
    t = t.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    t = t.replace("#", " hashtag ")
    # Clean up double spaces or floating commas
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\s+,\s*', ', ', t)
    return t.strip()


def _estimate_segment_timings(segments: list[tuple[str, str]], total_duration: float) -> list[dict]:
    weights = [_speech_weight(text) for _, text in segments]
    total_weight = sum(weights) or float(len(segments) or 1)
    cursor = 0.0
    result: list[dict] = []
    for i, ((name, text), weight) in enumerate(zip(segments, weights)):
        if i == len(segments) - 1:
            dur = max(0.0, total_duration - cursor)
        else:
            dur = max(0.0, total_duration * (weight / total_weight))
        start = cursor
        end = start + dur
        result.append({
            "name": name,
            "text": text,
            "start": start,
            "end": end,
            "duration": dur,
        })
        cursor = end
    return result


def _speech_weight(text: str) -> float:
    words = len(re.findall(r"\b[\w']+\b", text))
    sentence_breaks = sum(text.count(ch) for ch in ".!?")
    soft_breaks = sum(text.count(ch) for ch in ",;:")
    return max(1.0, words + (0.8 * sentence_breaks) + (0.35 * soft_breaks))


def _atempo_factor(rate_str: str) -> float:
    """Convert '+12%' / '-10%' / '+0%' -> 1.12 / 0.90 / 1.00.

    `atempo` accepts 0.5-2.0; we clamp to that to avoid ffmpeg errors.
    """
    s = (rate_str or "+0%").strip().replace("%", "")
    sign = -1.0 if s.startswith("-") else 1.0
    try:
        pct = float(s.lstrip("+-")) / 100.0
    except ValueError:
        return 1.0
    return max(0.5, min(2.0, 1.0 + sign * pct))


def _postprocess_segment(
    path: Path,
    *,
    trim: bool,
    atempo_factor: float,
    silence_db: int,
    keep_sec: float,
    codec: str,
    bitrate: str,
    force_reencode: bool = False,
) -> None:
    """Single ffmpeg pass: normalize codec, optional trim, optional tempo shift."""
    af_parts: list[str] = []
    if trim:
        af_parts.extend([
            f"silenceremove=start_periods=1:start_silence={keep_sec}:start_threshold={silence_db}dB",
            "areverse",
            f"silenceremove=start_periods=1:start_silence={keep_sec}:start_threshold={silence_db}dB",
            "areverse",
        ])
    if abs(atempo_factor - 1.0) > 0.005:
        af_parts.append(f"atempo={atempo_factor:.4f}")
    if not af_parts and not force_reencode:
        return

    tmp = path.with_suffix(".pp.mp3")
    cmd = ["ffmpeg", "-y", "-i", str(path)]
    if af_parts:
        cmd.extend(["-af", ",".join(af_parts)])
    elif force_reencode:
        cmd.extend(["-af", "anull"])
    cmd.extend(["-c:a", codec, "-b:a", bitrate, str(tmp)])

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        LOG.warning("audio post-process timed out for %s (leaving original)", path.name)
        return
    if res.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(path)
    else:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        last = (res.stderr or "").strip().splitlines()
        LOG.warning("audio post-process failed for %s (leaving original): %s",
                    path.name, last[-1] if last else "?")


def _ffmpeg_concat(files: list[Path], output: Path, *, codec: str, bitrate: str) -> None:
    listfile = output.with_suffix(".txt")
    listfile.write_text("\n".join(f"file '{f.resolve()}'" for f in files), encoding="utf-8")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(listfile),
        "-c", "copy",
        str(output),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as e:
        listfile.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg concat timed out for {output.name}") from e
    if res.returncode != 0:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c:a", codec, "-b:a", bitrate, str(output),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC)
    listfile.unlink(missing_ok=True)


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        LOG.warning("ffprobe timed out for %s", path.name)
        return 0.0
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0
