"""TTS narration: per-segment + master MP3, audio bitrate/codec from config.

Per segment we run a single ffmpeg pass that:
  1. Trims leading + trailing silence (kills the 200-300ms Edge-TTS dead air)
  2. Speeds up audio via `atempo` so the configured `tts.default_rate` is
     applied uniformly across providers (edge-tts can't handle SSML rate
     reliably; gTTS has no rate parameter at all — post-processing fixes both).

Across ~14 segments per video this saves several seconds of pause AND gives the
final cut a pro Shorts pace, all in one ffmpeg invocation per segment.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ..config import get_config
from ..providers.tts import TTSRouter

LOG = logging.getLogger("utube.audio")


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

    voice = slot.get("voice", "en-US-AriaNeural")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    segments = [("hook", script.get("hook", ""))]
    for i, sc in enumerate(script["scenes"]):
        segments.append((f"scene_{i:02d}", sc.get("narration", "")))
    if script.get("cta"):
        segments.append(("cta", script["cta"]))

    per_scene: list[dict] = []
    seg_files: list[Path] = []
    for name, text in segments:
        if not text.strip():
            continue
        mp3 = audio_dir / f"{name}.mp3"
        data = tts.synthesize(text, voice=voice)
        mp3.write_bytes(data)
        if trim or abs(atempo_factor - 1.0) > 0.005:
            _postprocess_segment(
                mp3,
                trim=trim,
                atempo_factor=atempo_factor,
                silence_db=silence_db,
                keep_sec=keep_sec,
                codec=cfg.get("codec", "libmp3lame"),
                bitrate=cfg.get("bitrate", "128k"),
            )
        dur = _probe_duration(mp3)
        per_scene.append({
            "name": name, "text": text,
            "file": str(mp3.relative_to(out_dir)),
            "duration": dur,
        })
        seg_files.append(mp3)
        LOG.info("  TTS %s -> %.2fs (atempo=%.3f)", name, dur, atempo_factor)

    master = audio_dir / "narration.mp3"
    _ffmpeg_concat(
        seg_files, master,
        codec=cfg.get("codec", "libmp3lame"),
        bitrate=cfg.get("bitrate", "128k"),
    )
    master_dur = _probe_duration(master)

    summary = {
        "voice": voice,
        "atempo": atempo_factor,
        "master": str(master.relative_to(out_dir)),
        "master_duration": master_dur,
        "segments": per_scene,
    }
    (out_dir / "audio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Narration: %.2fs total across %d segments (rate=%s)", master_dur, len(seg_files), rate_pct_str)
    return summary


# ---------- helpers ----------

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
    path: Path, *,
    trim: bool,
    atempo_factor: float,
    silence_db: int,
    keep_sec: float,
    codec: str,
    bitrate: str,
) -> None:
    """Single ffmpeg pass: optional silence trim + optional tempo shift.

    Uses `silenceremove,areverse,silenceremove,areverse` for trim so mid-clip
    pauses are untouched, then optionally chains an `atempo=N` filter.
    """
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
    if not af_parts:
        return

    af = ",".join(af_parts)
    tmp = path.with_suffix(".pp.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-af", af,
        "-c:a", codec, "-b:a", bitrate,
        str(tmp),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
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
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c:a", codec, "-b:a", bitrate, str(output),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    listfile.unlink(missing_ok=True)


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0
