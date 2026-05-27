"""TTS narration: per-segment + master MP3, audio bitrate/codec from config.

Per-segment silence is trimmed (controlled by tts.trim_silence in providers.yaml)
to remove the 200-300ms of dead air Edge-TTS pads each clip with. Across ~14
segments per video this saves several seconds of pause and gives the final cut
the wall-to-wall energy of pro Shorts.
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
        if trim:
            _trim_silence(mp3, silence_db=silence_db, keep_sec=keep_sec, codec=cfg.get("codec", "libmp3lame"), bitrate=cfg.get("bitrate", "128k"))
        dur = _probe_duration(mp3)
        per_scene.append({"name": name, "text": text, "file": str(mp3.relative_to(out_dir)), "duration": dur})
        seg_files.append(mp3)
        LOG.info("  TTS %s -> %.2fs", name, dur)

    master = audio_dir / "narration.mp3"
    _ffmpeg_concat(seg_files, master, codec=cfg.get("codec", "libmp3lame"), bitrate=cfg.get("bitrate", "128k"))
    master_dur = _probe_duration(master)

    summary = {
        "voice": voice,
        "master": str(master.relative_to(out_dir)),
        "master_duration": master_dur,
        "segments": per_scene,
    }
    (out_dir / "audio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Narration: %.2fs total across %d segments", master_dur, len(seg_files))
    return summary


def _trim_silence(path: Path, *, silence_db: int, keep_sec: float, codec: str, bitrate: str) -> None:
    """Trim leading + trailing silence from an MP3 in place.

    Uses the ``silenceremove,areverse,silenceremove,areverse`` idiom: the first
    pass trims leading silence; reversing the audio then trimming again removes
    what was originally the trailing silence (without touching mid-clip pauses).
    """
    tmp = path.with_suffix(".trim.mp3")
    af = (
        f"silenceremove=start_periods=1:start_silence={keep_sec}:start_threshold={silence_db}dB,"
        f"areverse,"
        f"silenceremove=start_periods=1:start_silence={keep_sec}:start_threshold={silence_db}dB,"
        f"areverse"
    )
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
        LOG.warning("silence trim failed for %s (leaving original): %s",
                    path.name, (res.stderr or "").strip().splitlines()[-1] if res.stderr else "?")


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
