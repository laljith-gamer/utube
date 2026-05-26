"""TTS narration: per-segment + master MP3, audio bitrate/codec from config."""
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
    cfg = get_config().get_path("audio", {}) or {}
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
        dur = _probe_duration(mp3)
        per_scene.append({"name": name, "text": text, "file": str(mp3.relative_to(out_dir)), "duration": dur})
        seg_files.append(mp3)
        LOG.info("  TTS %s → %.2fs", name, dur)

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
