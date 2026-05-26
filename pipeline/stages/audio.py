"""TTS narration — per-segment + master MP3.

Pacing controls (all from config/pipeline.yaml > pacing):
  - Per-niche voice_rate / voice_pitch override the pacing.default_*.
  - Optional silence trim at segment start/end (silenceremove ffmpeg filter).
  - Optional master tempo multiplier (atempo ffmpeg filter) to tighten the
    whole narration without re-rendering scenes.
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
    cfg = get_config()
    acfg = cfg.get_path("audio", {}) or {}
    pcfg = cfg.get_path("pacing", {}) or {}

    voice = slot.get("voice", "en-US-AriaNeural")
    # Per-slot pacing overrides → fall back to global defaults
    rate = slot.get("voice_rate") or pcfg.get("default_voice_rate", "+0%")
    pitch = slot.get("voice_pitch") or pcfg.get("default_voice_pitch", "+0Hz")
    trim_silence_enabled = bool(pcfg.get("trim_silence", False))
    silence_db = float(pcfg.get("silence_threshold_db", -45))
    silence_min_dur = float(pcfg.get("silence_min_duration_sec", 0.15))
    tempo = float(pcfg.get("master_tempo_multiplier", 1.0))

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "Audio: voice=%s rate=%s pitch=%s trim_silence=%s tempo=%.2fx",
        voice, rate, pitch, trim_silence_enabled, tempo,
    )

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
        raw_mp3 = audio_dir / f"{name}.raw.mp3"
        final_mp3 = audio_dir / f"{name}.mp3"

        data = tts.synthesize(text, voice=voice, rate=rate, pitch=pitch)
        raw_mp3.write_bytes(data)

        if trim_silence_enabled:
            _trim_silence(raw_mp3, final_mp3, threshold_db=silence_db, min_duration=silence_min_dur,
                          codec=acfg.get("codec", "libmp3lame"), bitrate=acfg.get("bitrate", "128k"))
            raw_mp3.unlink(missing_ok=True)
        else:
            raw_mp3.replace(final_mp3)

        dur = _probe_duration(final_mp3)
        per_scene.append({"name": name, "text": text, "file": str(final_mp3.relative_to(out_dir)), "duration": dur})
        seg_files.append(final_mp3)
        LOG.info("  TTS %s → %.2fs", name, dur)

    master = audio_dir / "narration.mp3"
    _ffmpeg_concat(seg_files, master, codec=acfg.get("codec", "libmp3lame"),
                   bitrate=acfg.get("bitrate", "128k"))

    # Optional master tempo speedup (1.0 = no change)
    if abs(tempo - 1.0) > 1e-3:
        sped = audio_dir / "narration.tempo.mp3"
        _atempo(master, sped, tempo, codec=acfg.get("codec", "libmp3lame"),
                bitrate=acfg.get("bitrate", "128k"))
        sped.replace(master)
        # Update per-segment durations proportionally so the assembler scales scenes
        for seg in per_scene:
            seg["duration"] = seg["duration"] / tempo

    master_dur = _probe_duration(master)

    summary = {
        "voice": voice,
        "voice_rate": rate,
        "voice_pitch": pitch,
        "tempo_multiplier": tempo,
        "master": str(master.relative_to(out_dir)),
        "master_duration": master_dur,
        "segments": per_scene,
    }
    (out_dir / "audio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Narration: %.2fs total across %d segments", master_dur, len(seg_files))
    return summary


# ---------- ffmpeg helpers ----------

def _trim_silence(src: Path, dst: Path, *, threshold_db: float, min_duration: float,
                  codec: str, bitrate: str) -> None:
    # silenceremove: trim leading silence, then reverse + trim again to remove trailing silence
    af = (
        f"silenceremove=start_periods=1:start_duration={min_duration}:start_threshold={threshold_db}dB:detection=peak,"
        f"aformat=dblp,areverse,"
        f"silenceremove=start_periods=1:start_duration={min_duration}:start_threshold={threshold_db}dB:detection=peak,"
        f"aformat=dblp,areverse"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", af,
        "-c:a", codec, "-b:a", bitrate,
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        LOG.warning("silence trim failed (%s); keeping original", res.stderr.splitlines()[-1] if res.stderr else "?")
        src.replace(dst)


def _atempo(src: Path, dst: Path, multiplier: float, *, codec: str, bitrate: str) -> None:
    # atempo accepts 0.5 to 2.0 per filter; chain for outside range
    chain = []
    m = multiplier
    while m > 2.0:
        chain.append("atempo=2.0"); m /= 2.0
    while m < 0.5:
        chain.append("atempo=0.5"); m *= 2.0
    chain.append(f"atempo={m:.4f}")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", ",".join(chain),
        "-c:a", codec, "-b:a", bitrate,
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


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
