"""Assemble final MP4 with FFmpeg: per-scene clips/images + Ken Burns + captions + music."""
from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

LOG = logging.getLogger("utube.assemble")


def assemble_video(
    *,
    visuals: list[dict],
    audio_summary: dict,
    srt_path: Path,
    out_dir: Path,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    music_path: Path | None = None,
    caption_style: dict | None = None,
) -> Path:
    """Build the final video. Uses one ffmpeg invocation per scene then a final concat."""
    work_dir = out_dir / "assemble"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-scene durations from audio segments (hook is part of scene 0)
    seg_durations = _scene_durations(audio_summary, num_scenes=len(visuals))

    scene_clips: list[Path] = []
    for i, v in enumerate(visuals):
        dur = max(seg_durations[i], 1.5)
        out_clip = work_dir / f"scene_{i:02d}.mp4"
        if "video" in v:
            _render_from_video(out_dir / v["video"], out_clip, dur, width, height, fps)
        elif "image" in v:
            _render_from_image(out_dir / v["image"], out_clip, dur, width, height, fps)
        else:
            # Solid color fallback
            _render_solid(out_clip, dur, width, height, fps)
        scene_clips.append(out_clip)

    # Concat all scene clips → silent_video.mp4
    silent_video = work_dir / "silent.mp4"
    _concat(scene_clips, silent_video)

    # Mux narration + music (ducked) + burn captions in one pass
    narration = out_dir / audio_summary["master"]
    _final_mux(
        silent_video=silent_video,
        narration=narration,
        music=music_path,
        srt=srt_path,
        out=output_path,
        caption_style=caption_style or {},
    )
    LOG.info("Final video → %s", output_path)
    return output_path


# ---------- per-scene renderers ----------

def _render_from_image(img: Path, out: Path, dur: float, w: int, h: int, fps: int) -> None:
    """Ken Burns: slow zoom-in centered."""
    zoom_frames = int(dur * fps)
    zoom_inc = 0.0015
    vf = (
        f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
        f"crop={w*2}:{h*2},"
        f"zoompan=z='min(zoom+{zoom_inc},1.4)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={zoom_frames}:s={w}x{h}:fps={fps},"
        f"setsar=1"
    )
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(img),
        "-t", f"{dur:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _render_from_video(src: Path, out: Path, dur: float, w: int, h: int, fps: int) -> None:
    """Loop or trim source video to dur, scale/crop to portrait, set fps."""
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1,fps={fps}"
    )
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(src),
        "-t", f"{dur:.2f}",
        "-vf", vf,
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _render_solid(out: Path, dur: float, w: int, h: int, fps: int) -> None:
    color = random.choice(["0x111111", "0x222244", "0x111122"])
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s={w}x{h}:r={fps}",
        "-t", f"{dur:.2f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------- composition ----------

def _concat(clips: list[Path], out: Path) -> None:
    listfile = out.with_suffix(".txt")
    listfile.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in clips), encoding="utf-8"
    )
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(listfile),
        "-c", "copy",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Re-encode fallback
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    listfile.unlink(missing_ok=True)


def _final_mux(
    *,
    silent_video: Path,
    narration: Path,
    music: Path | None,
    srt: Path,
    out: Path,
    caption_style: dict,
) -> None:
    fontsize = caption_style.get("fontsize", 72)
    fontcolor = caption_style.get("fontcolor", "white")
    bordercolor = caption_style.get("bordercolor", "black")
    borderw = caption_style.get("borderw", 4)

    # Subtitle filter — escape path for ffmpeg
    sub_path = str(srt).replace(":", r"\:").replace("'", r"\'")
    has_subs = srt.exists() and srt.stat().st_size > 0

    inputs = ["-i", str(silent_video), "-i", str(narration)]
    filter_parts: list[str] = []
    if has_subs:
        filter_parts.append(
            f"[0:v]subtitles='{sub_path}':force_style="
            f"'FontName=DejaVu Sans,Bold=1,FontSize={fontsize//3},"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline={borderw},"
            f"Alignment=2,MarginV=120'[v]"
        )
    else:
        filter_parts.append("[0:v]copy[v]")

    if music and music.exists():
        inputs += ["-i", str(music)]
        filter_parts.append(
            "[2:a]volume=0.18,aloop=loop=-1:size=2e9[bg];"
            "[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        amap = "[a]"
    else:
        amap = "1:a"

    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", amap,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------- timing ----------

def _scene_durations(audio_summary: dict, num_scenes: int) -> list[float]:
    """Map audio segments → scene durations.

    Convention: audio segments are 'hook', 'scene_00'..'scene_NN', optional 'cta'.
    Hook gets folded into scene 0; CTA into the last scene.
    """
    by_name = {seg["name"]: seg.get("duration", 0.0) for seg in audio_summary.get("segments", [])}
    durs = []
    for i in range(num_scenes):
        d = by_name.get(f"scene_{i:02d}", 0.0)
        if i == 0:
            d += by_name.get("hook", 0.0)
        if i == num_scenes - 1:
            d += by_name.get("cta", 0.0)
        durs.append(max(d, 1.5))
    return durs
