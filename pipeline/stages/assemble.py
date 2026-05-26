"""Final FFmpeg assembly: per-scene clips + Ken Burns + caption burn-in + music duck.
All ffmpeg knobs (codec, crf, preset, Ken Burns, music volume, caption style) live in
pipeline.yaml > assemble.
"""
from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

from ..config import get_config

LOG = logging.getLogger("utube.assemble")


def assemble_video(
    *,
    visuals: list[dict],
    audio_summary: dict,
    srt_path: Path,
    out_dir: Path,
    output_path: Path,
    music_path: Path | None = None,
) -> Path:
    cfg = get_config()
    vcfg = cfg.get_path("video", {}) or {}
    acfg = cfg.get_path("assemble", {}) or {}
    width = int(vcfg.get("width", 1080))
    height = int(vcfg.get("height", 1920))
    fps = int(vcfg.get("fps", 30))

    work_dir = out_dir / "assemble"
    work_dir.mkdir(parents=True, exist_ok=True)

    seg_durations = _scene_durations(audio_summary, num_scenes=len(visuals))

    scene_clips: list[Path] = []
    for i, v in enumerate(visuals):
        dur = max(seg_durations[i], 1.5)
        out_clip = work_dir / f"scene_{i:02d}.mp4"
        if "video" in v:
            _render_from_video(out_dir / v["video"], out_clip, dur, width, height, fps, acfg)
        elif "image" in v:
            _render_from_image(out_dir / v["image"], out_clip, dur, width, height, fps, acfg)
        else:
            _render_solid(out_clip, dur, width, height, fps, acfg)
        scene_clips.append(out_clip)

    silent_video = work_dir / "silent.mp4"
    _concat(scene_clips, silent_video, acfg)

    narration = out_dir / audio_summary["master"]
    _final_mux(
        silent_video=silent_video,
        narration=narration,
        music=music_path,
        srt=srt_path,
        out=output_path,
        cfg=acfg,
    )
    LOG.info("Final video → %s", output_path)
    return output_path


# ---------- per-scene renderers ----------

def _render_from_image(img: Path, out: Path, dur: float, w: int, h: int, fps: int, acfg: dict) -> None:
    kb = acfg.get("ken_burns", {}) or {}
    zoom_inc = float(kb.get("zoom_increment_per_frame", 0.0015))
    max_zoom = float(kb.get("max_zoom", 1.4))
    zoom_frames = int(dur * fps)
    vf = (
        f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
        f"crop={w*2}:{h*2},"
        f"zoompan=z='min(zoom+{zoom_inc},{max_zoom})':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={zoom_frames}:s={w}x{h}:fps={fps},"
        f"setsar=1"
    )
    f = acfg.get("ffmpeg", {}) or {}
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(img),
        "-t", f"{dur:.2f}",
        "-vf", vf,
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-r", str(fps),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _render_from_video(src: Path, out: Path, dur: float, w: int, h: int, fps: int, acfg: dict) -> None:
    f = acfg.get("ffmpeg", {}) or {}
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1,fps={fps}"
    )
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(src),
        "-t", f"{dur:.2f}",
        "-vf", vf, "-an",
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-r", str(fps),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _render_solid(out: Path, dur: float, w: int, h: int, fps: int, acfg: dict) -> None:
    f = acfg.get("ffmpeg", {}) or {}
    color = random.choice(["0x111111", "0x222244", "0x111122"])
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s={w}x{h}:r={fps}",
        "-t", f"{dur:.2f}",
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------- composition ----------

def _concat(clips: list[Path], out: Path, acfg: dict) -> None:
    listfile = out.with_suffix(".txt")
    listfile.write_text("\n".join(f"file '{c.resolve()}'" for c in clips), encoding="utf-8")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(listfile),
        "-c", "copy",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        f = acfg.get("ffmpeg", {}) or {}
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c:v", f.get("video_codec", "libx264"),
            "-pix_fmt", f.get("pix_fmt", "yuv420p"),
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    listfile.unlink(missing_ok=True)


def _final_mux(*, silent_video: Path, narration: Path, music: Path | None,
               srt: Path, out: Path, cfg: dict) -> None:
    cs = cfg.get("caption_subtitle_style", {}) or {}
    f = cfg.get("ffmpeg", {}) or {}
    music_cfg = cfg.get("music", {}) or {}

    sub_path = str(srt).replace(":", r"\:").replace("'", r"\'")
    has_subs = srt.exists() and srt.stat().st_size > 0

    inputs = ["-i", str(silent_video), "-i", str(narration)]
    filter_parts: list[str] = []
    if has_subs:
        style = (
            f"FontName={cs.get('fontname', 'DejaVu Sans')},"
            f"Bold={int(cs.get('bold', 1))},"
            f"FontSize={int(cs.get('fontsize_divisor', 3) and 24)},"
            f"PrimaryColour=&H{cs.get('primary_color_hex', '00FFFFFF')},"
            f"OutlineColour=&H{cs.get('outline_color_hex', '00000000')},"
            f"Outline={int(cs.get('outline_width', 4))},"
            f"Alignment={int(cs.get('alignment', 2))},"
            f"MarginV={int(cs.get('margin_v', 120))}"
        )
        filter_parts.append(f"[0:v]subtitles='{sub_path}':force_style='{style}'[v]")
    else:
        filter_parts.append("[0:v]copy[v]")

    if music and music.exists():
        inputs += ["-i", str(music)]
        vol = float(music_cfg.get("volume", 0.18))
        loop = "aloop=loop=-1:size=2e9" if music_cfg.get("loop", True) else "anull"
        filter_parts.append(
            f"[2:a]volume={vol},{loop}[bg];"
            f"[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        amap = "[a]"
    else:
        amap = "1:a"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[v]", "-map", amap,
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-crf", str(int(f.get("crf", 20))),
        "-preset", f.get("preset", "medium"),
        "-c:a", f.get("audio_codec", "aac"),
        "-b:a", f.get("audio_bitrate", "128k"),
        "-shortest",
        "-movflags", f.get("movflags", "+faststart"),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _scene_durations(audio_summary: dict, num_scenes: int) -> list[float]:
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
