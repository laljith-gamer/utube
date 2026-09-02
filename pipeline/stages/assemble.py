"""Final FFmpeg assembly: visuals + caption burn-in + music duck.

Per scene we render either:
  * a real video clip (SVD or stock), scaled+cropped to portrait, looped to fit duration; or
  * a static image with Ken Burns motion (zoom/pan); or
  * a synthesized animated gradient (motion filler) — used only when SVD, stock, and image both failed.

All ffmpeg knobs (codec, crf, preset, music volume, caption style, filler colors) live in
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

    import concurrent.futures

    fade_dur = float(acfg.get("transition_duration_sec", 0.3))
    
    def _render_scene_clip(i: int, v: dict) -> Path:
        # Extend the clip by the fade duration so the overlap doesn't desync audio
        added_dur = fade_dur if i < len(visuals) - 1 else 0.0
        dur = max(seg_durations[i], 1.5) + added_dur
        out_clip = work_dir / f"scene_{i:02d}.mp4"
        if "video" in v:
            _render_from_video(out_dir / v["video"], out_clip, dur, width, height, fps, acfg)
        elif "image" in v:
            _render_from_image(out_dir / v["image"], out_clip, dur, width, height, fps, acfg)
        else:
            _render_motion_filler(out_clip, dur, width, height, fps, acfg)
        return out_clip

    scene_clips: list[Path] = [Path()] * len(visuals)
    # ffmpeg already multithreads internally. Running parallel ffmpeg video scaling
    # jobs on a GitHub Actions runner (7GB RAM, 2 cores) alongside cached PyTorch models causes OOM.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(_render_scene_clip, i, v): i for i, v in enumerate(visuals)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            scene_clips[i] = future.result()

    silent_video = work_dir / "silent.mp4"
    _concat(scene_clips, silent_video, acfg, seg_durations, fade_dur)

    narration = out_dir / audio_summary["master"]
    _final_mux(
        silent_video=silent_video,
        narration=narration,
        music=music_path,
        srt=srt_path,
        out=output_path,
        cfg=acfg,
    )
    
    _technical_validation(output_path, srt_path, audio_summary)
    
    LOG.info("Final video -> %s", output_path)
    return output_path


# ---------- per-scene renderers ----------

def _render_from_video(src: Path, out: Path, dur: float, w: int, h: int, fps: int, acfg: dict) -> None:
    """Loop and crop any input clip to exactly `dur` seconds at the target portrait size."""
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


def _render_from_image(src: Path, out: Path, dur: float, w: int, h: int, fps: int, acfg: dict) -> None:
    """Render a static image with Ken Burns (zoom/pan) effect."""
    f = acfg.get("ffmpeg", {}) or {}
    image_cfg = acfg.get("image_motion", {}) or {}
    
    # We'll use zoompan filter with random cinematic motion.
    import random
    zoom_rate = float(image_cfg.get("zoom_rate", 0.0015))
    frames = int(dur * fps)
    motion_type = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"])

    if motion_type == "zoom_in":
        z = f"min(zoom+{zoom_rate},1.5)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif motion_type == "zoom_out":
        z = f"max(1.3-{zoom_rate}*on,1.0)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_left":
        z = "1.3"
        x = "max((iw-iw/zoom)-on*2,0)"
        y = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_right":
        z = "1.3"
        x = "min(on*2,iw-iw/zoom)"
        y = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_up":
        z = "1.3"
        x = "iw/2-(iw/zoom/2)"
        y = "max((ih-ih/zoom)-on*2,0)"
    else:  # pan_down
        z = "1.3"
        x = "iw/2-(iw/zoom/2)"
        y = "min(on*2,ih-ih/zoom)"

    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps}"
    )
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(src),
        "-t", f"{dur:.2f}",
        "-vf", vf, "-an",
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-r", str(fps),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _render_motion_filler(out: Path, dur: float, w: int, h: int, fps: int, acfg: dict) -> None:
    """Synthesize a moving gradient as a no-real-footage fallback."""
    f = acfg.get("ffmpeg", {}) or {}
    mf = acfg.get("motion_filler", {}) or {}
    colors = mf.get("colors", ["0x0a0a2a", "0x4f1eb1", "0x111122", "0x222244"])
    speed = float(mf.get("speed", 0.02))
    random.shuffle(colors)
    c0, c1, c2, c3 = (colors + colors)[:4]

    primary = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i",
        f"gradients=size={w}x{h}:rate={fps}:duration={dur:.2f}:"
        f"speed={speed}:c0={c0}:c1={c1}:c2={c2}:c3={c3}:type=linear",
        "-t", f"{dur:.2f}",
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-r", str(fps),
        str(out),
    ]
    res = subprocess.run(primary, capture_output=True, text=True)
    if res.returncode == 0:
        return

    LOG.warning("gradients filter unavailable, using hue-rotation fallback: %s",
                res.stderr.strip().splitlines()[-1] if res.stderr else "?")
    fallback = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={c0}:s={w}x{h}:r={fps}:d={dur:.2f}",
        "-vf", f"hue=h='t*40':s='0.6+0.4*sin(t)'",
        "-t", f"{dur:.2f}",
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-r", str(fps),
        str(out),
    ]
    subprocess.run(fallback, check=True, capture_output=True, text=True)


# ---------- composition ----------

def _concat(clips: list[Path], out: Path, acfg: dict, seg_durations: list[float], fade_dur: float) -> None:
    f = acfg.get("ffmpeg", {}) or {}
    
    if fade_dur <= 0.001 or len(clips) < 2:
        listfile = out.with_suffix(".txt")
        listfile.write_text("\n".join(f"file '{c.resolve()}'" for c in clips), encoding="utf-8")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c:v", f.get("video_codec", "libx264"),
            "-pix_fmt", f.get("pix_fmt", "yuv420p"),
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        listfile.unlink(missing_ok=True)
        return

    inputs = []
    for c in clips:
        inputs.extend(["-i", str(c)])

    filter_parts = []
    last_out = "[0:v]"
    current_offset = 0.0
    
    for i in range(1, len(clips)):
        current_offset += max(seg_durations[i-1], 1.5)
        out_pad = f"[v{i}]" if i < len(clips) - 1 else "[outv]"
        # Using a smooth crossfade transition
        filter_parts.append(f"{last_out}[{i}:v]xfade=transition=fade:duration={fade_dur}:offset={current_offset:.3f}{out_pad}")
        last_out = out_pad

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]",
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _final_mux(*, silent_video: Path, narration: Path, music: Path | None,
               srt: Path, out: Path, cfg: dict) -> None:
    f = cfg.get("ffmpeg", {}) or {}
    music_cfg = cfg.get("music", {}) or {}

    # Windows paths need forward slashes or escaped backslashes for FFmpeg filters
    sub_path = str(srt).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    has_subs = srt.exists() and srt.stat().st_size > 0

    inputs = ["-i", str(silent_video), "-i", str(narration)]
    filter_parts: list[str] = []
    if has_subs:
        # ASS file contains full cinematic styling — no force_style needed
        filter_parts.append(f"[0:v]subtitles='{sub_path}'[v]")
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
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        LOG.error("FFMPEG ERROR (final_mux): %s", e.stderr)
        raise


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


def _technical_validation(output_path: Path, srt_path: Path, audio_summary: dict) -> None:
    if not output_path.exists():
        raise RuntimeError("Output video file was not created.")
    
    if output_path.stat().st_size < 10_000:
        raise RuntimeError("Output video is unexpectedly small (less than 10KB).")
        
    if not srt_path.exists() or srt_path.stat().st_size == 0:
        LOG.warning("No captions file found or file is empty.")
    
    import json
    
    # Run ffprobe to check streams
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height",
        "-of", "json",
        str(output_path)
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        probe = json.loads(res.stdout)
    except Exception as e:
        LOG.error("ffprobe failed during technical validation: %s", e)
        raise RuntimeError(f"FFprobe validation failed: {e}")

    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise RuntimeError("Validation failed: No video stream found in final output.")
    if not audio_streams:
        raise RuntimeError("Validation failed: No audio stream found in final output.")

    # Check codec (we expect h264/aac from the pipeline config)
    v_codec = video_streams[0].get("codec_name", "")
    if v_codec not in ("h264", "libx264"):
        LOG.warning("Unexpected video codec: %s (expected h264)", v_codec)

    a_codec = audio_streams[0].get("codec_name", "")
    if a_codec not in ("aac", "mp3"):
        LOG.warning("Unexpected audio codec: %s", a_codec)

    # Check duration bounds
    try:
        duration = float(probe.get("format", {}).get("duration", 0))
        if duration < 5.0:
            raise RuntimeError(f"Validation failed: Final video is too short ({duration:.1f}s)")
        if duration > 120.0:
            LOG.warning("Final video is longer than 60s Shorts limit: %.1fs", duration)
    except (ValueError, TypeError):
        LOG.warning("Could not parse duration from ffprobe output.")

    LOG.info("Technical validation passed for %s (%.1fs, %s/%s)", 
             output_path.name, duration if 'duration' in locals() else 0, v_codec, a_codec)
