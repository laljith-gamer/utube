"""Final FFmpeg assembly.

Pipeline per video:
  1. Render each scene to a fixed-resolution clip (image+KenBurns OR raw video).
  2. Burn an optional per-scene text_overlay (big magazine-headline style)
     for scenes whose script-JSON included a text_overlay string. This is
     what makes the result feel hand-edited.
  3. Optionally append an outro card ("FOLLOW FOR MORE") generated from the
     last frame, dimmed.
  4. Concat all clips (hard cut OR crossfade).
  5. Mux narration + (optional) ducked music + burned captions.

All knobs from pipeline.yaml > assemble + > pacing + > overlay + > outro.
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
    pcfg = cfg.get_path("pacing", {}) or {}
    ocfg = cfg.get_path("overlay", {}) or {}
    outro_cfg = cfg.get_path("outro", {}) or {}
    width = int(vcfg.get("width", 1080))
    height = int(vcfg.get("height", 1920))
    fps = int(vcfg.get("fps", 30))

    work_dir = out_dir / "assemble"
    work_dir.mkdir(parents=True, exist_ok=True)

    seg_durations = _scene_durations(audio_summary, num_scenes=len(visuals))

    scene_clips: list[Path] = []
    for i, v in enumerate(visuals):
        dur = max(seg_durations[i], 1.5)
        base_clip = work_dir / f"scene_{i:02d}_base.mp4"
        if "video" in v:
            _render_from_video(out_dir / v["video"], base_clip, dur, width, height, fps, acfg)
        elif "image" in v:
            _render_from_image(out_dir / v["image"], base_clip, dur, width, height, fps, acfg, scene_idx=i)
        else:
            _render_solid(base_clip, dur, width, height, fps, acfg)

        # Apply per-scene text overlay if requested
        text_overlay = (v.get("text_overlay") or "").strip()
        if text_overlay and ocfg.get("enabled", True):
            overlaid = work_dir / f"scene_{i:02d}.mp4"
            try:
                _apply_text_overlay(base_clip, overlaid, text_overlay, dur, width, height, ocfg, acfg)
                base_clip.unlink(missing_ok=True)
                scene_clips.append(overlaid)
                continue
            except Exception as e:  # noqa: BLE001
                LOG.warning("scene %d: overlay failed (%s); keeping plain clip", i, e)
                # fall through to use base_clip
        scene_clips.append(base_clip)

    # Optional outro card built from the last scene's last frame
    if outro_cfg.get("enabled", True) and scene_clips:
        try:
            outro_path = work_dir / "outro.mp4"
            _build_outro_card(scene_clips[-1], outro_path, width, height, fps, outro_cfg, acfg)
            scene_clips.append(outro_path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Outro card failed (%s); skipping", e)

    silent_video = work_dir / "silent.mp4"
    transition = (pcfg.get("scene_transition") or "hard_cut").lower()
    if transition == "crossfade":
        _concat_crossfade(scene_clips, silent_video, fps,
                          float(pcfg.get("crossfade_duration_sec", 0.15)),
                          width, height, acfg)
    else:
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

def _render_from_image(img: Path, out: Path, dur: float, w: int, h: int, fps: int,
                       acfg: dict, *, scene_idx: int = 0) -> None:
    """Ken Burns: zoom + alternating-direction subtle pan to keep things alive."""
    kb = acfg.get("ken_burns", {}) or {}
    zoom_inc = float(kb.get("zoom_increment_per_frame", 0.0028))
    max_zoom = float(kb.get("max_zoom", 1.55))
    pan_x_amp = float(kb.get("pan_x_amplitude", 0.10))
    pan_y_amp = float(kb.get("pan_y_amplitude", 0.06))
    zoom_frames = max(int(dur * fps), 1)

    sign_x = 1 if scene_idx % 2 == 0 else -1
    sign_y = 1 if (scene_idx // 2) % 2 == 0 else -1
    x_expr = f"iw/2-(iw/zoom/2)+({sign_x}*{pan_x_amp}*iw*on/{zoom_frames})"
    y_expr = f"ih/2-(ih/zoom/2)+({sign_y}*{pan_y_amp}*ih*on/{zoom_frames})"

    vf = (
        f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
        f"crop={w*2}:{h*2},"
        f"zoompan=z='min(zoom+{zoom_inc},{max_zoom})':"
        f"x='{x_expr}':y='{y_expr}':"
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


# ---------- per-scene text overlay ----------

def _ffmpeg_escape(text: str) -> str:
    """Escape for FFmpeg drawtext text= argument."""
    return (
        text.replace("\\", r"\\")
            .replace(":", r"\:")
            .replace("'", r"\'")
            .replace(",", r"\,")
            .replace("[", r"\[")
            .replace("]", r"\]")
    )


def _apply_text_overlay(src: Path, dst: Path, text: str, dur: float,
                        w: int, h: int, ocfg: dict, acfg: dict) -> None:
    """Burn a big punchy headline over the clip with fade-in/out."""
    fontsize = int(ocfg.get("fontsize", 130))
    fontcolor = str(ocfg.get("fontcolor", "white"))
    bordercolor = str(ocfg.get("bordercolor", "black"))
    borderw = int(ocfg.get("borderw", 6))
    shadow_x = int(ocfg.get("shadow_x", 4))
    shadow_y = int(ocfg.get("shadow_y", 4))
    pos = str(ocfg.get("position", "top")).lower()
    margin_ratio = float(ocfg.get("margin_v_ratio", 0.18))
    fade_in = float(ocfg.get("fade_in_sec", 0.20))
    fade_out = float(ocfg.get("fade_out_sec", 0.25))

    # y position
    if pos == "top":
        y_expr = f"{int(h * margin_ratio)}"
    elif pos == "bottom":
        y_expr = f"h-text_h-{int(h * margin_ratio)}"
    else:  # center
        y_expr = "(h-text_h)/2"

    # Show overlay slightly clipped from start/end so it pops in the middle
    show_start = max(0.05, fade_in / 2)
    show_end = max(show_start + 0.5, dur - 0.05)
    enable = f"between(t,{show_start},{show_end})"
    alpha = (
        f"if(lt(t,{show_start + fade_in}),"
        f"(t-{show_start})/{fade_in},"
        f"if(gt(t,{show_end - fade_out}),"
        f"({show_end}-t)/{fade_out},1))"
    )

    safe_text = _ffmpeg_escape(text)
    drawtext = (
        f"drawtext=text='{safe_text}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize={fontsize}:fontcolor={fontcolor}:"
        f"borderw={borderw}:bordercolor={bordercolor}:"
        f"shadowx={shadow_x}:shadowy={shadow_y}:"
        f"x=(w-text_w)/2:y={y_expr}:"
        f"alpha='{alpha}':enable='{enable}'"
    )
    f = acfg.get("ffmpeg", {}) or {}
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", drawtext,
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-an", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------- outro card ----------

def _build_outro_card(last_clip: Path, dst: Path, w: int, h: int, fps: int,
                      ocfg: dict, acfg: dict) -> None:
    """Freeze the last frame, dim it, and overlay a follow-for-more headline."""
    duration = float(ocfg.get("duration_sec", 2.5))
    text_main = str(ocfg.get("text_main", "FOLLOW FOR MORE"))
    text_sub = str(ocfg.get("text_sub", ""))
    fontsize_main = int(ocfg.get("fontsize_main", 110))
    fontsize_sub = int(ocfg.get("fontsize_sub", 46))
    fontcolor = str(ocfg.get("fontcolor", "white"))
    dim_alpha = float(ocfg.get("background_dim_alpha", 0.55))
    borderw_main = int(ocfg.get("borderw_main", 7))
    borderw_sub = int(ocfg.get("borderw_sub", 3))

    # Extract last frame as PNG
    work = dst.parent
    last_frame = work / "outro_lastframe.png"
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(last_clip),
         "-frames:v", "1", "-update", "1", str(last_frame)],
        check=True, capture_output=True, text=True,
    )

    # Build dim + draw filters
    main_safe = _ffmpeg_escape(text_main)
    sub_safe = _ffmpeg_escape(text_sub) if text_sub else ""

    filters = [
        f"scale={w}:{h}:force_original_aspect_ratio=increase",
        f"crop={w}:{h}",
        f"eq=brightness=-{dim_alpha:.2f}",
        (
            f"drawtext=text='{main_safe}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize={fontsize_main}:fontcolor={fontcolor}:"
            f"borderw={borderw_main}:bordercolor=black:"
            f"x=(w-text_w)/2:y=(h-text_h)/2-40"
        ),
    ]
    if sub_safe:
        filters.append(
            f"drawtext=text='{sub_safe}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize={fontsize_sub}:fontcolor={fontcolor}:"
            f"borderw={borderw_sub}:bordercolor=black:"
            f"x=(w-text_w)/2:y=(h-text_h)/2+{fontsize_main}"
        )
    vf = ",".join(filters)

    f = acfg.get("ffmpeg", {}) or {}
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(last_frame),
        "-t", f"{duration:.2f}",
        "-vf", vf,
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-r", str(fps),
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    last_frame.unlink(missing_ok=True)


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


def _concat_crossfade(clips: list[Path], out: Path, fps: int, fade: float,
                      w: int, h: int, acfg: dict) -> None:
    if len(clips) < 2:
        _concat(clips, out, acfg); return
    f = acfg.get("ffmpeg", {}) or {}
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]
    durations = [_probe_duration_video(c) for c in clips]
    parts = []
    cur_label = "[0:v]"
    cur_dur = durations[0]
    for i in range(1, len(clips)):
        next_label = f"[v{i}]"
        offset = max(cur_dur - fade, 0.0)
        parts.append(f"{cur_label}[{i}:v]xfade=transition=fade:duration={fade}:offset={offset}{next_label}")
        cur_label = next_label
        cur_dur = cur_dur + durations[i] - fade
    filter_complex = ";".join(parts)
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", cur_label,
        "-c:v", f.get("video_codec", "libx264"),
        "-pix_fmt", f.get("pix_fmt", "yuv420p"),
        "-r", str(fps),
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        LOG.warning("crossfade failed (%s); falling back to hard cut",
                    res.stderr.splitlines()[-1] if res.stderr else "?")
        _concat(clips, out, acfg)


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
        style_kv = [
            ("FontName",       str(cs.get("fontname", "DejaVu Sans"))),
            ("Bold",           str(int(cs.get("bold", 1)))),
            ("FontSize",       str(int(cs.get("fontsize", 22)))),
            ("PrimaryColour",  f"&H{cs.get('primary_color_hex', '00FFFFFF')}"),
            ("OutlineColour",  f"&H{cs.get('outline_color_hex', '00000000')}"),
            ("BackColour",     f"&H{cs.get('back_color_hex', 'A0000000')}"),
            ("Outline",        str(int(cs.get("outline_width", 3)))),
            ("Shadow",         str(int(cs.get("shadow_depth", 1)))),
            ("Alignment",      str(int(cs.get("alignment", 2)))),
            ("MarginV",        str(int(cs.get("margin_v", 280)))),
            ("BorderStyle",    str(int(cs.get("border_style", 1)))),
        ]
        style = ",".join(f"{k}={v}" for k, v in style_kv)
        filter_parts.append(f"[0:v]subtitles='{sub_path}':force_style='{style}'[v]")
    else:
        filter_parts.append("[0:v]copy[v]")

    if music and music.exists():
        inputs += ["-i", str(music)]
        vol = float(music_cfg.get("volume", 0.20))
        loop_filter = "aloop=loop=-1:size=2e9" if music_cfg.get("loop", True) else "anull"
        filter_parts.append(
            f"[2:a]volume={vol},{loop_filter}[bg];"
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


# ---------- timing helpers ----------

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


def _probe_duration_video(path: Path) -> float:
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
        return 1.5
