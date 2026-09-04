"""Whisper captions — cinematic ASS subtitles with professional styling.

Output is ASS (Advanced SubStation Alpha) for full cinematic control:
  • PlayResX/Y match the video resolution → 1:1 pixel coordinates
  • Per-cue \\fad() for smooth fade in/out at chunk boundaries
  • Per-cue \\move() for subtle float-up entrance on new chunks
  • Word-by-word yellow highlight via ASS \\c override tags
  • Embedded Default style (font, outline, shadow, alignment, spacing)
  • Mid-chunk word swaps are instant — no fade, no flicker

The old SRT + force_style approach is replaced entirely.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import get_config

LOG = logging.getLogger("utube.captions")

# Each cue is a 5-tuple: (start, end, text, fade_in_ms, fade_out_ms)
CueT = tuple[float, float, str, int, int]


def transcribe_audio(audio_path: Path) -> tuple[str, list[Any], Any]:
    """Transcribe audio and return (full_text, segments_list, info)."""
    cfg = get_config()
    cap_cfg = cfg.get_path("captions", {}) or {}

    if not cap_cfg.get("enabled", True):
        return "", [], None

    model_size = cap_cfg.get("whisper_model_size", "base")
    device = cap_cfg.get("whisper_device", "cpu")
    compute_type = cap_cfg.get("whisper_compute_type", "int8")
    beam = int(cap_cfg.get("beam_size", 1))
    word_ts = bool(cap_cfg.get("word_timestamps", True))
    vad = bool(cap_cfg.get("vad_filter", True))

    try:
        from faster_whisper import WhisperModel

        LOG.info("Loading faster-whisper model %r on %s/%s …", model_size, device, compute_type)
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments_gen, info = model.transcribe(
            str(audio_path),
            beam_size=beam,
            word_timestamps=word_ts,
            vad_filter=vad,
        )
        
        segments = list(segments_gen)
        full_text = " ".join([s.text.strip() for s in segments])

        del model
        import gc
        gc.collect()

        return full_text, segments, info
    except Exception as e:  # noqa: BLE001
        LOG.warning("Whisper transcription failed (%s)", e)
        return "", [], None


def write_ass(segments: list[Any], info: Any, ass_path: Path) -> Path:
    """Generate cinematic ASS captions from ASR segments."""
    if not segments or info is None:
        ass_path.write_text("", encoding="utf-8")
        return ass_path

    cfg = get_config()
    cap_cfg = cfg.get_path("captions", {}) or {}
    fade_ms = int(cap_cfg.get("fade_ms", 150))
    layout = _layout_cfg(cap_cfg)

    vcfg = cfg.get_path("video", {}) or {}
    video_w = int(vcfg.get("width", 1080))
    video_h = int(vcfg.get("height", 1920))

    acfg = cfg.get_path("assemble", {}) or {}
    style_cfg = acfg.get("cinematic_caption_style", {}) or {}
    float_px = int(style_cfg.get("float_pixels", 6))

    timed_cues: list[CueT] = []
    for seg in segments:
        words = list(seg.words or [])
        if not words:
            timed_cues.extend(
                _plain_text_cues(seg.start, seg.end, seg.text.strip(), layout, fade_ms)
            )
            continue
        timed_cues.extend(_word_cues(words, layout, fade_ms))

    header = _ass_header(style_cfg, video_w, video_h)
    cx, cy = video_w // 2, video_h // 2
    dialogues = [
        _ass_dialogue(s, e, txt, fi, fo, float_px, cx, cy)
        for s, e, txt, fi, fo in timed_cues
    ]
    ass_path.write_text(header + "".join(dialogues), encoding="utf-8")
    LOG.info("Captions: %d cues from %.1fs audio → %s", len(timed_cues), info.duration, ass_path.name)
    return ass_path


def transcribe_to_srt(audio_path: Path, srt_path: Path) -> Path:
    """Legacy wrapper for callers that haven't been updated to split ASR/ASS."""
    full_text, segments, info = transcribe_audio(audio_path)
    ass_path = srt_path.with_suffix(".ass")
    return write_ass(segments, info, ass_path)


# ────────────────────────────────────────────────────────────────
#  ASS file structure
# ────────────────────────────────────────────────────────────────

def _ass_header(style_cfg: dict, w: int = 1080, h: int = 1920) -> str:
    """Generate ASS header with cinematic Default style.

    PlayResX/Y match the video so coordinates are 1:1 with pixels.
    ScaledBorderAndShadow ensures outline/shadow scale when the player
    up- or down-scales the video.
    """
    fn = style_cfg.get("fontname", "DejaVu Sans")
    # Default font size ≈ 3.8% of video height — large and readable on phones
    fs = int(style_cfg.get("fontsize", max(48, int(h * 0.038))))
    pc = style_cfg.get("primary_color", "&H00FFFFFF")       # White
    sc = style_cfg.get("secondary_color", "&H000000FF")     # Red (karaoke, unused)
    oc = style_cfg.get("outline_color", "&H00000000")       # Black outline
    bc = style_cfg.get("back_color", "&H80000000")          # Semi-transparent shadow
    bd = int(style_cfg.get("bold", -1))                     # -1 = true in ASS
    ol = float(style_cfg.get("outline", 3.5))
    sh = float(style_cfg.get("shadow", 1.5))
    sp = float(style_cfg.get("spacing", 1.5))               # Slight letter spacing
    al = int(style_cfg.get("alignment", 5))                  # 5 = center-center
    # Margins control text wrapping width: ~76% of video width
    ml = int(style_cfg.get("margin_l", int(w * 0.12)))
    mr = int(style_cfg.get("margin_r", int(w * 0.12)))
    mv = int(style_cfg.get("margin_v", 0))
    bs = int(style_cfg.get("border_style", 1))               # 1 = outline + drop shadow

    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\n"
        f"PlayResY: {h}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fn},{fs},{pc},{sc},{oc},{bc},{bd},0,0,0,"
        f"100,100,{sp},0,{bs},{ol},{sh},{al},{ml},{mr},{mv},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )


def _ass_dialogue(
    start: float,
    end: float,
    text: str,
    fade_in: int = 0,
    fade_out: int = 0,
    float_px: int = 6,
    cx: int = 540,
    cy: int = 960,
) -> str:
    """Build one ASS Dialogue line with cinematic animation.

    fade_in > 0  →  float-up + fade-in  (first word of chunk)
    fade_out > 0 →  fade-out, static    (last word of chunk)
    both == 0    →  instant display      (mid-chunk highlight swap)
    """
    parts: list[str] = []

    # Fade envelope
    if fade_in > 0 or fade_out > 0:
        parts.append(f"\\fad({fade_in},{fade_out})")

    # Position / float-up movement
    if fade_in > 0 and float_px > 0:
        # Glide from float_px below center to center over fade_in ms
        parts.append(
            f"\\move({cx},{cy + float_px},{cx},{cy},0,{fade_in})"
        )
    else:
        parts.append(f"\\pos({cx},{cy})")

    tags = "".join(parts)
    return f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},Default,,0,0,0,,{{{tags}}}{text}\n"


def _ass_ts(t: float) -> str:
    """Format seconds → ASS timestamp ``H:MM:SS.CC`` (centiseconds)."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ────────────────────────────────────────────────────────────────
#  Cue generation (Whisper word timestamps → timed cues)
# ────────────────────────────────────────────────────────────────

def _layout_cfg(cfg: dict) -> dict[str, int]:
    return {
        "chunk_size": int(cfg.get("words_per_chunk", 2)),
        "max_chars": int(cfg.get("max_chars_per_line", 30)),
        "max_lines": max(1, int(cfg.get("max_lines_per_cue", 2))),
    }


def _highlighted_cues(
    lines: list[tuple[list[Any], str]],
    fade_ms: int = 0,
) -> list[CueT]:
    """Build one cue per word, highlighting the active word in yellow.

    Uses ASS override tags ``\\c`` for color instead of HTML ``<font>`` tags.
    Fade is applied ONLY at chunk boundaries:
      • First word  → fade IN  (+ float-up handled by _ass_dialogue)
      • Last word   → fade OUT
      • Single-word → both
      • Middle      → instant highlight swap
    """
    all_tokens = [tok for line_tokens, _ in lines for tok in line_tokens]
    if not all_tokens:
        return []

    total = len(all_tokens)
    cues: list[CueT] = []
    for i, active_tok in enumerate(all_tokens):
        t_start = float(active_tok.start)
        t_end = float(all_tokens[i + 1].start) if i + 1 < total else float(all_tokens[-1].end)
        # Prevent 0-duration cues
        t_end = max(t_start + 0.01, t_end)

        formatted_lines = []

        for line_tokens, _ in lines:
            line_str = ""
            for tok in line_tokens:
                raw_word = str(getattr(tok, "word", tok))
                word_strip = raw_word.strip()

                if not line_str:
                    part = word_strip
                elif raw_word.startswith(" "):
                    part = " " + word_strip
                else:
                    part = word_strip

                if tok is active_tok:
                    # ASS override: bright yellow highlight, then reset to white
                    part = "{\\c&H00FFFF&}" + part + "{\\c&HFFFFFF&}"
                line_str += part
            formatted_lines.append(line_str)

        # Per-cue fade at chunk boundaries
        is_first = (i == 0)
        is_last = (i == total - 1)
        fade_in = fade_ms if is_first else 0
        fade_out = fade_ms if is_last else 0

        # ASS uses \\N for hard line breaks
        cues.append((t_start, t_end, "\\N".join(formatted_lines), fade_in, fade_out))
    return cues


def _word_cues(
    words: list[Any],
    layout: dict[str, int],
    fade_ms: int = 0,
) -> list[CueT]:
    """Group words into chunks, then build highlighted cues per chunk.

    fade_ms is passed through so each chunk gets fade-in on its first word
    and fade-out on its last word.  Mid-chunk word highlights are instant.
    """
    cues: list[CueT] = []
    chunk_size = layout["chunk_size"]

    current_chunk: list[Any] = []
    word_count = 0

    def _flush(chunk: list[Any]) -> None:
        wrapped = _wrap_tokens(chunk, layout["max_chars"])
        for i in range(0, len(wrapped), layout["max_lines"]):
            batch = wrapped[i:i + layout["max_lines"]]
            cues.extend(_highlighted_cues(batch, fade_ms))

    for w in words:
        raw = str(getattr(w, "word", w))
        is_new_word = raw.startswith(" ") or not current_chunk

        if is_new_word and word_count >= chunk_size:
            _flush(current_chunk)
            current_chunk = []
            word_count = 0

        current_chunk.append(w)
        if is_new_word:
            word_count += 1

    if current_chunk:
        _flush(current_chunk)

    return cues


def _plain_text_cues(
    start: float,
    end: float,
    text: str,
    layout: dict[str, int],
    fade_ms: int = 0,
) -> list[CueT]:
    """Fallback for segments without word-level timestamps.

    Each cue is standalone → full fade in and out.
    """
    words = text.split()
    if not words:
        return [(start, end, text.strip(), fade_ms, fade_ms)]

    cue_texts: list[str] = []
    chunk_size = layout["chunk_size"]
    max_lines = layout["max_lines"]
    max_chars = layout["max_chars"]
    for i in range(0, len(words), chunk_size):
        wrapped = _wrap_strings(words[i:i + chunk_size], max_chars)
        for j in range(0, len(wrapped), max_lines):
            cue_texts.append("\\N".join(wrapped[j:j + max_lines]))

    if not cue_texts:
        return [(start, end, text.strip(), fade_ms, fade_ms)]

    duration = max(float(end) - float(start), 0.01)
    word_counts = [max(1, len(cue.replace("\\N", " ").split())) for cue in cue_texts]
    total_words = sum(word_counts)
    t = float(start)
    cues: list[CueT] = []
    for cue_text, count in zip(cue_texts, word_counts):
        cue_end = t + duration * count / total_words
        cues.append((t, cue_end, cue_text, fade_ms, fade_ms))
        t = cue_end
    # Fix last cue end time
    cues[-1] = (cues[-1][0], float(end), cues[-1][2], cues[-1][3], cues[-1][4])
    return cues


# ────────────────────────────────────────────────────────────────
#  Text wrapping helpers
# ────────────────────────────────────────────────────────────────

def _wrap_tokens(tokens: list[Any], max_chars: int) -> list[tuple[list[Any], str]]:
    lines: list[tuple[list[Any], str]] = []
    current: list[Any] = []
    current_text = ""
    for tok in tokens:
        raw_word = str(getattr(tok, "word", tok))
        word_strip = raw_word.strip()
        if not word_strip:
            continue
        if not current_text:
            candidate = word_strip
        else:
            if raw_word.startswith(" "):
                candidate = current_text + " " + word_strip
            else:
                candidate = current_text + word_strip

        if not current_text or len(candidate) <= max_chars:
            current.append(tok)
            current_text = candidate
        else:
            lines.append((current, current_text))
            current = [tok]
            current_text = word_strip
    if current:
        lines.append((current, current_text))
    return lines


def _wrap_strings(words: list[str], max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if not current or len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
