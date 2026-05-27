"""Whisper captions.

Two output formats — picked by config/pipeline.yaml > captions.format:
  - "ass": full TikTok / Submagic-style karaoke captions:
       * line-level fade in / fade out  (\\fad)
       * per-word colour highlight       (\\K karaoke)
       * per-word size pulse             (\\t time animation on \\fscx\\fscy)
       * positioning + colours from assemble.caption_subtitle_style
  - "srt": classic plain SRT (no animation, no colour highlight) — kept as
    a fallback if the user disables ASS or libass is unavailable.

The function name `transcribe_to_srt` is preserved (writes either format
based on config) so older call-sites don't break.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from ..config import get_config

LOG = logging.getLogger("utube.captions")


# ---------- public entry point ----------

def transcribe_to_srt(audio_path: Path, out_path: Path) -> Path:
    """Transcribe audio to a subtitle file. Despite the legacy name, this
    writes either an SRT or an ASS file based on captions.format."""
    cfg = get_config()
    ccfg = cfg.get_path("captions", {}) or {}
    if not ccfg.get("enabled", True):
        out_path.write_text("", encoding="utf-8")
        return out_path

    fmt = (ccfg.get("format") or "srt").lower()

    # If format=ass and the caller passed a .srt path, redirect to .ass
    if fmt == "ass" and out_path.suffix.lower() != ".ass":
        out_path = out_path.with_suffix(".ass")
    elif fmt == "srt" and out_path.suffix.lower() != ".srt":
        out_path = out_path.with_suffix(".srt")

    try:
        words = _whisper_words(audio_path, ccfg)
    except Exception as e:  # noqa: BLE001
        LOG.warning("Whisper transcription failed (%s); writing empty subs", e)
        out_path.write_text("", encoding="utf-8")
        return out_path

    chunks = _chunk_words(words, int(ccfg.get("words_per_chunk", 3)))

    if fmt == "ass":
        try:
            video_cfg = cfg.get_path("video", {}) or {}
            style_cfg = cfg.get_path("assemble.caption_subtitle_style", {}) or {}
            content = _render_ass(
                chunks=chunks,
                ccfg=ccfg,
                style_cfg=style_cfg,
                play_x=int(video_cfg.get("width", 1080)),
                play_y=int(video_cfg.get("height", 1920)),
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("ASS render failed (%s); falling back to SRT", e)
            out_path = out_path.with_suffix(".srt")
            content = _render_srt(chunks)
    else:
        content = _render_srt(chunks)

    out_path.write_text(content, encoding="utf-8")
    LOG.info("Captions: %d cues -> %s", len(chunks), out_path.name)
    return out_path


# ---------- whisper -> word timestamps ----------

def _whisper_words(audio_path: Path, ccfg: dict) -> list[dict]:
    """Return a flat list of {start, end, text} word entries."""
    from faster_whisper import WhisperModel

    model_size = ccfg.get("whisper_model_size", "base")
    device = ccfg.get("whisper_device", "cpu")
    compute_type = ccfg.get("whisper_compute_type", "int8")
    beam = int(ccfg.get("beam_size", 1))
    word_ts = bool(ccfg.get("word_timestamps", True))
    vad = bool(ccfg.get("vad_filter", True))

    LOG.info("Loading faster-whisper %r on %s/%s ...", model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam,
        word_timestamps=word_ts,
        vad_filter=vad,
    )
    words: list[dict] = []
    for seg in segments:
        seg_words = list(seg.words or [])
        if seg_words:
            for w in seg_words:
                t = (w.word or "").strip()
                if not t:
                    continue
                words.append({"start": float(w.start), "end": float(w.end), "text": t})
        else:
            # No word timestamps — synthesize them by splitting the segment text evenly
            tokens = (seg.text or "").split()
            if not tokens:
                continue
            seg_dur = max(0.001, seg.end - seg.start)
            per = seg_dur / len(tokens)
            for i, tok in enumerate(tokens):
                ws = seg.start + i * per
                we = ws + per
                words.append({"start": ws, "end": we, "text": tok.strip()})
    LOG.info("Captions: %d words from %.1fs audio", len(words), info.duration)
    return words


def _chunk_words(words: list[dict], n: int) -> list[list[dict]]:
    n = max(1, n)
    return [words[i:i + n] for i in range(0, len(words), n)]


# ---------- ASS rendering ----------

def _render_ass(*, chunks: list[list[dict]], ccfg: dict, style_cfg: dict,
                play_x: int, play_y: int) -> str:
    highlight = bool(ccfg.get("highlight_active_word", True))
    scale_pct = int(ccfg.get("active_word_scale_pct", 115))

    fontname = str(style_cfg.get("fontname", "DejaVu Sans"))
    bold = int(style_cfg.get("bold", 1))
    fontsize = int(style_cfg.get("fontsize", 32))
    primary = str(style_cfg.get("primary_color_hex", "00FFFFFF"))
    highlight_col = str(style_cfg.get("highlight_color_hex", "0000FFFF"))
    outline = str(style_cfg.get("outline_color_hex", "00000000"))
    back = str(style_cfg.get("back_color_hex", "A0000000"))
    outline_w = int(style_cfg.get("outline_width", 4))
    shadow = int(style_cfg.get("shadow_depth", 1))
    alignment = int(style_cfg.get("alignment", 5))
    margin_v = int(style_cfg.get("margin_v", 0))
    margin_h = int(style_cfg.get("margin_h", 60))
    border_style = int(style_cfg.get("border_style", 1))
    wrap_style = int(style_cfg.get("wrap_style", 0))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_x}\n"
        f"PlayResY: {play_y}\n"
        f"WrapStyle: {wrap_style}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fontname},{fontsize},"
        f"&H{primary},&H{highlight_col},&H{outline},&H{back},"
        f"{bold},0,0,0,100,100,0,0,{border_style},{outline_w},{shadow},"
        f"{alignment},{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    events: list[str] = []
    for words in chunks:
        if not words:
            continue
        events.extend(_build_chunk_events(
            words=words,
            ccfg=ccfg,
            primary=primary,
            highlight=highlight_col,
            scale_pct=scale_pct,
            highlight_on=highlight,
        ))

    return header + "".join(events)


def _build_chunk_events(*, words: list[dict], ccfg: dict, primary: str,
                       highlight: str, scale_pct: int,
                       highlight_on: bool) -> list[str]:
    """Emit one Dialogue line PER WORD.

    Each event spans only that word's spoken window. The full chunk text is
    rendered every time, but exactly one word is the highlight colour
    (and optionally scaled up) — every other word stays primary (white).
    Back-to-back events keep the chunk visually continuous.

    Line-level fade-in is applied only to the FIRST event of the chunk,
    fade-out only to the LAST event.
    """
    fade_in_ms = int(ccfg.get("fade_in_ms", 150))
    fade_out_ms = int(ccfg.get("fade_out_ms", 250))

    chunk_start = words[0]["start"]
    chunk_end = words[-1]["end"] + 0.05

    out: list[str] = []
    for i, w in enumerate(words):
        evt_start = w["start"] if i > 0 else chunk_start
        evt_end = words[i + 1]["start"] if i + 1 < len(words) else chunk_end
        if evt_end <= evt_start:
            evt_end = evt_start + 0.1

        # Build the chunk text with word i highlighted, others primary.
        parts: list[str] = []
        for j, ww in enumerate(words):
            overrides: list[str] = []
            # Line-level fade only on first event (j==0 of FIRST word event)
            # and last event (j==0 of LAST word event).
            if j == 0:
                if i == 0 and i == len(words) - 1:
                    overrides.append(f"\\fad({fade_in_ms},{fade_out_ms})")
                elif i == 0:
                    overrides.append(f"\\fad({fade_in_ms},0)")
                elif i == len(words) - 1:
                    overrides.append(f"\\fad(0,{fade_out_ms})")
                # else middle event: no fade

            if j == i and highlight_on:
                overrides.append(f"\\1c&H{highlight}&")
                if scale_pct != 100:
                    overrides.append(f"\\fscx{scale_pct}\\fscy{scale_pct}")
            else:
                overrides.append(f"\\1c&H{primary}&")
                overrides.append("\\fscx100\\fscy100")

            prefix = "{" + "".join(overrides) + "}" if overrides else ""
            parts.append(prefix + _ass_escape_text(ww["text"]))

        text = " ".join(parts)
        out.append(
            f"Dialogue: 0,{_ass_ts(evt_start)},{_ass_ts(evt_end)},Default,,0,0,0,,{text}\n"
        )
    return out


def _ass_escape_text(text: str) -> str:
    """Escape characters that have meaning inside an ASS Dialogue text field."""
    return (
        text.replace("\\", "\\\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\n", r"\N")
    )


def _ass_ts(t: float) -> str:
    """ASS timestamp: H:MM:SS.cs (centiseconds, single-digit hours)."""
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    cs = int(round((s - int(s)) * 100))
    si = int(s)
    if cs >= 100:
        cs -= 100
        si += 1
    return f"{h}:{m:02d}:{si:02d}.{cs:02d}"


# ---------- SRT rendering (legacy fallback) ----------

def _render_srt(chunks: list[list[dict]]) -> str:
    lines: list[str] = []
    for i, words in enumerate(chunks, start=1):
        if not words:
            continue
        start = words[0]["start"]
        end = words[-1]["end"]
        if end <= start:
            end = start + 0.5
        text = " ".join(w["text"].strip() for w in words)
        lines.append(_srt_block(i, start, end, text))
    return "\n".join(lines)


def _srt_block(idx: int, start: float, end: float, text: str) -> str:
    return f"{idx}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}\n"


def _srt_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
