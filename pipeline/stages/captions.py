"""Whisper captions — model size, device, words-per-cue from pipeline.yaml > captions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import get_config

LOG = logging.getLogger("utube.captions")


def transcribe_to_srt(audio_path: Path, srt_path: Path) -> Path:
    cfg = get_config().get_path("captions", {}) or {}
    if not cfg.get("enabled", True):
        srt_path.write_text("", encoding="utf-8")
        return srt_path

    model_size = cfg.get("whisper_model_size", "base")
    device = cfg.get("whisper_device", "cpu")
    compute_type = cfg.get("whisper_compute_type", "int8")
    beam = int(cfg.get("beam_size", 1))
    word_ts = bool(cfg.get("word_timestamps", True))
    vad = bool(cfg.get("vad_filter", True))
    fade_ms = int(cfg.get("fade_ms", 100))
    layout = _layout_cfg(cfg)

    try:
        from faster_whisper import WhisperModel

        LOG.info("Loading faster-whisper model %r on %s/%s …", model_size, device, compute_type)
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=beam,
            word_timestamps=word_ts,
            vad_filter=vad,
        )
        timed_cues: list[tuple[float, float, str]] = []
        for seg in segments:
            words = list(seg.words or [])
            if not words:
                timed_cues.extend(
                    _plain_text_cues(seg.start, seg.end, seg.text.strip(), layout)
                )
                continue
            timed_cues.extend(_word_cues(words, layout))

        lines = [_srt_block(i, start, end, text, fade_ms) for i, (start, end, text) in enumerate(timed_cues, 1)]
        srt_path.write_text("\n".join(lines), encoding="utf-8")
        LOG.info("Captions: %d cues from %.1fs audio", len(timed_cues), info.duration)

        # Free memory explicitly
        del model
        import gc
        gc.collect()

        return srt_path
    except Exception as e:  # noqa: BLE001
        LOG.warning("Whisper transcription failed (%s); writing empty SRT", e)
        srt_path.write_text("", encoding="utf-8")
        return srt_path


def _layout_cfg(cfg: dict) -> dict[str, int]:
    return {
        "chunk_size": int(cfg.get("words_per_chunk", 3)),
        "max_chars": int(cfg.get("max_chars_per_line", 24)),
        "max_lines": max(1, int(cfg.get("max_lines_per_cue", 2))),
    }


def _highlighted_cues(lines: list[tuple[list[Any], str]]) -> list[tuple[float, float, str]]:
    all_tokens = [tok for line_tokens, _ in lines for tok in line_tokens]
    if not all_tokens:
        return []
        
    cues = []
    for i, active_tok in enumerate(all_tokens):
        t_start = float(active_tok.start)
        t_end = float(all_tokens[i+1].start) if i + 1 < len(all_tokens) else float(all_tokens[-1].end)
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
                    part = f'<font color="#FFFF00">{part}</font>'
                line_str += part
            formatted_lines.append(line_str)
            
        cues.append((t_start, t_end, "\n".join(formatted_lines)))
    return cues


def _word_cues(words: list[Any], layout: dict[str, int]) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    chunk_size = layout["chunk_size"]
    
    current_chunk = []
    word_count = 0
    
    def _flush(chunk):
        wrapped = _wrap_tokens(chunk, layout["max_chars"])
        # _group_wrapped_lines groups lines into batches of max_lines
        for i in range(0, len(wrapped), layout["max_lines"]):
            batch = wrapped[i:i + layout["max_lines"]]
            cues.extend(_highlighted_cues(batch))

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
) -> list[tuple[float, float, str]]:
    words = text.split()
    if not words:
        return [(start, end, text.strip())]

    cue_texts: list[str] = []
    chunk_size = layout["chunk_size"]
    max_lines = layout["max_lines"]
    max_chars = layout["max_chars"]
    for i in range(0, len(words), chunk_size):
        wrapped = _wrap_strings(words[i:i + chunk_size], max_chars)
        for j in range(0, len(wrapped), max_lines):
            cue_texts.append("\n".join(wrapped[j:j + max_lines]))

    if not cue_texts:
        return [(start, end, text.strip())]

    duration = max(float(end) - float(start), 0.01)
    word_counts = [max(1, len(cue.replace("\n", " ").split())) for cue in cue_texts]
    total_words = sum(word_counts)
    t = float(start)
    cues: list[tuple[float, float, str]] = []
    for cue_text, count in zip(cue_texts, word_counts):
        cue_end = t + duration * count / total_words
        cues.append((t, cue_end, cue_text))
        t = cue_end
    cues[-1] = (cues[-1][0], float(end), cues[-1][2])
    return cues


def _wrap_tokens(tokens: list[Any], max_chars: int) -> list[tuple[list[Any], str]]:
    lines: list[tuple[list[Any], str]] = []
    current: list[Any] = []
    current_text = ""
    for tok in tokens:
        raw_word = str(getattr(tok, "word", tok))
        word_strip = raw_word.strip()
        if not word_strip:
            continue
        # Use the raw word (which usually has a leading space if needed)
        # unless it's the first word on the line, then we strip it.
        if not current_text:
            candidate = word_strip
        else:
            # If the raw word didn't have a leading space, don't add one (e.g., punctuation or split numbers)
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


def _group_wrapped_lines(
    wrapped: list[tuple[list[Any], str]],
    max_lines: int,
) -> list[tuple[list[Any], str]]:
    cues: list[tuple[list[Any], str]] = []
    for i in range(0, len(wrapped), max_lines):
        batch = wrapped[i:i + max_lines]
        tokens = [tok for batch_tokens, _ in batch for tok in batch_tokens]
        text = "\n".join(line_text for _, line_text in batch)
        cues.append((tokens, text))
    return cues


def _srt_block(idx: int, start: float, end: float, text: str, fade_ms: int = 0) -> str:
    if fade_ms > 0:
        text = f"{{\\fad({fade_ms},{fade_ms})}}{text}"
    return f"{idx}\n{_ts(start)} --> {_ts(end)}\n{text}\n"


def _ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
