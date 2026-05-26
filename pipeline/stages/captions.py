"""Whisper captions — model size, device, words-per-cue from pipeline.yaml > captions."""
from __future__ import annotations

import logging
from pathlib import Path

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
    chunk_size = int(cfg.get("words_per_chunk", 3))

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
        lines: list[str] = []
        idx = 1
        for seg in segments:
            words = list(seg.words or [])
            if not words:
                lines.append(_srt_block(idx, seg.start, seg.end, seg.text.strip()))
                idx += 1
                continue
            chunk: list = []
            for w in words:
                chunk.append(w)
                if len(chunk) >= chunk_size:
                    lines.append(_srt_block(idx, chunk[0].start, chunk[-1].end,
                                            " ".join(c.word for c in chunk).strip()))
                    idx += 1
                    chunk = []
            if chunk:
                lines.append(_srt_block(idx, chunk[0].start, chunk[-1].end,
                                        " ".join(c.word for c in chunk).strip()))
                idx += 1
        srt_path.write_text("\n".join(lines), encoding="utf-8")
        LOG.info("Captions: %d cues from %.1fs audio", idx - 1, info.duration)
        return srt_path
    except Exception as e:  # noqa: BLE001
        LOG.warning("Whisper transcription failed (%s); writing empty SRT", e)
        srt_path.write_text("", encoding="utf-8")
        return srt_path


def _srt_block(idx: int, start: float, end: float, text: str) -> str:
    return f"{idx}\n{_ts(start)} --> {_ts(end)}\n{text}\n"


def _ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
