"""Thumbnail: SDXL background + PIL text overlay.

Layout entirely from pipeline.yaml > thumbnail. The renderer:
  1. Generates an SDXL background sized to (width, height) — 9:16 by default.
  2. Darkens the lower 2/3 with a translucent overlay for legibility.
  3. Wraps the title text using REAL pixel measurement (PIL textbbox), so a
     long word never overflows the canvas. If even the longest word can't
     fit at the configured font size, the size is auto-shrunk until it does.
  4. Caps the result at thumbnail.max_lines lines (default 4).
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import get_config
from ..providers.image import ImageRouter

LOG = logging.getLogger("utube.thumbnail")


def make_thumbnail(
    *,
    image: ImageRouter,
    prompt: str,
    text: str,
    out_path: Path,
    palette: list[str] | None = None,
) -> Path:
    cfg = get_config().get_path("thumbnail", {}) or {}
    width = int(cfg.get("width", 720))
    height = int(cfg.get("height", 1280))
    quality = int(cfg.get("jpeg_quality", 90))
    text_height_ratio = float(cfg.get("text_height_ratio", 0.085))
    text_y_ratio = float(cfg.get("text_y_ratio", 0.50))
    line_spacing_ratio = float(cfg.get("text_line_spacing_ratio", 0.13))
    max_text_width_ratio = float(cfg.get("max_text_width_ratio", 0.86))
    max_lines = int(cfg.get("max_lines", 4))
    margin_left = int(cfg.get("margin_left", 50))
    stroke = int(cfg.get("stroke_width", 5))
    overlay_alpha = int(cfg.get("darken_overlay_alpha", 170))
    palette = palette or ["#FFFFFF", "#000000"]

    # ---- background ----
    try:
        bg_bytes = image.generate(prompt, width=width, height=height)
        tmp_bg = out_path.with_suffix(".bg.png")
        tmp_bg.write_bytes(bg_bytes)
        bg = Image.open(tmp_bg).convert("RGB").resize((width, height))
        tmp_bg.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        LOG.warning("Thumbnail SDXL failed (%s), using solid background", e)
        bg = Image.new("RGB", (width, height), palette[1])

    # Darken the lower portion behind the text
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, height // 3), (width, height)], fill=(0, 0, 0, overlay_alpha))
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    bg = bg.filter(ImageFilter.SMOOTH)

    # ---- text ----
    text = text.upper().strip()
    font_paths = cfg.get("font_paths", []) or []

    # The usable text canvas: inside max_text_width_ratio.
    max_text_w = int(width * max_text_width_ratio)
    initial_size = max(20, int(height * text_height_ratio))

    font, lines = _fit_font_and_wrap(
        text, font_paths, initial_size, max_text_w, max_lines,
    )
    LOG.info("Thumbnail text: %d lines at fontsize %d", len(lines), getattr(font, "size", 0))

    draw = ImageDraw.Draw(bg)
    line_h = int(height * line_spacing_ratio)
    if line_h <= 0:
        line_h = getattr(font, "size", 32) + 8
    block_h = line_h * len(lines)

    # Vertically center the block around text_y_ratio
    y = int(height * text_y_ratio) - block_h // 2

    for line in lines:
        line_w = _text_width(line, font)
        # Horizontally center each line, but never less than margin_left
        x = max(margin_left, (width - line_w) // 2)

        # Stroke / outline
        for ox in (-stroke, 0, stroke):
            for oy in (-stroke, 0, stroke):
                if ox == 0 and oy == 0:
                    continue
                draw.text((x + ox, y + oy), line, font=font, fill="black")
        # Fill
        draw.text((x, y), line, font=font, fill=palette[0])
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out_path, "JPEG", quality=quality)
    LOG.info("Thumbnail saved → %s", out_path.name)
    return out_path


# ---------- text fitting ----------

def _fit_font_and_wrap(
    text: str,
    font_paths: list[str],
    initial_size: int,
    max_width: int,
    max_lines: int,
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Find the largest font size at which the text fits in max_width / max_lines.

    Strategy:
      - Start at initial_size, then shrink in 10% steps down to 18.
      - At each size, greedy-wrap words. If any individual word is wider
        than max_width at this size, shrink and try again.
      - If wrap produces > max_lines, shrink and try again.
      - On exhaustion, return the smallest with whatever wrapping survives.
    """
    size = max(20, int(initial_size))
    last_font = None
    last_lines: list[str] = []
    while size >= 18:
        font = _load_font(size, font_paths)
        words = text.split()
        # Reject if a single word is wider than the canvas at this size
        if any(_text_width(w, font) > max_width for w in words):
            last_font, last_lines = font, words[:max_lines]
            size = max(18, int(size * 0.9))
            if size == int(initial_size):
                break  # safety
            continue
        wrapped = _greedy_wrap(words, font, max_width)
        if len(wrapped) <= max_lines:
            return font, wrapped
        last_font, last_lines = font, wrapped[:max_lines]
        size = max(18, int(size * 0.9))
        if last_font is font and size >= initial_size:
            break  # converged
    return last_font or _load_font(18, font_paths), last_lines or [text]


def _greedy_wrap(words: list[str], font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word wrap. Each line is filled until adding the next word
    would exceed max_width, then a new line starts."""
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        candidate = " ".join(current + [w]) if current else w
        if _text_width(candidate, font) <= max_width:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def _text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Measure rendered text width in pixels. Works across Pillow versions."""
    try:
        img = Image.new("RGB", (1, 1))
        d = ImageDraw.Draw(img)
        bbox = d.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except Exception:
        try:
            return int(font.getlength(text))
        except Exception:
            try:
                return font.getsize(text)[0]
            except Exception:
                return len(text) * (getattr(font, "size", 16) // 2)


def _load_font(size: int, paths: list[str]) -> ImageFont.FreeTypeFont:
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
