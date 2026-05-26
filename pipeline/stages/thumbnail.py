"""Thumbnail: SDXL background + PIL text overlay. Layout entirely from pipeline.yaml > thumbnail."""
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
    width = int(cfg.get("width", 1280))
    height = int(cfg.get("height", 720))
    quality = int(cfg.get("jpeg_quality", 88))
    text_height_ratio = float(cfg.get("text_height_ratio", 0.16))
    text_y_ratio = float(cfg.get("text_y_ratio", 0.55))
    line_spacing_ratio = float(cfg.get("text_line_spacing_ratio", 0.18))
    margin_left = int(cfg.get("margin_left", 40))
    stroke = int(cfg.get("stroke_width", 3))
    overlay_alpha = int(cfg.get("darken_overlay_alpha", 140))
    palette = palette or ["#FFFFFF", "#000000"]

    try:
        bg_bytes = image.generate(prompt, width=width, height=height)
        tmp_bg = out_path.with_suffix(".bg.png")
        tmp_bg.write_bytes(bg_bytes)
        bg = Image.open(tmp_bg).convert("RGB").resize((width, height))
        tmp_bg.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        LOG.warning("Thumbnail SDXL failed (%s), using solid background", e)
        bg = Image.new("RGB", (width, height), palette[1])

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, height // 3), (width, height)], fill=(0, 0, 0, overlay_alpha))
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    bg = bg.filter(ImageFilter.SMOOTH)

    draw = ImageDraw.Draw(bg)
    font = _load_font(int(height * text_height_ratio), cfg.get("font_paths", []) or [])
    words = text.upper().split()
    half = max(1, len(words) // 2)
    line1, line2 = " ".join(words[:half]), " ".join(words[half:])
    lines = [l for l in (line1, line2) if l]

    y = int(height * text_y_ratio)
    for line in lines:
        for ox in (-stroke, 0, stroke):
            for oy in (-stroke, 0, stroke):
                draw.text((margin_left + ox, y + oy), line, font=font, fill="black")
        draw.text((margin_left, y), line, font=font, fill=palette[0])
        y += int(height * line_spacing_ratio)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out_path, "JPEG", quality=quality)
    LOG.info("Thumbnail saved → %s", out_path.name)
    return out_path


def _load_font(size: int, paths: list[str]) -> ImageFont.FreeTypeFont:
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
