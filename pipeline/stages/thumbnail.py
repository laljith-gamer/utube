"""Thumbnail: generated portrait background + PIL text overlay.

Layout is config-driven from pipeline.yaml > thumbnail. Defaults stay safe when
older configs are used.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

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
    width = int(cfg.get("width", 1080))
    height = int(cfg.get("height", 1920))
    quality = int(cfg.get("jpeg_quality", 92))
    palette = palette or ["#FFFFFF", "#000000", "#FFDD00"]

    render_prompt = _thumbnail_prompt(prompt, cfg)

    try:
        bg_bytes = image.generate(render_prompt, width=width, height=height)
        tmp_bg = out_path.with_suffix(".bg.png")
        tmp_bg.write_bytes(bg_bytes)
        bg = Image.open(tmp_bg).convert("RGB")
        bg = ImageOps.fit(bg, (width, height), method=_resample_lanczos())
        tmp_bg.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        LOG.warning("Thumbnail image generation failed (%s), using gradient background", e)
        bg = _fallback_gradient(width, height, palette)

    bg = _enhance_background(bg, cfg)
    bg = _apply_portrait_overlays(bg, cfg, palette)
    bg = _draw_thumbnail_text(bg, text, cfg, palette)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out_path, "JPEG", quality=quality, optimize=True)
    LOG.info("Portrait thumbnail saved → %s (%sx%s)", out_path.name, width, height)
    return out_path


def _thumbnail_prompt(prompt: str, cfg: dict) -> str:
    suffix = str(cfg.get("prompt_suffix", "")).strip()
    base = prompt.strip()
    if suffix and suffix.lower() not in base.lower():
        return f"{base}, {suffix}" if base else suffix
    return base


def _enhance_background(bg: Image.Image, cfg: dict) -> Image.Image:
    bg = ImageEnhance.Contrast(bg).enhance(float(cfg.get("enhance_contrast", 1.12)))
    bg = ImageEnhance.Color(bg).enhance(float(cfg.get("enhance_color", 1.08)))
    bg = ImageEnhance.Sharpness(bg).enhance(float(cfg.get("enhance_sharpness", 1.15)))
    return bg.filter(ImageFilter.SMOOTH)


def _apply_portrait_overlays(bg: Image.Image, cfg: dict, palette: list[str]) -> Image.Image:
    width, height = bg.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    bottom_alpha = int(cfg.get("darken_overlay_alpha", 165))
    top_alpha = int(cfg.get("top_overlay_alpha", 60))
    vignette_alpha = int(cfg.get("vignette_alpha", 110))
    accent_h = max(0, int(height * float(cfg.get("accent_bar_height_ratio", 0.012))))

    od.rectangle([(0, int(height * 0.42)), (width, height)], fill=(0, 0, 0, bottom_alpha))
    od.rectangle([(0, 0), (width, int(height * 0.18))], fill=(0, 0, 0, top_alpha))

    if accent_h > 0:
        accent = palette[2] if len(palette) > 2 else palette[0]
        od.rectangle([(0, height - accent_h), (width, height)], fill=accent)

    vignette = Image.new("L", (width, height), 0)
    vd = ImageDraw.Draw(vignette)
    inset_x = int(width * 0.08)
    inset_y = int(height * 0.05)
    vd.ellipse((-inset_x, -inset_y, width + inset_x, height + inset_y), fill=255)
    vignette = ImageOps.invert(vignette.filter(ImageFilter.GaussianBlur(int(width * 0.08))))
    vignette_layer = Image.new("RGBA", (width, height), (0, 0, 0, vignette_alpha))
    overlay = Image.alpha_composite(
        overlay,
        Image.composite(
            Image.new("RGBA", (width, height), (0, 0, 0, 0)),
            vignette_layer,
            vignette,
        ),
    )

    return Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")


def _draw_thumbnail_text(
    bg: Image.Image,
    text: str,
    cfg: dict,
    palette: list[str],
) -> Image.Image:
    width, height = bg.size
    draw = ImageDraw.Draw(bg)
    title = " ".join((text or "WATCH THIS").upper().split())
    max_width = int(width * float(cfg.get("text_max_width_ratio", 0.86)))
    max_lines = max(1, int(cfg.get("text_max_lines", 3)))

    start_size = int(height * float(cfg.get("text_height_ratio", 0.135)))
    min_size = int(height * float(cfg.get("text_min_height_ratio", 0.07)))
    font_paths = cfg.get("font_paths", []) or []

    font = _load_font(start_size, font_paths)
    lines = _wrap_text(draw, title, font, max_width)
    size = start_size
    while (len(lines) > max_lines or _widest_line(draw, lines, font) > max_width) and size > min_size:
        size = max(min_size, size - 4)
        font = _load_font(size, font_paths)
        lines = _wrap_text(draw, title, font, max_width)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _ellipsize(draw, lines[-1], font, max_width)

    stroke = int(cfg.get("stroke_width", max(3, int(height * 0.002))))
    line_gap = int(height * float(cfg.get("text_line_spacing_ratio", 0.08)))
    shadow_offset = max(1, int(height * float(cfg.get("text_shadow_offset_ratio", 0.006))))
    text_fill = palette[0] if palette else "#FFFFFF"
    accent_fill = palette[2] if len(palette) > 2 else text_fill

    line_sizes = [_text_size(draw, line, font) for line in lines]
    block_w = max(w for w, _ in line_sizes)
    block_h = sum(h for _, h in line_sizes) + line_gap * max(0, len(lines) - 1)
    center_y = int(height * float(cfg.get("text_y_ratio", 0.66)))
    y = max(int(height * 0.08), min(height - block_h - int(height * 0.04), center_y - block_h // 2))

    align = str(cfg.get("text_align", "center")).lower()
    if align == "left":
        x_base = int(cfg.get("margin_left", width * 0.07))
    else:
        x_base = (width - block_w) // 2

    panel_alpha = int(cfg.get("text_panel_alpha", 145))
    if panel_alpha > 0:
        padding = int(height * float(cfg.get("text_panel_padding_ratio", 0.028)))
        radius = int(height * float(cfg.get("text_panel_radius_ratio", 0.028)))
        panel = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        panel_box = [
            max(0, x_base - padding),
            max(0, y - padding),
            min(width, x_base + block_w + padding),
            min(height, y + block_h + padding),
        ]
        pd.rounded_rectangle(panel_box, radius=radius, fill=(0, 0, 0, panel_alpha))
        bg = Image.alpha_composite(bg.convert("RGBA"), panel).convert("RGB")
        draw = ImageDraw.Draw(bg)

    for index, line in enumerate(lines):
        line_w, line_h = _text_size(draw, line, font)
        if align == "left":
            x = x_base
        else:
            x = (width - line_w) // 2

        fill = accent_fill if index == 0 and len(lines) > 1 else text_fill
        draw.text(
            (x + shadow_offset, y + shadow_offset),
            line,
            font=font,
            fill="black",
            stroke_width=stroke,
            stroke_fill="black",
        )
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=stroke,
            stroke_fill="black",
        )
        y += line_h + line_gap

    badge = str(cfg.get("badge_text", "")).strip().upper()
    if badge:
        bg = _draw_badge(bg, badge, cfg, palette)

    return bg


def _draw_badge(bg: Image.Image, badge: str, cfg: dict, palette: list[str]) -> Image.Image:
    width, height = bg.size
    draw = ImageDraw.Draw(bg)
    font = _load_font(int(height * float(cfg.get("badge_height_ratio", 0.042))), cfg.get("font_paths", []) or [])
    pad_x = int(width * 0.035)
    pad_y = int(height * 0.012)
    x = int(width * 0.06)
    y = int(height * 0.045)
    badge_w, badge_h = _text_size(draw, badge, font)
    box = [x, y, x + badge_w + pad_x * 2, y + badge_h + pad_y * 2]
    accent = palette[2] if len(palette) > 2 else "#FFDD00"
    draw.rounded_rectangle(box, radius=int(height * 0.018), fill=accent)
    draw.text((x + pad_x, y + pad_y), badge, font=font, fill="black")
    return bg


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return ["WATCH THIS"]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or _text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _ellipsize(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    ellipsis = "..."
    trimmed = text
    while trimmed and _text_size(draw, trimmed + ellipsis, font)[0] > max_width:
        trimmed = trimmed[:-1].rstrip()
    return f"{trimmed}{ellipsis}" if trimmed else ellipsis


def _widest_line(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont) -> int:
    return max((_text_size(draw, line, font)[0] for line in lines), default=0)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:
        return draw.textsize(text, font=font)


def _fallback_gradient(width: int, height: int, palette: list[str]) -> Image.Image:
    top = Image.new("RGB", (width, 1), palette[1] if len(palette) > 1 else "#111111")
    bottom = Image.new("RGB", (width, 1), "#1A1A1A")
    bg = Image.new("RGB", (width, height))
    for y in range(height):
        blend = y / max(1, height - 1)
        row = Image.blend(top, bottom, blend)
        bg.paste(row, (0, y))
    return bg


def _load_font(size: int, paths: list[str]) -> ImageFont.ImageFont:
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resample_lanczos() -> int:
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS
