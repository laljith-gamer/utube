"""Generate the thumbnail: SDXL background + Pillow text overlay."""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..providers.image import ImageRouter

LOG = logging.getLogger("utube.thumbnail")


def make_thumbnail(
    *,
    image: ImageRouter,
    prompt: str,
    text: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    palette: list[str] | None = None,
) -> Path:
    palette = palette or ["#FFFFFF", "#000000"]
    try:
        bg_bytes = image.generate(prompt, width=width, height=height)
        with open(out_path.with_suffix(".bg.png"), "wb") as f:
            f.write(bg_bytes)
        bg = Image.open(out_path.with_suffix(".bg.png")).convert("RGB").resize((width, height))
    except Exception as e:  # noqa: BLE001
        LOG.warning("Thumbnail SDXL failed (%s), using solid background", e)
        bg = Image.new("RGB", (width, height), palette[1])

    # Darken bottom-left for text legibility
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, height // 3), (width, height)], fill=(0, 0, 0, 140))
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    bg = bg.filter(ImageFilter.SMOOTH)

    # Big punchy text
    draw = ImageDraw.Draw(bg)
    font = _load_font(int(height * 0.16))
    words = text.upper().split()
    # Wrap to ~2 lines max
    line1, line2 = (" ".join(words[: len(words) // 2 or 1]),
                    " ".join(words[len(words) // 2 or 1 :]))
    lines = [l for l in (line1, line2) if l]

    y = int(height * 0.55)
    for line in lines:
        # Stroke for legibility
        for ox in (-3, 0, 3):
            for oy in (-3, 0, 3):
                draw.text((40 + ox, y + oy), line, font=font, fill="black")
        draw.text((40, y), line, font=font, fill=palette[0])
        y += int(height * 0.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out_path, "JPEG", quality=88)
    out_path.with_suffix(".bg.png").unlink(missing_ok=True)
    LOG.info("Thumbnail saved → %s", out_path.name)
    return out_path


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
