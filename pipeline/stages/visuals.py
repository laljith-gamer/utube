"""Per-scene visuals — MOTION ONLY (no static image slideshow).

Pipeline per scene:
    1. SDXL still -> SVD animation         (primary, gives generated motion)
    2. Stock B-roll video                  (real-footage fallback)
    3. Synthesized moving gradient filler  (final fallback, marked in record)

The SDXL still is kept on disk only as SVD's input + for debugging. It is never
emitted as a "use this still as the scene" instruction to the assemble stage.

Config:
    visuals.skip_svd: bool   - if true, skip SDXL+SVD and go straight to stock+filler.
                               Used by `--skip-svd` for fast manual tests.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import get_config
from ..providers.image import ImageRouter
from ..providers.stock import StockRouter
from ..providers.video import VideoRouter

LOG = logging.getLogger("utube.visuals")


def generate_visuals(
    *,
    image: ImageRouter,
    video: VideoRouter,
    stock: StockRouter,
    script: dict,
    out_dir: Path,
) -> list[dict]:
    cfg = get_config()
    width = int(cfg.get_path("video.width", 1080))
    height = int(cfg.get_path("video.height", 1920))
    skip_svd = bool(cfg.get_path("visuals.skip_svd", False))

    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    scenes = script["scenes"]
    out: list[dict] = []
    for i, scene in enumerate(scenes):
        scene_dir = visuals_dir / f"scene_{i:02d}"
        scene_dir.mkdir(exist_ok=True)
        prompt = scene.get("visual_prompt", "")
        broll = scene.get("broll_keywords") or []
        record: dict = {"index": i, "prompt": prompt}

        # ----- 1. SDXL still -> SVD animation -----
        if not skip_svd:
            still_path = scene_dir / "still.png"
            still_ok = False
            try:
                png_bytes = image.generate(prompt, width=width, height=height)
                still_path.write_bytes(png_bytes)
                still_ok = True
                LOG.info("scene %d: still generated for SVD (%d bytes)", i, len(png_bytes))
            except Exception as e:  # noqa: BLE001
                LOG.warning("scene %d: SDXL image gen failed: %s", i, e)

            if still_ok:
                try:
                    mp4_bytes = video.animate(still_path.read_bytes())
                    mp4_path = scene_dir / "clip.mp4"
                    mp4_path.write_bytes(mp4_bytes)
                    record["video"] = str(mp4_path.relative_to(out_dir))
                    record["source"] = "svd"
                    LOG.info("scene %d: SVD motion clip ok (%d bytes)", i, len(mp4_bytes))
                except Exception as e:  # noqa: BLE001
                    LOG.warning("scene %d: SVD failed, will try stock video: %s", i, e)

        # ----- 2. Stock B-roll video fallback -----
        if "video" not in record:
            try:
                stock_bytes = stock.find_video(broll, orientation="portrait")
            except Exception as e:  # noqa: BLE001
                LOG.warning("scene %d: stock provider raised: %s", i, e)
                stock_bytes = None
            if stock_bytes:
                mp4_path = scene_dir / "stock.mp4"
                mp4_path.write_bytes(stock_bytes)
                record["video"] = str(mp4_path.relative_to(out_dir))
                record["source"] = "stock"
                LOG.info("scene %d: stock video fallback used", i)

        # ----- 3. Synthesized motion filler (no static image fallback) -----
        if "video" not in record:
            record["motion_fallback"] = True
            record["source"] = "filler"
            LOG.warning(
                "scene %d: no SVD or stock available; assemble will render motion filler",
                i,
            )

        out.append(record)

    return out
