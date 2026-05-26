"""Per-scene visuals.

For each scene:
  1. Augment the LLM's visual_prompt with the niche's `style_suffix`
     (cinematic/dramatic style words from config/niches.yaml).
  2. Generate SDXL still.
  3. If the scene is flagged for SVD (or in the first N scenes), animate it.
  4. If both fail, fall back to stock B-roll using broll_keywords.
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
    slot: dict,
    out_dir: Path,
) -> list[dict]:
    cfg = get_config()
    width = int(cfg.get_path("video.width", 1080))
    height = int(cfg.get_path("video.height", 1920))
    use_svd_for_n_scenes = int(cfg.get_path("video.use_svd_for_n_scenes", 4))
    style_suffix = (slot.get("style_suffix") or "").strip()

    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    scenes = script["scenes"]
    svd_scene_idxs = [i for i, s in enumerate(scenes) if s.get("use_motion")]
    if not svd_scene_idxs:
        svd_scene_idxs = list(range(min(len(scenes), use_svd_for_n_scenes)))
    svd_scene_idxs = svd_scene_idxs[:use_svd_for_n_scenes]

    out: list[dict] = []
    for i, scene in enumerate(scenes):
        scene_dir = visuals_dir / f"scene_{i:02d}"
        scene_dir.mkdir(exist_ok=True)
        base_prompt = (scene.get("visual_prompt") or "").strip()
        # Append cinematic style suffix only if it's not already implied
        prompt = f"{base_prompt}, {style_suffix}" if style_suffix and base_prompt else base_prompt or style_suffix
        broll = scene.get("broll_keywords") or []
        record: dict = {"index": i, "prompt": prompt}

        png_path = scene_dir / "image.png"
        try:
            png_bytes = image.generate(prompt, width=width, height=height)
            png_path.write_bytes(png_bytes)
            record["image"] = str(png_path.relative_to(out_dir))
            LOG.info("scene %d: image generated (%d bytes)", i, len(png_bytes))
        except Exception as e:  # noqa: BLE001
            LOG.warning("scene %d: image gen failed: %s", i, e)
            png_path = None

        if i in svd_scene_idxs and png_path is not None:
            try:
                mp4_bytes = video.animate(png_path.read_bytes())
                mp4_path = scene_dir / "clip.mp4"
                mp4_path.write_bytes(mp4_bytes)
                record["video"] = str(mp4_path.relative_to(out_dir))
                LOG.info("scene %d: SVD clip generated (%d bytes)", i, len(mp4_bytes))
            except Exception as e:  # noqa: BLE001
                LOG.warning("scene %d: SVD failed, will use Ken Burns on still: %s", i, e)

        if "image" not in record and "video" not in record:
            stock_bytes = stock.find_video(broll, orientation="portrait")
            if stock_bytes:
                mp4_path = scene_dir / "stock.mp4"
                mp4_path.write_bytes(stock_bytes)
                record["video"] = str(mp4_path.relative_to(out_dir))
                LOG.info("scene %d: stock B-roll fallback used", i)
            else:
                LOG.error("scene %d: no visual could be obtained!", i)
                record["error"] = "no visual"

        out.append(record)

    return out
