"""Per-scene visuals with explicit provider/fallback provenance."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import get_config
from ..providers.image import ImageRouter
from ..providers.stock import StockRouter
from ..providers.video import VideoRouter

LOG = logging.getLogger("utube.visuals")


def generate_visuals(*, image: ImageRouter, video: VideoRouter, stock: StockRouter, script: dict, out_dir: Path) -> list[dict]:
    cfg = get_config()
    width = int(cfg.get_path("video.width", 1080))
    height = int(cfg.get_path("video.height", 1920))
    skip_svd = bool(cfg.get_path("visuals.skip_svd", False))
    min_relevance = float(cfg.get_path("visual_qc.min_relevance_score", 0.3))
    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    def generate_scene(i: int, scene: dict) -> dict:
        scene_dir = visuals_dir / f"scene_{i:02d}"
        scene_dir.mkdir(exist_ok=True)
        prompt = scene.get("visual_prompt", "")
        broll = scene.get("broll_keywords") or []
        record = {"index": i, "prompt": prompt, "attempts": []}

        # ── Priority 1: Stock B-roll (always motion, cheapest) ──
        if broll:
            try:
                stock_bytes = stock.find_video(broll, orientation="portrait")
                if stock_bytes:
                    path = scene_dir / "stock.mp4"
                    path.write_bytes(stock_bytes)
                    record.update({"video": str(path.relative_to(out_dir)), "source": "stock"})
                    record["attempts"].append({"type": "stock", "status": "ok", "relevance": None, "relevance_status": "unavailable"})
                    LOG.info("scene %d: stock B-roll used", i)
                    return record
                else:
                    record["attempts"].append({"type": "stock", "status": "not_found"})
            except Exception as exc:
                record["attempts"].append({"type": "stock", "status": "failed", "error": str(exc)})
                LOG.warning("scene %d: stock search failed: %s", i, exc)

        # ── Priority 2: Generated still (+ optional SVD animation) ──
        still_path = scene_dir / "still.png"
        still_ok = False
        try:
            png = image.generate(prompt, width=width, height=height)
            still_path.write_bytes(png)
            still_ok = True
            record["attempts"].append({"type": "image", "provider": "pollinations", "status": "ok"})
        except Exception as exc:
            record["attempts"].append({"type": "image", "provider": "pollinations", "status": "failed", "error": str(exc)})
            LOG.warning("scene %d: image generation failed: %s", i, exc)

        if still_ok and not skip_svd:
            try:
                clip = video.animate(still_path.read_bytes())
                path = scene_dir / "clip.mp4"
                path.write_bytes(clip)
                record.update({"video": str(path.relative_to(out_dir)), "source": "svd"})
                record["attempts"].append({"type": "svd", "status": "ok"})
            except Exception as exc:
                record["attempts"].append({"type": "svd", "status": "failed", "error": str(exc)})
                LOG.warning("scene %d: SVD failed: %s", i, exc)

        # If SVD produced a video, we're done
        if "video" in record:
            return record

        # Still image with Ken Burns motion treatment
        if still_ok:
            record.update({"image": str(still_path.relative_to(out_dir)), "source": "image_motion", "motion_treatment": "zoom_pan"})
            return record

        # ── Priority 3: Animated-gradient filler (last resort) ──
        record.update({"motion_fallback": True, "source": "filler"})
        record["attempts"].append({"type": "filler", "status": "used"})
        return record

    return [generate_scene(i, scene) for i, scene in enumerate(script["scenes"])]
