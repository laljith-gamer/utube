"""Per-scene visuals.

Routing (controlled by config/pipeline.yaml > visuals):

  prefer_stock_video=true (default)
    1. Try stock B-roll by [key_subject, ...broll_keywords]
    2. If no stock match: SDXL still
    3. If `enable_svd` AND scene flagged use_motion: animate the still with SVD
    4. If both fail: solid colour fallback (handled in assemble.py)

  prefer_stock_video=false
    1. SDXL still
    2. If enable_svd AND use_motion: SVD animate
    3. Stock B-roll fallback only if both fail

The LLM provides `key_subject` (literal, searchable noun phrase) which gives
us a concrete query for stock services and grounds the visual to what the
narrator is actually saying — instead of an abstract "the AI revolution".
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
    style_suffix = (slot.get("style_suffix") or "").strip()

    vis_cfg = cfg.get_path("visuals", {}) or {}
    prefer_stock = bool(vis_cfg.get("prefer_stock_video", True))
    enable_svd = bool(vis_cfg.get("enable_svd", False))
    motion_default = int(vis_cfg.get("motion_scenes_default", 5))

    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    scenes = script["scenes"]
    # Determine which scenes get motion: explicit use_motion=true wins;
    # otherwise the first N scenes are marked.
    explicit_motion = [i for i, s in enumerate(scenes) if s.get("use_motion")]
    if explicit_motion:
        motion_idxs = set(explicit_motion)
    else:
        motion_idxs = set(range(min(len(scenes), motion_default)))

    out: list[dict] = []
    for i, scene in enumerate(scenes):
        scene_dir = visuals_dir / f"scene_{i:02d}"
        scene_dir.mkdir(exist_ok=True)

        key_subject = (scene.get("key_subject") or "").strip()
        broll = [k for k in (scene.get("broll_keywords") or []) if k]
        base_prompt = (scene.get("visual_prompt") or "").strip()
        full_prompt = f"{base_prompt}, {style_suffix}" if style_suffix and base_prompt \
            else (base_prompt or style_suffix)

        record: dict = {
            "index": i,
            "key_subject": key_subject,
            "prompt": full_prompt,
            "use_motion": i in motion_idxs,
            "text_overlay": (scene.get("text_overlay") or "").strip(),
        }
        produced_video = False
        produced_image = False

        # ---- Stock-first path ----
        if prefer_stock and (key_subject or broll):
            queries = [q for q in [key_subject, *broll] if q]
            try:
                stock_bytes = stock.find_video(queries, orientation="portrait")
            except Exception as e:  # noqa: BLE001
                LOG.warning("scene %d: stock search errored: %s", i, e)
                stock_bytes = None
            if stock_bytes:
                mp4_path = scene_dir / "stock.mp4"
                mp4_path.write_bytes(stock_bytes)
                record["video"] = str(mp4_path.relative_to(out_dir))
                LOG.info("scene %d: stock B-roll matched key_subject=%r", i, key_subject)
                produced_video = True

        # ---- SDXL still (always try if no stock match yet) ----
        png_path = scene_dir / "image.png"
        if not produced_video:
            try:
                png_bytes = image.generate(full_prompt, width=width, height=height)
                png_path.write_bytes(png_bytes)
                record["image"] = str(png_path.relative_to(out_dir))
                LOG.info("scene %d: SDXL image generated (%d bytes)", i, len(png_bytes))
                produced_image = True
            except Exception as e:  # noqa: BLE001
                LOG.warning("scene %d: image gen failed: %s", i, e)

        # ---- SVD animation on the still (optional) ----
        if not produced_video and produced_image and enable_svd and i in motion_idxs:
            try:
                mp4_bytes = video.animate(png_path.read_bytes())
                mp4_path = scene_dir / "clip.mp4"
                mp4_path.write_bytes(mp4_bytes)
                record["video"] = str(mp4_path.relative_to(out_dir))
                # Drop the image record — the animated clip supersedes it
                record.pop("image", None)
                LOG.info("scene %d: SVD animated (%d bytes)", i, len(mp4_bytes))
                produced_video = True
            except Exception as e:  # noqa: BLE001
                LOG.warning("scene %d: SVD failed, falling back to Ken Burns on still: %s", i, e)

        # ---- Last resort: stock B-roll if everything above failed ----
        if not produced_video and not produced_image:
            queries = [q for q in [key_subject, *broll] if q]
            if queries:
                try:
                    stock_bytes = stock.find_video(queries, orientation="portrait")
                    if stock_bytes:
                        mp4_path = scene_dir / "stock.mp4"
                        mp4_path.write_bytes(stock_bytes)
                        record["video"] = str(mp4_path.relative_to(out_dir))
                        LOG.info("scene %d: stock B-roll fallback used", i)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("scene %d: stock fallback errored: %s", i, e)

        if "image" not in record and "video" not in record:
            LOG.error("scene %d: no visual could be obtained!", i)
            record["error"] = "no visual"

        out.append(record)

    n_video = sum(1 for r in out if "video" in r)
    n_image = sum(1 for r in out if "image" in r and "video" not in r)
    LOG.info("Visuals summary: %d/%d real video clips, %d/%d still+kenburns",
             n_video, len(out), n_image, len(out))
    return out
