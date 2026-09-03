"""Per-scene visuals with explicit provider/fallback provenance."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import get_config
from ..providers.image import ImageRouter
from ..providers.stock import StockRouter
from ..providers.video import VideoRouter
from ..providers.llm import LLMRouter

LOG = logging.getLogger("utube.visuals")


def generate_visuals(*, image: ImageRouter, video: VideoRouter, stock: StockRouter, llm_vision: LLMRouter | None = None, script: dict, out_dir: Path) -> list[dict]:
    cfg = get_config()
    width = int(cfg.get_path("video.width", 1080))
    height = int(cfg.get_path("video.height", 1920))
    skip_svd = bool(cfg.get_path("visuals.skip_svd", False))
    min_relevance = float(cfg.get_path("visual_qc.min_relevance_score", 0.3))
    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    used_image_urls = set()

    def generate_scene(i: int, scene: dict) -> dict:
        scene_dir = visuals_dir / f"scene_{i:02d}"
        scene_dir.mkdir(exist_ok=True)
        prompt = scene.get("visual_prompt", "")
        broll = scene.get("broll_keywords") or []
        record = {"index": i, "prompt": prompt, "attempts": []}

        # ── Priority 1: Stock B-roll (Background) ──
        has_stock = False
        if broll:
            try:
                stock_bytes = stock.find_video(broll, orientation="portrait")
                if stock_bytes:
                    path = scene_dir / "stock.mp4"
                    path.write_bytes(stock_bytes)
                    record.update({"video": str(path.relative_to(out_dir))})
                    has_stock = True
                    record["attempts"].append({"type": "stock", "status": "ok", "relevance": None, "relevance_status": "unavailable"})
                    LOG.info("scene %d: stock B-roll found", i)
                else:
                    record["attempts"].append({"type": "stock", "status": "not_found"})
            except Exception as exc:
                record["attempts"].append({"type": "stock", "status": "failed", "error": str(exc)})
                LOG.warning("scene %d: stock search failed: %s", i, exc)

        # ── Priority 1.5: Brave Search Images (Evidence Foreground) ──
        has_brave = False
        if llm_vision and (broll or prompt):
            try:
                from ..providers.brave import BraveProvider
                from ..providers.llm import ProviderStatus
                import requests
                import base64
                
                search_q = " ".join(broll) if broll else prompt
                search_q = BraveProvider.spellcheck(search_q) or search_q
                
                img_cands = BraveProvider.search_images(search_q, count=5)
                real_img_path = scene_dir / "real.jpg"
                
                for cand in img_cands:
                    img_url = cand.get("url")
                    if not img_url or img_url in used_image_urls:
                        continue
                    
                    try:
                        resp = requests.get(img_url, timeout=10)
                        resp.raise_for_status()
                        mime = resp.headers.get("content-type", "image/jpeg")
                        b64 = base64.b64encode(resp.content).decode("utf-8")
                        data_uri = f"data:{mime};base64,{b64}"
                        
                        sys_prompt = "You are a visual investigator. Given the image and a scene description, rate the image relevance from 0 to 100. Return JSON: {'relevance': int, 'reason': 'str'}."
                        user_prompt = f"Scene Description: {prompt}\nKeywords: {broll}"
                        
                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": data_uri}}
                            ]}
                        ]
                        
                        res = llm_vision.chat_json_structured(messages, max_tokens=150)
                        if res.status == ProviderStatus.SUCCESS and res.parsed:
                            score = res.parsed.get("relevance", 0)
                            if score > 70:
                                real_img_path.write_bytes(resp.content)
                                has_brave = True
                                record.update({"image": str(real_img_path.relative_to(out_dir)), "evidence_url": img_url})
                                record["attempts"].append({"type": "brave_image", "status": "ok", "relevance": score, "url": img_url})
                                LOG.info("scene %d: brave image validated with score %s", i, score)
                                used_image_urls.add(img_url)
                                break
                            else:
                                record["attempts"].append({"type": "brave_image", "status": "rejected", "relevance": score})
                    except Exception as e:
                        LOG.debug("scene %d: brave image check failed: %s", i, e)
                        
            except Exception as exc:
                record["attempts"].append({"type": "brave_image", "status": "failed", "error": str(exc)})
                LOG.warning("scene %d: brave image search failed: %s", i, exc)
                
        # Determine composition based on found assets
        if has_stock and has_brave:
            record.update({"composition": "pip_evidence", "source": "stock_and_brave"})
            return record
        elif has_stock:
            record.update({"composition": "fullscreen", "source": "stock"})
            return record
        elif has_brave:
            record.update({"composition": "fullscreen", "source": "brave_images", "motion_treatment": "zoom_pan"})
            return record

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
