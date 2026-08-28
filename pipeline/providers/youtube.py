"""YouTube uploader — all knobs (chunk size, scopes, category) read from config."""
from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from ..config import get_config
from ..utils import env, env_bool

LOG = logging.getLogger("utube.youtube")


def _yt_cfg() -> dict:
    return get_config().get_path("youtube", {}) or {}


def _credentials() -> Credentials:
    cfg = _yt_cfg()
    client_id = env(cfg.get("client_id_env", "YOUTUBE_CLIENT_ID"))
    client_secret = env(cfg.get("client_secret_env", "YOUTUBE_CLIENT_SECRET"))
    refresh_token = env(cfg.get("refresh_token_env", "YOUTUBE_REFRESH_TOKEN"))
    if not (client_id and client_secret and refresh_token):
        raise RuntimeError(
            "YouTube OAuth not configured. Set the env vars referenced by youtube.* in providers.yaml."
        )
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=cfg.get("token_uri", "https://oauth2.googleapis.com/token"),
        scopes=cfg.get("scopes", ["https://www.googleapis.com/auth/youtube.upload"]),
    )
    creds.refresh(Request())
    return creds


def upload_video(
    video_path: Path,
    *,
    title: str,
    description: str,
    tags: list[str],
    publish_at_iso: str | None = None,
    thumbnail_path: Path | None = None,
    privacy_status: str = "private",
) -> dict:
    cfg = _yt_cfg()

    if env_bool("DRY_RUN"):
        return {
            "id": "dry-run",
            "video_id": "dry-run",
            "url": "https://youtu.be/dry-run",
            "title": title,
            "dry_run": True,
        }

    creds = _credentials()
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)

    title_max = int(cfg.get("title_max_chars", 100))
    desc_max = int(cfg.get("description_max_chars", 5000))
    tags_max = int(cfg.get("tags_max", 30))

    body: dict = {
        "snippet": {
            "title": title[:title_max],
            "description": description[:desc_max],
            "tags": tags[:tags_max],
            "categoryId": str(cfg.get("category_id", "28")),
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": bool(cfg.get("default_made_for_kids", False)),
        },
    }
    if publish_at_iso:
        body["status"]["publishAt"] = publish_at_iso
        body["status"]["privacyStatus"] = "private"

    chunk_size = int(cfg.get("upload_chunk_size_mb", 8)) * 1024 * 1024
    media = MediaFileUpload(str(video_path), chunksize=chunk_size, resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    LOG.info("Uploading %s …", video_path.name)
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            LOG.info("  upload %d%%", int(status.progress() * 100))

    video_id = response["id"]
    LOG.info("Upload complete: https://youtu.be/%s", video_id)

    if thumbnail_path and thumbnail_path.exists():
        try:
            yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(thumbnail_path))).execute()
            LOG.info("Thumbnail set for %s", video_id)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Thumbnail upload failed: %s", e)

    return {
        "id": video_id,
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}",
        "title": title,
        "dry_run": False,
    }
