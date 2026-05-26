"""YouTube Data API v3 uploader using OAuth2 refresh tokens."""
from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from ..utils import env, env_bool

LOG = logging.getLogger("utube.youtube")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _credentials() -> Credentials:
    client_id = env("YOUTUBE_CLIENT_ID")
    client_secret = env("YOUTUBE_CLIENT_SECRET")
    refresh_token = env("YOUTUBE_REFRESH_TOKEN")
    if not (client_id and client_secret and refresh_token):
        raise RuntimeError(
            "YouTube OAuth not configured. "
            "Set YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN."
        )
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=SCOPES,
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
    category_id: str = "28",  # Science & Technology
    made_for_kids: bool = False,
    privacy_status: str = "private",
) -> dict:
    if env_bool("DRY_RUN"):
        LOG.info("[DRY_RUN] Would upload %s as %r (publish_at=%s)", video_path, title, publish_at_iso)
        return {"id": "dry-run", "url": "https://youtu.be/dry-run", "dry_run": True}

    creds = _credentials()
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body: dict = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:30],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }
    if publish_at_iso:
        body["status"]["publishAt"] = publish_at_iso
        body["status"]["privacyStatus"] = "private"  # required for scheduled publishing

    media = MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
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

    return {"id": video_id, "url": f"https://youtu.be/{video_id}", "dry_run": False}
