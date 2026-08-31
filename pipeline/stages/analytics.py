"""YouTube performance collection and normalization."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

from ..utils import env, repo_root

LOG = logging.getLogger("utube.analytics")


def _youtube_data_service(api_key: str | None = None):
    if api_key:
        return None
    from ..config import get_config
    cfg = get_config().get_path("youtube", {}) or {}
    client_id = env(cfg.get("client_id_env", "YOUTUBE_CLIENT_ID"))
    client_secret = env(cfg.get("client_secret_env", "YOUTUBE_CLIENT_SECRET"))
    refresh_token = env(cfg.get("refresh_token_env", "YOUTUBE_REFRESH_TOKEN"))
    if not (client_id and client_secret and refresh_token):
        return None
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(token=None, refresh_token=refresh_token, token_uri=cfg.get("token_uri", "https://oauth2.googleapis.com/token"), client_id=client_id, client_secret=client_secret)
    return build("youtube", "v3", credentials=creds)


def collect_performance_data(published_videos: list[dict], api_key: str | None = None) -> list[dict]:
    key = api_key or env("YOUTUBE_API_KEY")
    video_ids = [v.get("video_id") for v in published_videos if v.get("video_id")]
    if not video_ids:
        return published_videos
    stats_map: dict[str, dict] = {}
    service = _youtube_data_service(key)
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            if key:
                r = requests.get("https://www.googleapis.com/youtube/v3/videos", params={"part": "statistics,snippet", "id": ",".join(batch), "key": key}, timeout=20)
                r.raise_for_status()
                items = r.json().get("items", [])
            elif service:
                items = service.videos().list(part="statistics,snippet", id=",".join(batch)).execute().get("items", [])
            else:
                LOG.warning("No YouTube Data API credentials available; preserving existing metrics")
                break
            for item in items:
                s = item.get("statistics", {})
                stats_map[item.get("id")] = {
                    "views": int(s.get("viewCount", 0)),
                    "likes": int(s.get("likeCount", 0)),
                    "comments": int(s.get("commentCount", 0)),
                    "published_at": item.get("snippet", {}).get("publishedAt"),
                }
        except Exception as exc:
            LOG.error("YouTube Data API fetch failed: %s", exc)
    out = []
    now = datetime.now(timezone.utc)
    for original in published_videos:
        vid = original.get("video_id")
        enriched = dict(original)
        s = stats_map.get(vid)
        if not s:
            enriched.setdefault("metrics_status", "unavailable")
            out.append(enriched)
            continue
        enriched.update({"views": s["views"], "likes": s["likes"], "comments": s["comments"], "published_at": s["published_at"], "metrics_status": "ok"})
        age_days = 1.0
        if s.get("published_at"):
            try:
                dt = datetime.fromisoformat(s["published_at"].replace("Z", "+00:00"))
                age_days = max(0.1, (now - dt).total_seconds() / 86400.0)
            except ValueError:
                pass
        enriched["age_days"] = round(age_days, 3)
        enriched["views_per_day"] = round(s["views"] / max(0.5, age_days), 1)
        enriched["engagement_rate"] = round((s["likes"] + s["comments"]) / s["views"], 4) if s["views"] else 0.0
        out.append(enriched)
    _classify_performance(out)
    return out


def _classify_performance(videos: list[dict]) -> None:
    cohorts = {
        "new": [v for v in videos if 1 <= v.get("age_days", 0) < 7 and v.get("metrics_status") == "ok"],
        "mid": [v for v in videos if 7 <= v.get("age_days", 0) < 30 and v.get("metrics_status") == "ok"],
        "old": [v for v in videos if v.get("age_days", 0) >= 30 and v.get("metrics_status") == "ok"],
    }
    for v in videos:
        age = v.get("age_days", 0)
        if age < 1:
            v["performance_label"] = "too_new"
            continue
        cohort = cohorts["new"] if age < 7 else cohorts["mid"] if age < 30 else cohorts["old"]
        if len(cohort) < 3:
            v["performance_label"] = "unknown"
            continue
        ordered = sorted(cohort, key=lambda x: x.get("views_per_day", 0))
        n = len(ordered)
        p20, p40, p60, p80 = (ordered[min(n - 1, max(0, int(n * p)))].get("views_per_day", 0) for p in (0.2, 0.4, 0.6, 0.8))
        vel = v.get("views_per_day", 0)
        v["performance_label"] = "winner" if vel >= p80 else "above_average" if vel >= p60 else "average" if vel >= p40 else "below_average" if vel >= p20 else "failure"


def collect_video_analytics(video_ids: list[str]) -> dict:
    from ..config import get_config
    cfg = get_config().get_path("youtube", {}) or {}
    client_id = env(cfg.get("client_id_env", "YOUTUBE_CLIENT_ID"))
    client_secret = env(cfg.get("client_secret_env", "YOUTUBE_CLIENT_SECRET"))
    refresh_token = env(cfg.get("refresh_token_env", "YOUTUBE_REFRESH_TOKEN"))
    if not (client_id and client_secret and refresh_token):
        LOG.warning("Missing OAuth credentials for YouTube Analytics API")
        return {}
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(token=None, refresh_token=refresh_token, token_uri=cfg.get("token_uri", "https://oauth2.googleapis.com/token"), client_id=client_id, client_secret=client_secret)
    analytics = build("youtubeAnalytics", "v2", credentials=creds)
    result = {}
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=60)
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            res = analytics.reports().query(ids="channel==MINE", startDate=start.isoformat(), endDate=today.isoformat(), metrics="averageViewDuration,estimatedMinutesWatched", dimensions="video", filters=f"video=={','.join(batch)}").execute()
            for row in res.get("rows", []):
                result[row[0]] = {"averageViewDuration": row[1], "estimatedMinutesWatched": row[2]}
        except Exception as exc:
            LOG.error("YouTube Analytics API fetch failed: %s", exc)
    return result


def update_performance_records(ledger_entries: list[dict]) -> None:
    perf_path = repo_root() / "data" / "performance.json"
    try:
        data = json.loads(perf_path.read_text(encoding="utf-8")) if perf_path.exists() else {"videos": []}
    except (json.JSONDecodeError, OSError):
        data = {"videos": []}
    existing = {v.get("video_id"): v for v in data.get("videos", []) if v.get("video_id")}
    records = []
    for entry in ledger_entries:
        upload = entry.get("upload") or {}
        vid = upload.get("video_id") or upload.get("id")
        if not vid or vid == "dry-run":
            continue
        topic, concept = entry.get("topic") or {}, entry.get("concept") or {}
        record = dict(existing.get(vid, {}))
        record.update({
            "video_id": vid,
            "title": upload.get("title") or topic.get("title", ""),
            "topic_hash": topic.get("topic_hash", topic.get("content_hash", entry.get("topic_hash", ""))),
            "topic_family": topic.get("topic_family", topic.get("family", "")),
            "hook_type": concept.get("hook_type", "unknown"),
            "chosen_angle": concept.get("chosen_angle", ""),
            "emotional_driver": concept.get("emotional_driver", "unknown"),
            "duration_seconds": concept.get("duration_seconds", entry.get("duration_seconds")),
            "duration_bucket": concept.get("duration_bucket", entry.get("duration_bucket", "")),
            "title_pattern": concept.get("title_pattern", entry.get("title_pattern", "")),
            "visual_sources": concept.get("visual_sources", entry.get("visual_sources", {})),
            "strategy_version": concept.get("strategy_version", entry.get("strategy_version", 0)),
        })
        records.append(record)
    if records:
        updated = collect_performance_data(records)
        retention = collect_video_analytics([r["video_id"] for r in updated])
        for r in updated:
            stats = retention.get(r["video_id"])
            if stats:
                r["retention_seconds"] = stats.get("averageViewDuration")
                r["minutes_watched"] = stats.get("estimatedMinutesWatched")
                r["analytics_status"] = "ok"
            else:
                r.setdefault("retention_seconds", None)
                r.setdefault("minutes_watched", None)
                r["analytics_status"] = "unavailable"
            existing[r["video_id"]] = r
    data["videos"] = list(existing.values())
    data["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
    history = data.setdefault("history", [])
    history.append({"timestamp": data["last_fetch_at"], "videos": [dict(v) for v in data["videos"]]})
    data["history"] = history[-10:]
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    perf_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOG.info("Updated performance records for %d videos", len(records))
