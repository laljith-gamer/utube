"""Performance collection via YouTube Data API v3.

Fetches video-level stats (views, likes, comments) and normalizes them by age
to classify performance (winner, average, failure).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..utils import env, repo_root

LOG = logging.getLogger("utube.analytics")


def collect_performance_data(published_videos: list[dict], api_key: str | None = None) -> list[dict]:
    """Fetch stats for a list of videos using YouTube Data API.

    Each input dict must have at least `video_id`.
    Expects to return enriched dictionaries with performance metrics.
    """
    key = api_key or env("YOUTUBE_API_KEY")
    if not key:
        LOG.warning("YOUTUBE_API_KEY not found. Cannot fetch video performance.")
        return published_videos

    video_ids = [v["video_id"] for v in published_videos if v.get("video_id")]
    if not video_ids:
        return published_videos

    # Batch into groups of 50
    stats_map = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "statistics,snippet",
            "id": ",".join(batch),
            "key": key
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            for item in r.json().get("items", []):
                stats = item.get("statistics", {})
                vid = item.get("id")
                stats_map[vid] = {
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "published_at": item.get("snippet", {}).get("publishedAt")
                }
        except Exception as e:
            LOG.error("YouTube Data API fetch failed: %s", e)

    out = []
    now = datetime.now(timezone.utc)
    
    for v in published_videos:
        vid = v.get("video_id")
        if not vid or vid not in stats_map:
            out.append(v)
            continue
            
        s = stats_map[vid]
        enriched = dict(v)
        enriched["views"] = s["views"]
        enriched["likes"] = s["likes"]
        enriched["comments"] = s["comments"]
        
        # Calculate age in days
        pub_dt = s.get("published_at")
        age_days = 1.0
        if pub_dt:
            try:
                # Handle ISO 8601 string from YouTube
                dt = datetime.fromisoformat(pub_dt.replace("Z", "+00:00"))
                age_days = max(0.1, (now - dt).total_seconds() / 86400.0)
            except ValueError:
                pass
                
        enriched["age_days"] = round(age_days, 2)
        
        # Calculate velocity (views per day)
        velocity = s["views"] / max(0.5, age_days)  # cap young video explosions
        enriched["views_per_day"] = round(velocity, 1)
        
        # Engagement rate (likes + comments) / views
        if s["views"] > 0:
            eng_rate = (s["likes"] + s["comments"]) / s["views"]
        else:
            eng_rate = 0.0
        enriched["engagement_rate"] = round(eng_rate, 4)
        
        out.append(enriched)

    _classify_performance(out)
    return out


def _classify_performance(videos: list[dict]) -> None:
    """Add a 'performance_label' to each video based on relative stats within age cohorts."""
    cohorts = {
        "new": [v for v in videos if 1.0 <= v.get("age_days", 0) < 7.0 and "views_per_day" in v],
        "mid": [v for v in videos if 7.0 <= v.get("age_days", 0) < 30.0 and "views_per_day" in v],
        "old": [v for v in videos if v.get("age_days", 0) >= 30.0 and "views_per_day" in v],
    }
    
    for v in videos:
        if v.get("age_days", 0) < 1.0:
            v["performance_label"] = "too_new"
            continue
            
        age = v.get("age_days", 0)
        if age < 7.0:
            cohort = cohorts["new"]
        elif age < 30.0:
            cohort = cohorts["mid"]
        else:
            cohort = cohorts["old"]
            
        if len(cohort) < 3:
            v["performance_label"] = "average"  # Not enough data in cohort
            continue
            
        cohort.sort(key=lambda x: x["views_per_day"])
        n = len(cohort)
        
        p20 = cohort[int(n * 0.2)]["views_per_day"]
        p40 = cohort[int(n * 0.4)]["views_per_day"]
        p60 = cohort[int(n * 0.6)]["views_per_day"]
        p80 = cohort[int(n * 0.8)]["views_per_day"]
        
        vel = v.get("views_per_day", 0)
        if vel >= p80:
            label = "winner"
        elif vel >= p60:
            label = "above_average"
        elif vel >= p40:
            label = "average"
        elif vel >= p20:
            label = "below_average"
        else:
            label = "failure"
            
        v["performance_label"] = label


def collect_video_analytics(video_ids: list[str]) -> dict:
    """Fetch retention metrics using YouTube Analytics API."""
    client_id = env("YOUTUBE_CLIENT_ID")
    client_secret = env("YOUTUBE_CLIENT_SECRET")
    refresh_token = env("YOUTUBE_REFRESH_TOKEN")
    
    if not client_id or not client_secret or not refresh_token:
        LOG.warning("Missing OAuth credentials for YouTube Analytics API.")
        return {}
        
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    import datetime
    
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
    )
    
    analytics = build("youtubeAnalytics", "v2", credentials=creds)
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=60)
    
    stats_map = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            res = analytics.reports().query(
                ids="channel==MINE",
                startDate=start_date.isoformat(),
                endDate=end_date.isoformat(),
                metrics="averageViewDuration,estimatedMinutesWatched",
                dimensions="video",
                filters=f"video=={','.join(batch)}"
            ).execute()
            
            for row in res.get("rows", []):
                vid = row[0]
                stats_map[vid] = {
                    "averageViewDuration": row[1],
                    "estimatedMinutesWatched": row[2]
                }
        except Exception as e:
            LOG.error("YouTube Analytics API fetch failed: %s", e)
            
    return stats_map


def update_performance_records(ledger_entries: list[dict]) -> None:
    """Sync ledger runs with the performance datastore."""
    perf_path = repo_root() / "data" / "performance.json"
    
    # Load existing
    if perf_path.exists():
        try:
            with open(perf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {"videos": []}
    else:
        data = {"videos": []}
        
    existing_videos = {v.get("video_id"): v for v in data.get("videos", []) if v.get("video_id")}
    
    # Find ledger entries that have video_ids but aren't in performance, or need update
    # Note: The ledger stores the entire run record, we want to extract concept/topic metadata
    to_update = []
    
    for entry in ledger_entries:
        vid = entry.get("upload", {}).get("video_id")
        if not vid:
            continue
            
        # Extract metadata
        topic = entry.get("topic", {})
        concept = entry.get("concept", {})
        
        record = existing_videos.get(vid, {})
        record.update({
            "video_id": vid,
            "title": entry.get("upload", {}).get("title") or topic.get("title", ""),
            "topic_hash": topic.get("topic_hash", ""),
            "hook_type": concept.get("hook_type", "unknown"),
            "chosen_angle": concept.get("chosen_angle", ""),
            "emotional_driver": concept.get("emotional_driver", "unknown"),
        })
        to_update.append(record)
        
    # Fetch latest stats
    if to_update:
        updated = collect_performance_data(to_update)
        
        # Also fetch Analytics API stats for retention
        vid_list = [u["video_id"] for u in updated if "video_id" in u]
        retention_stats = collect_video_analytics(vid_list)
        
        # Merge back
        for u in updated:
            vid = u["video_id"]
            if vid in retention_stats:
                u["retention_seconds"] = retention_stats[vid].get("averageViewDuration", 0)
                u["minutes_watched"] = retention_stats[vid].get("estimatedMinutesWatched", 0)
            existing_videos[vid] = u
            
    data["videos"] = list(existing_videos.values())
    data["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
    
    # Store snapshot
    if "history" not in data:
        data["history"] = []
    snapshot = {
        "timestamp": data["last_fetch_at"],
        "videos": [dict(v) for v in data["videos"]]
    }
    data["history"].append(snapshot)
    
    # Keep only last 10 snapshots to avoid huge files
    if len(data["history"]) > 10:
        data["history"] = data["history"][-10:]
    
    # Save
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    LOG.info("Updated performance records for %d videos", len(to_update))
