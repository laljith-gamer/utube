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
    """Add a 'performance_label' to each video based on relative stats."""
    # Only classify videos that have been out for at least 24h
    valid = [v for v in videos if v.get("age_days", 0) >= 1.0 and "views_per_day" in v]
    if len(valid) < 5:
        # Not enough baseline data to classify
        for v in videos:
            if "performance_label" not in v:
                v["performance_label"] = "unknown"
        return

    # Sort by velocity
    valid.sort(key=lambda x: x["views_per_day"])
    
    n = len(valid)
    p20 = valid[int(n * 0.2)]["views_per_day"]
    p40 = valid[int(n * 0.4)]["views_per_day"]
    p60 = valid[int(n * 0.6)]["views_per_day"]
    p80 = valid[int(n * 0.8)]["views_per_day"]
    
    for v in videos:
        # Don't re-classify if already labeled, unless we want to dynamically update
        # Actually, dynamic update is good as older videos might die off
        if v.get("age_days", 0) < 1.0:
            v["performance_label"] = "too_new"
            continue
            
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
        
        # Merge back
        for u in updated:
            vid = u["video_id"]
            existing_videos[vid] = u
            
    data["videos"] = list(existing_videos.values())
    data["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
    
    # Save
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    LOG.info("Updated performance records for %d videos", len(to_update))
