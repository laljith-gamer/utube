"""Thursday learning loop: refresh performance, memory, then generate strategy."""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ledger import Ledger
from pipeline.providers.llm import LLMRouter
from pipeline.stages.analytics import update_performance_records
from pipeline.stages.content_memory import ContentMemory
from pipeline.utils import env, repo_root

LOG = logging.getLogger("analyze_trends")


def get_youtube_analytics() -> dict | None:
    from pipeline.config import get_config
    cfg = get_config().get_path("youtube", {}) or {}
    client_id = env(cfg.get("client_id_env", "YOUTUBE_CLIENT_ID"))
    client_secret = env(cfg.get("client_secret_env", "YOUTUBE_CLIENT_SECRET"))
    refresh_token = env(cfg.get("refresh_token_env", "YOUTUBE_REFRESH_TOKEN"))
    if not client_id or not client_secret or not refresh_token:
        LOG.error("Missing YouTube OAuth credentials")
        return None

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=cfg.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=client_id,
        client_secret=client_secret,
    )
    analytics = build("youtubeAnalytics", "v2", credentials=creds)
    today = datetime.date.today()
    thirty_days_ago = today - datetime.timedelta(days=30)
    seven_days_ago = today - datetime.timedelta(days=7)

    def get_metrics(start_date: datetime.date, end_date: datetime.date) -> dict:
        res = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration,likes,comments,shares,subscribersGained",
        ).execute()
        if not res.get("rows"):
            return {k: 0 for k in ("views", "estimatedMinutesWatched", "averageViewDuration", "likes", "comments", "shares", "subscribersGained")}
        row = res["rows"][0]
        return dict(zip(("views", "estimatedMinutesWatched", "averageViewDuration", "likes", "comments", "shares", "subscribersGained"), row))

    def get_day_of_week_stats(start_date: datetime.date, end_date: datetime.date) -> dict:
        try:
            res = analytics.reports().query(
                ids="channel==MINE",
                startDate=start_date.isoformat(),
                endDate=end_date.isoformat(),
                metrics="views",
                dimensions="dayOfWeek",
            ).execute()
            return {row[0]: row[1] for row in res.get("rows", [])}
        except Exception as exc:
            LOG.warning("Could not fetch dayOfWeek: %s", exc)
            return {}

    return {
        "channel_id": "MINE",
        "monthly": get_metrics(thirty_days_ago, today),
        "weekly": get_metrics(seven_days_ago, today),
        "best_days": get_day_of_week_stats(thirty_days_ago, today),
    }


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def build_learning_summary() -> dict:
    root = repo_root()
    performance = _load_json(root / "data" / "performance.json", {"videos": []})
    memory = _load_json(root / "data" / "content_memory.json", {})
    videos = performance.get("videos", [])

    def top_patterns(category: str, limit: int = 8):
        vals = []
        for key, value in (memory.get("winning_patterns", {}).get(category, {}) or {}).items():
            vals.append({"key": key, **value})
        return sorted(vals, key=lambda x: (x.get("posterior_mean", 0), x.get("evidence_strength", 0)), reverse=True)[:limit]

    def weak_patterns(category: str, limit: int = 8):
        vals = []
        for key, value in (memory.get("weak_patterns", {}).get(category, {}) or {}).items():
            vals.append({"key": key, **value})
        return sorted(vals, key=lambda x: (x.get("posterior_mean", 1), -x.get("evidence_strength", 0)))[:limit]

    winners = [v for v in videos if v.get("performance_label") in ("winner", "above_average")]
    losers = [v for v in videos if v.get("performance_label") in ("failure", "below_average")]
    return {
        "video_count": len(videos),
        "winner_count": len(winners),
        "loser_count": len(losers),
        "top_winners": [
            {k: v.get(k) for k in ("title", "topic_family", "hook_type", "emotional_driver", "duration_bucket", "title_pattern", "visual_source", "views", "views_per_day", "engagement_rate", "performance_label")}
            for v in sorted(winners, key=lambda x: x.get("views_per_day", 0), reverse=True)[:8]
        ],
        "top_losers": [
            {k: v.get(k) for k in ("title", "topic_family", "hook_type", "emotional_driver", "duration_bucket", "title_pattern", "visual_source", "views", "views_per_day", "engagement_rate", "performance_label")}
            for v in sorted(losers, key=lambda x: x.get("views_per_day", 0))[:8]
        ],
        "winning_topic_families": top_patterns("topic_families"),
        "weak_topic_families": weak_patterns("topic_families"),
        "winning_hooks": top_patterns("hook_types"),
        "weak_hooks": weak_patterns("hook_types"),
        "winning_emotions": top_patterns("emotional_drivers"),
        "winning_durations": top_patterns("duration_buckets"),
        "winning_title_patterns": top_patterns("title_patterns"),
        "winning_visual_sources": top_patterns("visual_sources"),
        "winning_combinations": top_patterns("topic_hook_emotion"),
    }


def _write_strategy(strategy: dict, video_count: int) -> None:
    root = repo_root()
    path = root / "data" / "dynamic_strategy.json"
    previous = _load_json(path, {})
    previous_version = int(previous.get("strategy_version", 0) or 0)
    clean = {
        "version": 1,
        "strategy_version": previous_version + 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "based_on_video_count": video_count,
        "confidence": max(0.0, min(1.0, min(1.0, video_count / 30.0))),
        "overall_direction": str(strategy.get("overall_direction", "Stay focused on surprising technology that matters to ordinary people.")),
        "focused_themes": list(strategy.get("focused_themes", []))[:8],
        "avoid_themes": list(strategy.get("avoid_themes", []))[:8],
        "recommended_hooks": list(strategy.get("recommended_hooks", []))[:8],
        "avoid_hooks": list(strategy.get("avoid_hooks", []))[:8],
        "recommended_emotions": list(strategy.get("recommended_emotions", []))[:6],
        "duration_recommendation": str(strategy.get("duration_recommendation", "")),
        "experiments": list(strategy.get("experiments", []))[:8],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    LOG.info("Saved dynamic strategy v%d", clean["strategy_version"])


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    root = repo_root()
    stats = get_youtube_analytics()
    analytics_available = stats is not None
    if not analytics_available:
        LOG.warning("YouTube Analytics unavailable — refreshing learning state from ledger only")
        stats = {
            "weekly": {k: 0 for k in ("views", "estimatedMinutesWatched", "averageViewDuration", "likes", "comments", "shares", "subscribersGained")},
            "monthly": {k: 0 for k in ("views", "estimatedMinutesWatched", "averageViewDuration", "likes", "comments", "shares", "subscribersGained")},
            "best_days": {},
        }

    ledger = Ledger.load(root / "ledger.json")
    update_performance_records(ledger.data.get("runs", []))
    memory = ContentMemory()
    memory.refresh_from_performance()
    learning = build_learning_summary()

    def format_stats(s: dict) -> str:
        return (
            f"Views={s.get('views', 0)}, watch_minutes={s.get('estimatedMinutesWatched', 0)}, "
            f"avg_view_duration={s.get('averageViewDuration', 0)}, likes={s.get('likes', 0)}, "
            f"comments={s.get('comments', 0)}, shares={s.get('shares', 0)}, subs={s.get('subscribersGained', 0)}"
        )

    report = (
        "## YouTube Trends Report\n\n"
        f"**Weekly:** {format_stats(stats['weekly'])}\n\n"
        f"**Monthly:** {format_stats(stats['monthly'])}\n\n"
        f"**Views by day:** {json.dumps(stats.get('best_days', {}), sort_keys=True)}\n\n"
        f"## Learned Video-Level Patterns\n```json\n{json.dumps(learning, indent=2, default=str)}\n```\n"
    )

    prompt = f"""You are the weekly strategist for an automated YouTube Shorts channel.
Do not invent evidence. Prefer patterns with meaningful sample size and evidence strength.
Recent video-level performance and learned patterns are the primary evidence; channel aggregates are secondary.

CHANNEL ANALYTICS:
{format_stats(stats['weekly'])}
{format_stats(stats['monthly'])}
Day-of-week views: {json.dumps(stats.get('best_days', {}), sort_keys=True)}

LEARNING DATA:
{json.dumps(learning, indent=2, default=str)}

Create a conservative strategy for the NEXT publishing period.
Separate proven patterns from experiments. Do not overfit tiny samples.

Return ONLY JSON with:
analysis_rationale: concise evidence-based explanation
new_goal_summary: 4-5 sentences; preserve the channel identity and quality bar
 timing_strategy: practical scheduling guidance based only on available day data; tell the creator to use YouTube Studio's audience-online graph for the exact peak hour
 dynamic_strategy: object containing:
  overall_direction: string
  focused_themes: array
  avoid_themes: array
  recommended_hooks: array
  avoid_hooks: array
  recommended_emotions: array
  duration_recommendation: string
  experiments: array of specific testable hypotheses
"""

    llm = LLMRouter("llm_script")
    try:
        result = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=2200)
        if not isinstance(result, dict) or not isinstance(result.get("dynamic_strategy"), dict):
            raise ValueError("Strategist returned invalid dynamic_strategy JSON")
        dynamic = result["dynamic_strategy"]
        # Validate required strategy keys
        required_strategy_keys = ["overall_direction", "focused_themes", "avoid_themes",
                                  "recommended_hooks", "recommended_emotions"]
        missing_keys = [k for k in required_strategy_keys if k not in dynamic]
        if missing_keys:
            LOG.warning("Strategy missing keys: %s — filling with defaults", missing_keys)
            dynamic.setdefault("overall_direction", "Stay focused on surprising technology that matters to ordinary people.")
            dynamic.setdefault("focused_themes", [])
            dynamic.setdefault("avoid_themes", [])
            dynamic.setdefault("recommended_hooks", [])
            dynamic.setdefault("recommended_emotions", [])
        _write_strategy(dynamic, learning["video_count"])

        goal = result.get("new_goal_summary")
        if goal:
            goal_path = root / "config" / "goal.yaml"
            goal_data = yaml.safe_load(goal_path.read_text(encoding="utf-8")) or {}
            goal_data["summary"] = goal
            goal_path.write_text(yaml.safe_dump(goal_data, sort_keys=False), encoding="utf-8")

        report += (
            "\n## AI Deep Analysis\n"
            f"**Rationale:** {result.get('analysis_rationale', '')}\n\n"
            f"**New Goal Summary:** {result.get('new_goal_summary', '')}\n\n"
            f"**Timing & Scheduling Strategy:** {result.get('timing_strategy', '')}\n"
        )
    except Exception as exc:
        LOG.error("Strategist failed: %s", exc)
        raise

    (root / "trend_report.md").write_text(report, encoding="utf-8")
    LOG.info("Saved trend_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
