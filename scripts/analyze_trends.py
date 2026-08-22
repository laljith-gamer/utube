import datetime
import json
import logging
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.providers.llm import LLMRouter
from pipeline.utils import env, repo_root

LOG = logging.getLogger("analyze_trends")

def get_youtube_analytics():
    client_id = env("YOUTUBE_CLIENT_ID")
    client_secret = env("YOUTUBE_CLIENT_SECRET")
    refresh_token = env("YOUTUBE_REFRESH_TOKEN")
    
    if not client_id or not client_secret or not refresh_token:
        LOG.error("Missing YouTube credentials in env")
        return None
        
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
    )
    
    youtube = build("youtube", "v3", credentials=creds)
    analytics = build("youtubeAnalytics", "v2", credentials=creds)
    
    channel_id = "MINE"

    today = datetime.date.today()
    thirty_days_ago = today - datetime.timedelta(days=30)
    seven_days_ago = today - datetime.timedelta(days=7)
    
    def get_metrics(start_date, end_date):
        res = analytics.reports().query(
            ids=f"channel==MINE",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration,likes,comments,shares,subscribersGained",
        ).execute()
        if not res.get("rows"):
            return {"views": 0, "estimatedMinutesWatched": 0, "averageViewDuration": 0, "likes": 0, "comments": 0, "shares": 0, "subscribersGained": 0}
        row = res["rows"][0]
        return {
            "views": row[0],
            "estimatedMinutesWatched": row[1],
            "averageViewDuration": row[2],
            "likes": row[3],
            "comments": row[4],
            "shares": row[5],
            "subscribersGained": row[6]
        }
        
    monthly = get_metrics(thirty_days_ago, today)
    weekly = get_metrics(seven_days_ago, today)
    
    return {
        "channel_id": channel_id,
        "monthly": monthly,
        "weekly": weekly
    }

def main():
    logging.basicConfig(level=logging.INFO)
    stats = get_youtube_analytics()
    if not stats:
        LOG.warning("Could not fetch analytics, exiting.")
        return

    def format_stats(s):
        return f"- Views: {s['views']}\n- Watch Time (min): {s['estimatedMinutesWatched']}\n- Avg View Duration (s): {s['averageViewDuration']}\n- Likes: {s['likes']}\n- Comments: {s['comments']}\n- Shares: {s['shares']}\n- Subs Gained: {s['subscribersGained']}"

    report = f"## YouTube Trends Report\n\n**Weekly**\n{format_stats(stats['weekly'])}\n\n**Monthly**\n{format_stats(stats['monthly'])}\n"
    
    prompt = f"""You are a Master YouTube Strategist analyzing an automated channel's performance.
Here are the latest channel metrics:
{report}

Based on these detailed metrics, perform a deep analysis. Think about audience engagement (likes/comments vs views), shareability, and viewer retention.
Suggest a comprehensive, high-converting strategy update. We need a new `goal_summary` for config/goal.yaml. This dictates what topics the AI chooses and how it writes scripts. 
Make the new summary detailed (4-5 sentences), focusing on specific script structures, pacing, hooks, and topic angles that will drastically improve our stats.

Respond ONLY with a JSON object containing two keys:
`analysis_rationale`: A short paragraph explaining your strategic reasoning based on the numbers.
`new_goal_summary`: The detailed 4-5 sentence summary to use moving forward.
"""
    llm = LLMRouter("llm_script")
    try:
        res = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=600)
        new_goal = res.get("new_goal_summary")
        rationale = res.get("analysis_rationale")
        if new_goal:
            report += f"\n## AI Deep Analysis\n**Rationale:**\n{rationale}\n\n**New Goal Summary:**\n{new_goal}\n"
            
            goal_yaml_path = repo_root() / "config" / "goal.yaml"
            with open(goal_yaml_path, "r", encoding="utf-8") as f:
                goal_data = yaml.safe_load(f) or {}
                
            goal_data["summary"] = new_goal
            
            with open(goal_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(goal_data, f, default_flow_style=False)
                
            LOG.info("Updated goal.yaml with new summary.")
    except Exception as e:
        LOG.error(f"Failed to generate AI insights: {e}")
        
    report_path = repo_root() / "trend_report.md"
    report_path.write_text(report, encoding="utf-8")
    LOG.info("Saved trend_report.md")

if __name__ == "__main__":
    main()
