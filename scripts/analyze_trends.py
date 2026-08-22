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
    
    try:
        channels_response = youtube.channels().list(mine=True, part="id").execute()
        channel_id = channels_response["items"][0]["id"]
    except Exception as e:
        LOG.error(f"Failed to fetch channel ID: {e}")
        return None

    today = datetime.date.today()
    thirty_days_ago = today - datetime.timedelta(days=30)
    seven_days_ago = today - datetime.timedelta(days=7)
    
    def get_metrics(start_date, end_date):
        res = analytics.reports().query(
            ids=f"channel==MINE",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration",
        ).execute()
        if not res.get("rows"):
            return {"views": 0, "estimatedMinutesWatched": 0, "averageViewDuration": 0}
        row = res["rows"][0]
        return {
            "views": row[0],
            "estimatedMinutesWatched": row[1],
            "averageViewDuration": row[2]
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

    report = f"## YouTube Trends Report\n\n**Weekly**\n- Views: {stats['weekly']['views']}\n- Watch Time (min): {stats['weekly']['estimatedMinutesWatched']}\n- Avg View Duration (s): {stats['weekly']['averageViewDuration']}\n\n**Monthly**\n- Views: {stats['monthly']['views']}\n- Watch Time (min): {stats['monthly']['estimatedMinutesWatched']}\n- Avg View Duration (s): {stats['monthly']['averageViewDuration']}\n"
    
    prompt = f"""You are the AI producer for an automated YouTube Shorts channel.
Here are the latest channel metrics:
{report}

Based on this, suggest 2-3 short, punchy sentences to update our `goal_summary` in config/goal.yaml. The goal summary dictates what topics the AI chooses and how it writes scripts. Focus on what might increase retention or CTR. 
Respond ONLY with a JSON object containing one key: `new_goal_summary`.
"""
    llm = LLMRouter()
    try:
        res = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=300)
        new_goal = res.get("new_goal_summary")
        if new_goal:
            report += f"\n## AI Prompt Update\n**New Goal Summary:**\n{new_goal}\n"
            
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
