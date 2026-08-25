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
        def get_day_of_week_stats(start_date, end_date):
            try:
                res = analytics.reports().query(
                    ids=f"channel==MINE",
                    startDate=start_date.isoformat(),
                    endDate=end_date.isoformat(),
                    metrics="views",
                    dimensions="dayOfWeek"
                ).execute()
                if not res.get("rows"):
                    return {}
                return {row[0]: row[1] for row in res["rows"]}
            except Exception as e:
                LOG.warning(f"Could not fetch dayOfWeek: {e}")
                return {}

        monthly = get_metrics(thirty_days_ago, today)
        weekly = get_metrics(seven_days_ago, today)
        best_days = get_day_of_week_stats(thirty_days_ago, today)
        
        return {
            "channel_id": channel_id,
            "monthly": monthly,
            "weekly": weekly,
            "best_days": best_days
        }

def main():
    logging.basicConfig(level=logging.INFO)
    stats = get_youtube_analytics()
    if not stats:
        LOG.warning("Could not fetch analytics, exiting.")
        return

    def format_stats(s):
        return f"- Views: {s.get('views', 0)}\n- Watch Time (min): {s.get('estimatedMinutesWatched', 0)}\n- Avg View Duration (s): {s.get('averageViewDuration', 0)}\n- Likes: {s.get('likes', 0)}\n- Comments: {s.get('comments', 0)}\n- Shares: {s.get('shares', 0)}\n- Subs Gained: {s.get('subscribersGained', 0)}"

    report = f"## YouTube Trends Report\n\n**Weekly**\n{format_stats(stats['weekly'])}\n\n**Monthly**\n{format_stats(stats['monthly'])}\n"
    
    if stats.get("best_days"):
        report += "\n**Views by Day of Week (Last 30 Days)**\n"
        for day, views in stats["best_days"].items():
            report += f"- {day}: {views}\n"
    
    prompt = f"""You are a Master YouTube Strategist analyzing an automated channel's performance.
Here are the latest channel metrics:
{report}

Based on these detailed metrics, perform a deep analysis. Think about audience engagement, shareability, and viewer retention.
1. Suggest a comprehensive, high-converting strategy update. We need a new `goal_summary` for config/goal.yaml.
2. Provide a `timing_strategy`. Analyze the best days to post based on the data. Remind the creator to check their 'When your viewers are on YouTube' graph in YouTube Studio and advise them to publish 30-90 minutes before the peak hour to allow YouTube time to process and distribute the video.

Respond ONLY with a JSON object containing three keys:
`analysis_rationale`: A short paragraph explaining your strategic reasoning based on the numbers.
`new_goal_summary`: The detailed 4-5 sentence summary to use moving forward focusing on specific hooks and angles.
`timing_strategy`: Advice on upload scheduling and leveraging peak viewer times.
"""
    llm = LLMRouter("llm_script")
    try:
        res = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=800)
        new_goal = res.get("new_goal_summary")
        rationale = res.get("analysis_rationale")
        timing = res.get("timing_strategy")
        
        if new_goal:
            report += f"\n## AI Deep Analysis\n**Rationale:**\n{rationale}\n\n**New Goal Summary:**\n{new_goal}\n\n**Timing & Scheduling Strategy:**\n{timing}\n"
            
            goal_yaml_path = repo_root() / "config" / "goal.yaml"
            with open(goal_yaml_path, "r", encoding="utf-8") as f:
                goal_data = yaml.safe_load(f) or {}
                
            goal_data["summary"] = new_goal
            
            with open(goal_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(goal_data, f, default_flow_style=False)
                
            LOG.info("Updated goal.yaml with new summary.")
            
            # --- Auto-update prompts ---
            prompt_dir = repo_root() / "prompts"
            script_prompt_path = prompt_dir / "script.txt"
            topic_prompt_path = prompt_dir / "topic_select.txt"
            
            if script_prompt_path.exists() and topic_prompt_path.exists():
                script_txt = script_prompt_path.read_text(encoding="utf-8")
                topic_txt = topic_prompt_path.read_text(encoding="utf-8")
                
                update_prompt_msg = f'''You are a Prompt Engineer. We just updated our YouTube channel strategy based on latest metrics.
Rationale: {rationale}
New Goal: {new_goal}

Here is our current `script.txt` prompt:
---
{script_txt}
---

Here is our current `topic_select.txt` prompt:
---
{topic_txt}
---

Rewrite these two prompts. Keep ALL of the original structural constraints, output JSON formats, and strict rules intact.
However, gracefully weave the New Goal and Rationale into the stylistic instructions, hook guidelines, and topic selection criteria.
Make sure the updated prompts will naturally steer the AI to produce scripts and topics aligned with the new strategy.

Respond ONLY with a JSON object containing two keys:
`new_script_prompt`: The complete updated text for script.txt
`new_topic_prompt`: The complete updated text for topic_select.txt
'''
                try:
                    res_prompts = llm.chat_json([{"role": "user", "content": update_prompt_msg}], max_tokens=8000)
                    new_script = res_prompts.get("new_script_prompt")
                    new_topic = res_prompts.get("new_topic_prompt")
                    if new_script and new_topic:
                        script_prompt_path.write_text(new_script, encoding="utf-8")
                        topic_prompt_path.write_text(new_topic, encoding="utf-8")
                        LOG.info("Autonomously updated script.txt and topic_select.txt with new strategy.")
                except Exception as e:
                    LOG.error(f"Failed to auto-update prompts: {e}")
    except Exception as e:
        LOG.error(f"Failed to generate AI insights: {e}")
        
    report_path = repo_root() / "trend_report.md"
    report_path.write_text(report, encoding="utf-8")
    LOG.info("Saved trend_report.md")

if __name__ == "__main__":
    main()
