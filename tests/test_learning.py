import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.stages.content_memory import ContentMemory
from pipeline.stages.topic_scoring import score_candidates

mock_google = MagicMock()
sys.modules["google"] = mock_google
sys.modules["google.oauth2"] = mock_google
sys.modules["google.oauth2.credentials"] = mock_google
sys.modules["googleapiclient"] = mock_google
sys.modules["googleapiclient.discovery"] = mock_google

from scripts.analyze_trends import get_youtube_analytics


def _video(i, label, family="ai scams", hook="personal_danger", emotion="fear"):
    return {
        "video_id": str(i), "performance_label": label, "topic_family": family,
        "hook_type": hook, "emotional_driver": emotion, "duration_seconds": 31,
        "title": f"Why this {family} matters?", "age_days": 5, "topic_hash": f"hash-{i}",
        "published_at": f"2026-08-{10+i:02d}T00:00:00+00:00",
    }


def test_content_memory_v3_dimensions(tmp_path):
    videos = [_video(i, "winner") for i in range(1, 7)]
    videos += [_video(i, "failure", family="old tech", hook="mystery", emotion="curiosity") for i in range(7, 13)]
    with patch("pipeline.stages.content_memory.repo_root", return_value=tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        (data_dir / "performance.json").write_text(json.dumps({"videos": videos}))
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "quality.yaml").write_text("content_memory:\n  min_samples_for_pattern: 5\n")
        mem = ContentMemory(); mem.refresh_from_performance(); ctx = mem.get_context_for_scoring()
        assert any(x["key"] == "ai scams" for x in ctx["strong_topic_families"])
        assert any(x["key"] == "personal_danger" for x in ctx["strong_hooks"])
        assert any(x["key"] == "ai scams|personal_danger|fear" for x in ctx["strong_combinations"])
        assert any(x["key"] == "old tech" for x in ctx["weak_topic_families"])


def test_memory_changes_topic_ranking(tmp_path):
    with patch("pipeline.stages.topic_scoring.repo_root", return_value=tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        strategy = {"focused_themes": [], "avoid_themes": []}
        (data_dir / "dynamic_strategy.json").write_text(json.dumps(strategy))
        memory = {
            "winning_patterns": {"topic_families": {"ai scams": {"posterior_mean": 0.9, "evidence_strength": 1.0}}},
            "weak_patterns": {"topic_families": {}},
            "strong_hooks": [], "recent_hashes": []
        }
        candidates = [
            {"title": "AI voice cloning scam explained", "summary": "A concrete consumer risk", "source": "github", "normalized_hotness": 80},
            {"title": "History of an old computer", "summary": "A technology history story", "source": "github", "normalized_hotness": 80},
        ]
        with patch("pipeline.stages.topic_scoring._llm_batch_score", return_value=[{}, {}]):
            scored = score_candidates(candidates, content_memory=memory)
        assert scored[0]["title"] == "AI voice cloning scam explained"
        assert scored[0]["memory_adjustment"] > 0


def test_dynamic_strategy_alignment(tmp_path):
    with patch("pipeline.stages.topic_scoring.repo_root", return_value=tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        (data_dir / "dynamic_strategy.json").write_text(json.dumps({"focused_themes": ["AI coding", "agents"], "avoid_themes": ["crypto", "web3"]}))
        candidates = [
            {"title": "The future of AI coding agents", "summary": "How AI is taking over.", "source": "github", "normalized_hotness": 90},
            {"title": "Why Web3 and crypto failed", "summary": "The truth about blockchain.", "source": "github", "normalized_hotness": 90},
        ]
        with patch("pipeline.stages.topic_scoring._llm_batch_score", return_value=[{}, {}]):
            scored = score_candidates(candidates, content_memory={})
        assert scored[0]["scores"]["strategy_alignment"] == 90
        assert scored[1]["scores"]["strategy_alignment"] == 0


def test_analyze_trends_control_flow():
    with patch("scripts.analyze_trends.env", return_value="fake"), patch("scripts.analyze_trends.Credentials"), patch("scripts.analyze_trends.build") as mock_build:
        mock_analytics = MagicMock()
        mock_analytics.reports().query().execute.return_value = {"rows": [[100, 10, 5, 2, 1, 0, 0]]}
        mock_build.return_value = mock_analytics
        res = get_youtube_analytics()
        assert res is not None and "monthly" in res and "best_days" in res
