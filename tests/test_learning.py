import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.stages.content_memory import ContentMemory
from pipeline.stages.topic_scoring import score_candidates

# Mock google modules before importing analyze_trends
mock_google = MagicMock()
sys.modules['google'] = mock_google
sys.modules['google.oauth2'] = mock_google
sys.modules['google.oauth2.credentials'] = mock_google
sys.modules['googleapiclient'] = mock_google
sys.modules['googleapiclient.discovery'] = mock_google

from scripts.analyze_trends import get_youtube_analytics


def test_content_memory_v3_combinations(tmp_path):
    videos = [
        {"video_id": "1", "performance_label": "winner", "hook_type": "contradiction", "emotional_driver": "curiosity", "age_days": 5},
        {"video_id": "2", "performance_label": "winner", "hook_type": "contradiction", "emotional_driver": "curiosity", "age_days": 10},
        {"video_id": "3", "performance_label": "average", "hook_type": "contradiction", "emotional_driver": "curiosity", "age_days": 12},
        # weak comb
        {"video_id": "4", "performance_label": "failure", "hook_type": "mystery", "emotional_driver": "fear", "age_days": 5},
        {"video_id": "5", "performance_label": "failure", "hook_type": "mystery", "emotional_driver": "fear", "age_days": 5},
    ]
    
    with patch("pipeline.stages.content_memory.repo_root") as mock_root:
        mock_root.return_value = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "performance.json").write_text(json.dumps({"videos": videos}))
        
        mem = ContentMemory()
        mem.refresh_from_performance()
        
        ctx = mem.get_context_for_scoring()
        assert "contradiction" in ctx["strong_hooks"]
        assert "mystery" in ctx["weak_hooks"]
        
        comb_found = any("contradiction|curiosity" == c["combination"] for c in ctx["strong_combinations"])
        assert comb_found, "Combination not extracted"


def test_topic_scoring_dynamic_strategy(tmp_path):
    with patch("pipeline.stages.topic_scoring.repo_root") as mock_root:
        mock_root.return_value = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        strategy = {
            "focused_themes": ["AI coding", "agents"],
            "avoid_themes": ["crypto", "web3"]
        }
        (data_dir / "dynamic_strategy.json").write_text(json.dumps(strategy))
        
        candidates = [
            {"title": "The future of AI coding agents", "summary": "How AI is taking over.", "source": "github", "normalized_hotness": 90},
            {"title": "Why Web3 and crypto failed", "summary": "The truth about blockchain.", "source": "github", "normalized_hotness": 90}
        ]
        
        with patch("pipeline.stages.topic_scoring._llm_batch_score", return_value=[{}, {}]):
            scored = score_candidates(candidates, content_memory={})
            
            assert scored[0]["title"] == "The future of AI coding agents"
            assert scored[0]["scores"]["strategy_alignment"] == 90
            assert scored[1]["scores"]["strategy_alignment"] == 0


def test_analyze_trends_control_flow():
    with patch("scripts.analyze_trends.env") as mock_env, \
         patch("scripts.analyze_trends.Credentials") as mock_creds, \
         patch("scripts.analyze_trends.build") as mock_build:
         
        mock_env.return_value = "fake"
        mock_analytics = MagicMock()
        mock_analytics.reports().query().execute.return_value = {"rows": [[100, 10, 5, 2, 1, 0, 0]]}
        
        def build_side_effect(service, version, **kwargs):
            if service == "youtubeAnalytics":
                return mock_analytics
            return MagicMock()
            
        mock_build.side_effect = build_side_effect
        
        res = get_youtube_analytics()
        assert res is not None
        assert "monthly" in res
        assert "best_days" in res
