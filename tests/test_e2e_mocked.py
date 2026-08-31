import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Mock external dependencies that might not be installed locally
for mod in [
    'f5_tts', 'f5_tts.infer', 'f5_tts.infer.utils_infer', 'f5_tts.model',
    'faster_whisper',
    'moviepy', 'moviepy.editor',
]:
    sys.modules[mod] = MagicMock()

from pipeline.orchestrator import produce_one
from pipeline.ledger import Ledger
from pipeline.repetition import RepetitionReport


@pytest.fixture
def mock_ledger(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text('{"topics": {"global": {}}, "families": {}}', encoding="utf-8")
    return Ledger(ledger_path)


@pytest.fixture
def mock_providers():
    with patch("pipeline.stages.discover.discover_candidates") as m_discover, \
         patch("pipeline.stages.research.fetch_source_text") as m_scrape, \
         patch("pipeline.providers.llm.LLMRouter.chat_json") as m_llm_json, \
         patch("pipeline.providers.llm.LLMRouter.chat") as m_llm_text, \
         patch("pipeline.providers.image.ImageRouter.generate") as m_img, \
         patch("pipeline.providers.video.VideoRouter.animate") as m_vid, \
         patch("pipeline.providers.tts.TTSRouter.synthesize") as m_tts, \
         patch("pipeline.providers.stock.StockRouter.find_video") as m_stock, \
         patch("pipeline.providers.youtube.upload_video") as m_upload, \
         patch("pipeline.providers.llm.env", return_value="fake_api_key"):

        # Mock Discovery
        m_discover.return_value = [
            {
                "title": "A Mocked Discover Title",
                "url": "https://example.com/mock",
                "source": "hackernews",
                "summary": "Mocked summary of discovery.",
                "source_score": 0.9,
                "freshness_score": 0.9,
                "source_quality_score": 0.9,
                "keywords": ["mock", "test"]
            }
        ]

        # Mock Scrape
        m_scrape.return_value = "This is a mocked scraped article about the mocked title."

        # Mock LLM JSON (concept, script, script_qc, visual_qc, etc)
        def mock_llm_json(messages, **kwargs):
            prompt = messages[0]["content"] if messages else ""
            if "different angles" in prompt or "winning_hooks" in prompt:
                # Concept stage
                return {
                    "angles": [
                        {
                            "chosen_angle": "The mock angle",
                            "hook_type": "contradiction",
                            "curiosity_gap": "Why mocks are better than reality",
                            "emotional_driver": "curiosity",
                            "concept_score": 85,
                            "explanation": "Mocked concept"
                        }
                    ]
                }
            elif "Voice persona:" in prompt or "Visual style:" in prompt:
                # Script stage
                return {
                    "title": "Mocked Title for Video",
                    "description": "Mocked desc",
                    "hashtags": ["mock", "test"],
                    "hook": "This is a mocked hook.",
                    "scenes": [
                        {"narration": "First scene starts here.", "visual_prompt": "Prompt 1", "caption": "Mock 1", "broll_keywords": ["mock"]},
                        {"narration": "Then we do this.", "visual_prompt": "Prompt 2", "caption": "Mock 2", "broll_keywords": ["mock"]},
                        {"narration": "After that happens.", "visual_prompt": "Prompt 3", "caption": "Mock 3", "broll_keywords": ["mock"]},
                        {"narration": "Finally we conclude.", "visual_prompt": "Prompt 4", "caption": "Mock 4", "broll_keywords": ["mock"]},
                        {"narration": "Just kidding one more.", "visual_prompt": "Prompt 5", "caption": "Mock 5", "broll_keywords": ["mock"]},
                    ],
                    "cta": "Like and subscribe mock.",
                    "thumbnail_prompt": "Mocked thumb prompt"
                }
            elif "Evaluate this YouTube Short script" in prompt:
                # Script QC stage
                return {
                    "passed": True,
                    "scores": {
                        "hook_strength": 90,
                        "clarity": 90,
                        "specificity": 90,
                        "story_progression": 90,
                        "payoff_strength": 90,
                        "natural_voice": 90,
                        "channel_fit": 90
                    },
                    "issues": [],
                    "feedback": "LGTM"
                }
            elif "VISUAL QC" in prompt:
                # Visual QC stage
                return {
                    "passed": True,
                    "issues": [],
                    "feedback": ""
                }
            
            # Default to Research brief if not matched
            return {
                "confidence": 95,
                "facts": ["Fact 1 mock", "Fact 2 mock"],
                "gaps": []
            }
        
        m_llm_json.side_effect = mock_llm_json

        # Mock LLM Text (research brief, thumbnail overlay)
        def mock_llm_text(messages, **kwargs):
            return "Mocked text response"
        m_llm_text.side_effect = mock_llm_text

        # Mock Image / Video Generation
        def mock_img(prompt, **kwargs):
            import base64
            # 1x1 transparent PNG
            return base64.b64decode(b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
        m_img.side_effect = mock_img

        def get_valid_mp4_bytes():
            import subprocess, tempfile
            from pathlib import Path
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp = f.name
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
                "-c:v", "libx264", tmp
            ], check=True, capture_output=True)
            with open(tmp, "rb") as f:
                data = f.read()
            Path(tmp).unlink()
            return data

        m_vid.return_value = get_valid_mp4_bytes()
        m_stock.return_value = get_valid_mp4_bytes()

        # Mock TTS Generation
        def mock_tts(text, voice=None, **kwargs):
            # Return a valid 2s silent mp3
            import subprocess, tempfile
            from pathlib import Path
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:a", "libmp3lame", tmp
            ], check=True, capture_output=True)
            with open(tmp, "rb") as f:
                data = f.read()
            Path(tmp).unlink()
            return data
        m_tts.side_effect = mock_tts

        # Mock YouTube Upload
        m_upload.return_value = "mocked_video_id"
        
        yield {
            "discover": m_discover,
            "llm_json": m_llm_json,
            "llm_text": m_llm_text,
            "img": m_img,
            "vid": m_vid,
            "tts": m_tts,
            "stock": m_stock,
            "upload": m_upload,
        }

@patch("pipeline.orchestrator.upload_video")
@patch("pipeline.stages.captions.transcribe_to_srt")
def test_pipeline_e2e_mocked(mock_captions, mock_upload, mock_providers, mock_ledger):
    # Mock captions generator to just create empty SRT/ASS
    def fake_captions(audio_path, out_path):
        ass = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Mocked caption\n"
        )
        out_path.write_text(ass, encoding="utf-8")
        return out_path
    mock_captions.side_effect = fake_captions
    # Mock upload return value to be JSON serializable
    mock_upload.return_value = {"videoId": "mock123", "url": "https://youtube.com/shorts/mock123"}
    
    # Run pipeline in script-only=False, upload=True (mocked)
    result = produce_one(upload=True, skip_svd=True, script_only=False, ledger=mock_ledger)
    assert result["ok"] is True, f"Result failed: {result}"
    out_dir = Path(result["out_dir"])
    assert out_dir.exists()
    
    # Check that key artifacts were created
    assert (out_dir / "5_script_v1.json").exists()
    assert (out_dir / "mocked-title-for-video.mp4").exists()
    assert (out_dir / "thumbnail.jpg").exists()

    # Check LLM was called
    assert mock_providers["llm_json"].call_count >= 3
    # Check TTS was called for the scenes
    assert mock_providers["tts"].call_count >= 1
    # Check upload was called
    mock_upload.assert_called_once()
