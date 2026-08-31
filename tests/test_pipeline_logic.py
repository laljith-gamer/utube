import pytest
from pipeline.repetition import RepetitionChecker
from pipeline.stages.audio_validation import validate_audio

def test_audio_validation_pass():
    script = {
        "hook": "This is a test.",
        "scenes": [{"narration": "We are verifying audio."}],
        "cta": "Like and subscribe."
    }
    asr_text = "This is a test. We are verifying audio. Like and subscribe."
    ref_text = "Some call it fate, I call it faith."
    
    # Should not raise
    validate_audio(script, asr_text, ref_text)

def test_audio_validation_fail_fidelity():
    script = {
        "hook": "This is a very long script that has many specific words.",
        "scenes": [{"narration": "If none of this is spoken, it fails."}],
        "cta": "Like and subscribe."
    }
    asr_text = "This is a test."
    ref_text = "Some call it fate, I call it faith."
    
    with pytest.raises(ValueError, match="ASR fidelity too low"):
        validate_audio(script, asr_text, ref_text)

def test_audio_validation_fail_leakage():
    script = {
        "hook": "This is a test.",
        "scenes": [{"narration": "We are verifying audio."}],
        "cta": "Like and subscribe."
    }
    # ASR text includes the reference text (leakage)
    asr_text = "This is a test. Some call it fate, I call it faith. We are verifying audio."
    ref_text = "Some call it fate, I call it faith."
    
    with pytest.raises(ValueError, match="TTS leaked reference text"):
        validate_audio(script, asr_text, ref_text)

def test_repetition_checker_style_memory():
    # Provide dummy archive data
    archive_data = [
        {"hook": "Did you know that X is Y?", "cta": "Subscribe for more!"},
        {"hook": "Here is a secret about Z.", "cta": "Leave a comment."}
    ]
    
    class DummyChecker(RepetitionChecker):
        def _load_archive(self):
            self.archive = archive_data
            
    checker = DummyChecker()
    memory = checker.get_style_memory(archive_data)
    
    # Should extract hooks and CTAs from the archive
    assert "Did you know that X is Y?" in memory["recent_hooks"]
    assert "Subscribe for more!" in memory["recent_ctas"]
