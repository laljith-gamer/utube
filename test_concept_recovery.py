import os
import sys
import unittest

# Need to make sure we can import from pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.stages import concept
from pipeline.providers.llm import ProviderStatus

# Minimal mock configuration
import pipeline.config as config_mod

class DummyConfig:
    def get(self, key, default=None):
        if key == "concept":
            return {"llm": {"max_tokens": 1000}, "min_concept_score": 70}
        if key == "llm":
            return {
                "chain": ["openrouter_fast", "nvidia_fast"],
                "providers": {
                    "openrouter_fast": {"model": "deepseek/deepseek-v4-flash-0731", "api_key_env": "FAKE"},
                    "nvidia_fast": {"model": "nvidia/nemotron-3.5-lightning-30b-a3b", "api_key_env": "FAKE"}
                }
            }
        return default
        
    def get_path(self, key, default=None):
        return self.get(key, default)

config_mod.get_config = lambda: DummyConfig()
os.environ["FAKE"] = "dummy_key"

config_mod.goal_summary = lambda: "Testing goal"


class TestConceptRecovery(unittest.TestCase):
    def setUp(self):
        self.topic = {
            "title": "Artificial Intelligence in 2026",
            "summary": "AI has progressed rapidly...",
            "source": "Tech News",
            "total_score": 85.0
        }

    def test_openrouter_length_recovery(self):
        """Simulate OpenRouter truncating with finish_reason=length."""
        os.environ["CONCEPT_FAULT_INJECTION"] = "openrouter_length"
        res = concept.generate_concept(self.topic)
        self.assertIsNotNone(res)
        self.assertEqual(res["validation_status"], "valid")
        del os.environ["CONCEPT_FAULT_INJECTION"]

    def test_nvidia_truncated_recovery(self):
        """Simulate NVIDIA returning malformed JSON."""
        os.environ["CONCEPT_FAULT_INJECTION"] = "nvidia_truncated"
        res = concept.generate_concept(self.topic)
        self.assertIsNotNone(res)
        self.assertEqual(res["validation_status"], "valid")
        del os.environ["CONCEPT_FAULT_INJECTION"]

if __name__ == '__main__':
    unittest.main()
