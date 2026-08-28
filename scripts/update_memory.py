"""Updates content memory from YouTube performance data.

Intended to run on a schedule (e.g. weekly).
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ledger import Ledger
from pipeline.stages.content_memory import refresh_memory
from pipeline.utils import repo_root, setup_logging

LOG = logging.getLogger("update_memory")

def main():
    setup_logging(level="INFO")
    LOG.info("Starting memory refresh...")
    
    ledger = Ledger.load(repo_root() / "ledger.json")
    runs = ledger.data.get("runs", [])
    
    LOG.info(f"Loaded {len(runs)} runs from ledger.")
    
    refresh_memory(runs)
    
    LOG.info("Memory refresh complete.")

if __name__ == "__main__":
    main()
