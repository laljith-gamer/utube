import json, glob
runs = sorted(glob.glob('runs/run_*'))
with open(runs[-1] + '/2_scored_candidates.json', encoding='utf-8') as f:
    c = json.load(f)[0]
    print(c['total_score'])
from pipeline.config import get_config
cfg = get_config()
print(cfg.get_path('topic_scoring', {}))
