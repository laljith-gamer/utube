# Graph Report - utube  (2026-08-22)

## Corpus Check
- 29 files · ~16,825 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 231 nodes · 521 edges · 19 communities (16 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e7e57ba9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- get_config
- orchestrator.py
- TTSRouter
- ImageRouter
- thumbnail.py
- themes.py
- audio.py
- discover.py
- captions.py
- Ledger
- assemble.py
- Config
- generate_youtube_token.py
- providers/__init__.py
- env
- CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `get_config()` - 40 edges
2. `produce_one()` - 20 edges
3. `TTSRouter` - 20 edges
4. `env()` - 17 edges
5. `repo_root()` - 15 edges
6. `LLMRouter` - 14 edges
7. `Ledger` - 12 edges
8. `main()` - 11 edges
9. `ImageRouter` - 11 edges
10. `discover_for_niche()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `repo_root()`  [EXTRACTED]
  scripts/analyze_trends.py → pipeline/utils.py
- `main()` --uses--> `Ledger`  [INFERRED]
  pipeline/orchestrator.py → pipeline/ledger.py
- `produce_one()` --uses--> `Ledger`  [INFERRED]
  pipeline/orchestrator.py → pipeline/ledger.py
- `_select_slots()` --uses--> `Ledger`  [INFERRED]
  pipeline/orchestrator.py → pipeline/ledger.py
- `main()` --calls--> `LLMRouter`  [EXTRACTED]
  scripts/analyze_trends.py → pipeline/providers/llm.py

## Import Cycles
- None detected.

## Communities (19 total, 3 thin omitted)

### Community 0 - "get_config"
Cohesion: 0.24
Nodes (13): _deep_merge(), get_config(), goal_summary(), Single source of truth for ALL configuration. Loads and deep-merges:…, Compact text summary of the channel goal — injected into LLM prompts., TTS router - config-driven multi-provider chain with full error visibility.…, build_research_brief(), fetch_source_text() (+5 more)

### Community 1 - "orchestrator.py"
Cohesion: 0.12
Nodes (28): utube — YouTube automation pipeline (5 videos/day)., main(), _pick_music(), produce_one(), _publish_at_for_slot(), Path, Top-level orchestrator. ALL configuration via config/*.yaml — no constants…, Resolve CLI flags into a concrete list of slot dicts the orchestrator will run. (+20 more)

### Community 2 - "TTSRouter"
Cohesion: 0.14
Nodes (9): Any, Synthesize with Camb.ai's streaming TTS endpoint. The narration script is…, One streaming Camb.ai request. Returns raw audio bytes., Split text into <=limit-char pieces at sentence/phrase boundaries. Keeps…, Synthesize with Microsoft Edge TTS at neutral rate/pitch. Sending non-zero…, Google Translate TTS — robust no-key fallback. gTTS has no rate parameter;…, Try each provider in order; return audio bytes from the first success.…, Synthesize with ElevenLabs TTS API. (+1 more)

### Community 3 - "ImageRouter"
Cohesion: 0.11
Nodes (10): ImageRouter, Image-generation router — config-driven, no hardcoded URLs/models/params., Stock B-roll router — Pexels + Pixabay, fully config-driven., StockRouter, SVD (Stable Video Diffusion) router — config-driven., Return MP4 bytes of an ~4s clip animated from the given image., VideoRouter, generate_visuals() (+2 more)

### Community 4 - "thumbnail.py"
Cohesion: 0.27
Nodes (18): Image, ImageDraw, ImageFont, _apply_portrait_overlays(), _draw_badge(), _draw_thumbnail_text(), _ellipsize(), _enhance_background() (+10 more)

### Community 5 - "themes.py"
Cohesion: 0.25
Nodes (13): all_themes(), _build_themes_for_lane(), _cli(), find_theme(), materialize_slot(), pick_themes(), Theme pool — 1000+ video theme seeds, generated from curated angle x seed…, Return every theme across every lane in `lanes_cfg`. (+5 more)

### Community 6 - "audio.py"
Cohesion: 0.28
Nodes (14): _atempo_factor(), _estimate_segment_timings(), _ffmpeg_concat(), _postprocess_segment(), _probe_duration(), Path, TTS narration: continuous master MP3, audio bitrate/codec from config. The…, Convert '+12%' / '-10%' / '+0%' -> 1.12 / 0.90 / 1.00. `atempo` accepts… (+6 more)

### Community 7 - "discover.py"
Cohesion: 0.33
Nodes (13): _cfg(), _devto(), discover_for_niche(), _github_trending(), _hackernews(), Any, Discover trending topics. All limits + UA + timeouts read from pipeline.yaml >…, Return up to N candidate topics from this slot's configured sources. (+5 more)

### Community 8 - "captions.py"
Cohesion: 0.28
Nodes (12): _group_wrapped_lines(), _layout_cfg(), _plain_text_cues(), Any, Path, Whisper captions — model size, device, words-per-cue from pipeline.yaml >…, _srt_block(), transcribe_to_srt() (+4 more)

### Community 9 - "Ledger"
Cohesion: 0.17
Nodes (4): Ledger, Path, Lightweight quota / topic-history ledger persisted to disk. Tracks: - recent…, Return theme ids used in the last `days`. Used to skip-pick repeats.

### Community 10 - "assemble.py"
Cohesion: 0.31
Nodes (10): assemble_video(), _concat(), _final_mux(), Path, Final FFmpeg assembly: motion-only scenes + caption burn-in + music duck. Per…, Synthesize a moving gradient as a no-real-footage fallback. Uses ffmpeg's…, Loop and crop any input clip to exactly `dur` seconds at the target portrait…, _render_from_video() (+2 more)

### Community 11 - "Config"
Cohesion: 0.33
Nodes (5): dict, Config, Any, Plain dict with attribute access and a `get_path` helper for nested keys., `cfg.get_path('llm.providers.nvidia_nim.model')`

### Community 15 - "env"
Cohesion: 0.20
Nodes (12): LLMRouter, _looks_like_json(), _parse_json(), Any, LLM provider router — fully driven by config/providers.yaml > llm. No URLs,…, Cheap test: does this string contain a JSON object (vs prose)?, Strict json.loads → strip code fences → first {…} block → repair-truncated., Tries each provider in `llm.chain` until one succeeds. (+4 more)

## Knowledge Gaps
- **1 isolated node(s):** `graphify`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_config()` connect `get_config` to `orchestrator.py`, `TTSRouter`, `ImageRouter`, `thumbnail.py`, `themes.py`, `audio.py`, `discover.py`, `captions.py`, `assemble.py`, `Config`, `env`?**
  _High betweenness centrality (0.249) - this node is a cross-community bridge._
- **Why does `TTSRouter` connect `TTSRouter` to `get_config`, `orchestrator.py`, `audio.py`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `produce_one()` connect `orchestrator.py` to `get_config`, `TTSRouter`, `ImageRouter`, `themes.py`, `discover.py`, `Ledger`, `assemble.py`, `env`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **What connects `graphify` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `orchestrator.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12299465240641712 - nodes in this community are weakly interconnected._
- **Should `TTSRouter` be split into smaller, more focused modules?**
  _Cohesion score 0.13852813852813853 - nodes in this community are weakly interconnected._
- **Should `ImageRouter` be split into smaller, more focused modules?**
  _Cohesion score 0.11384615384615385 - nodes in this community are weakly interconnected._