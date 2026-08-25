# Graph Report - utube  (2026-08-22)

## Corpus Check
- Corpus is ~16,050 words - fits in a single context window. You may not need a graph.

## Summary
- 224 nodes · 509 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Core Scripting
- Orchestration Utils
- TTS Generation
- Visual Generation
- Thumbnail Engine
- Themes Selection
- Audio Processing
- Content Discovery
- Captions Generation
- Ledger State
- Video Assembly
- Config Dictionaries
- YouTube Tokens
- Providers Init

## God Nodes (most connected - your core abstractions)
1. `get_config()` - 40 edges
2. `produce_one()` - 20 edges
3. `TTSRouter` - 20 edges
4. `env()` - 15 edges
5. `repo_root()` - 13 edges
6. `Ledger` - 12 edges
7. `LLMRouter` - 12 edges
8. `main()` - 11 edges
9. `ImageRouter` - 11 edges
10. `discover_for_niche()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `main()` --uses--> `Ledger`  [INFERRED]
  pipeline/orchestrator.py → pipeline/ledger.py
- `produce_one()` --uses--> `Ledger`  [INFERRED]
  pipeline/orchestrator.py → pipeline/ledger.py
- `_select_slots()` --uses--> `Ledger`  [INFERRED]
  pipeline/orchestrator.py → pipeline/ledger.py
- `main()` --calls--> `get_config()`  [EXTRACTED]
  pipeline/orchestrator.py → pipeline/config.py
- `produce_one()` --calls--> `get_config()`  [EXTRACTED]
  pipeline/orchestrator.py → pipeline/config.py

## Import Cycles
- None detected.

## Communities (15 total, 2 thin omitted)

### Community 0 - "Core Scripting"
Cohesion: 0.13
Nodes (25): _deep_merge(), get_config(), goal_summary(), Single source of truth for ALL configuration. Loads and deep-merges:…, Compact text summary of the channel goal — injected into LLM prompts., LLMRouter, _looks_like_json(), _parse_json() (+17 more)

### Community 1 - "Orchestration Utils"
Cohesion: 0.13
Nodes (24): utube — YouTube automation pipeline (5 videos/day)., main(), _pick_music(), produce_one(), _publish_at_for_slot(), Path, Top-level orchestrator. ALL configuration via config/*.yaml — no constants…, SVD (Stable Video Diffusion) router — config-driven. (+16 more)

### Community 2 - "TTS Generation"
Cohesion: 0.13
Nodes (11): Any, TTS router - config-driven multi-provider chain with full error visibility.…, Synthesize with Camb.ai's streaming TTS endpoint. The narration script is…, One streaming Camb.ai request. Returns raw audio bytes., Split text into <=limit-char pieces at sentence/phrase boundaries. Keeps…, Synthesize with Microsoft Edge TTS at neutral rate/pitch. Sending non-zero…, Google Translate TTS — robust no-key fallback. gTTS has no rate parameter;…, Try each provider in order; return audio bytes from the first success.… (+3 more)

### Community 3 - "Visual Generation"
Cohesion: 0.12
Nodes (9): ImageRouter, Image-generation router — config-driven, no hardcoded URLs/models/params., Stock B-roll router — Pexels + Pixabay, fully config-driven., StockRouter, Return MP4 bytes of an ~4s clip animated from the given image., VideoRouter, generate_visuals(), Path (+1 more)

### Community 4 - "Thumbnail Engine"
Cohesion: 0.27
Nodes (18): Image, ImageDraw, ImageFont, _apply_portrait_overlays(), _draw_badge(), _draw_thumbnail_text(), _ellipsize(), _enhance_background() (+10 more)

### Community 5 - "Themes Selection"
Cohesion: 0.24
Nodes (14): Resolve CLI flags into a concrete list of slot dicts the orchestrator will run., _select_slots(), all_themes(), _build_themes_for_lane(), _cli(), find_theme(), materialize_slot(), pick_themes() (+6 more)

### Community 6 - "Audio Processing"
Cohesion: 0.28
Nodes (14): _atempo_factor(), _estimate_segment_timings(), _ffmpeg_concat(), _postprocess_segment(), _probe_duration(), Path, TTS narration: continuous master MP3, audio bitrate/codec from config. The…, Convert '+12%' / '-10%' / '+0%' -> 1.12 / 0.90 / 1.00. `atempo` accepts… (+6 more)

### Community 7 - "Content Discovery"
Cohesion: 0.33
Nodes (13): _cfg(), _devto(), discover_for_niche(), _github_trending(), _hackernews(), Any, Discover trending topics. All limits + UA + timeouts read from pipeline.yaml >…, Return up to N candidate topics from this slot's configured sources. (+5 more)

### Community 8 - "Captions Generation"
Cohesion: 0.28
Nodes (12): _group_wrapped_lines(), _layout_cfg(), _plain_text_cues(), Any, Path, Whisper captions — model size, device, words-per-cue from pipeline.yaml >…, _srt_block(), transcribe_to_srt() (+4 more)

### Community 9 - "Ledger State"
Cohesion: 0.17
Nodes (4): Ledger, Path, Lightweight quota / topic-history ledger persisted to disk. Tracks: - recent…, Return theme ids used in the last `days`. Used to skip-pick repeats.

### Community 10 - "Video Assembly"
Cohesion: 0.31
Nodes (10): assemble_video(), _concat(), _final_mux(), Path, Final FFmpeg assembly: motion-only scenes + caption burn-in + music duck. Per…, Synthesize a moving gradient as a no-real-footage fallback. Uses ffmpeg's…, Loop and crop any input clip to exactly `dur` seconds at the target portrait…, _render_from_video() (+2 more)

### Community 11 - "Config Dictionaries"
Cohesion: 0.33
Nodes (5): dict, Config, Any, Plain dict with attribute access and a `get_path` helper for nested keys., `cfg.get_path('llm.providers.nvidia_nim.model')`

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_config()` connect `Core Scripting` to `Orchestration Utils`, `TTS Generation`, `Visual Generation`, `Thumbnail Engine`, `Themes Selection`, `Audio Processing`, `Content Discovery`, `Captions Generation`, `Video Assembly`, `Config Dictionaries`?**
  _High betweenness centrality (0.261) - this node is a cross-community bridge._
- **Why does `TTSRouter` connect `TTS Generation` to `Orchestration Utils`, `Audio Processing`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `produce_one()` connect `Orchestration Utils` to `Core Scripting`, `TTS Generation`, `Visual Generation`, `Content Discovery`, `Ledger State`, `Video Assembly`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Should `Core Scripting` be split into smaller, more focused modules?**
  _Cohesion score 0.13446969696969696 - nodes in this community are weakly interconnected._
- **Should `Orchestration Utils` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Should `TTS Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.12666666666666668 - nodes in this community are weakly interconnected._
- **Should `Visual Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.12318840579710146 - nodes in this community are weakly interconnected._