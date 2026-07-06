# utube — automated YouTube Shorts (5 videos/day, $0)

Fully-automated YouTube Shorts pipeline that runs daily on GitHub Actions and produces
**5 vertical videos per day** across 5 niche lanes — script, voice, AI imagery, animated
clips, captions, thumbnail, and upload — all from free providers.

## Pipeline at a glance

```
discover  →  research  →  script  →  TTS  →  SDXL images
                                                 ↓
                                              SVD animation (4 of N scenes)
                                                 ↓
                                              stock B-roll fallback
                                                 ↓
                                          captions (Whisper)
                                                 ↓
                                          thumbnail (SDXL + PIL)
                                                 ↓
                                          FFmpeg assemble
                                                 ↓
                                          YouTube upload (publishAt)
```

## The 5 niche slots

| Slot id | Title | Voice | Publish (UTC) |
|---|---|---|---|
| `did_you_know`     | Did You Know? Tech Facts          | en-US-AvaMultilingualNeural | 04:00 |
| `tech_news`        | Tech News of the Day              | en-US-GuyNeural             | 09:00 |
| `ai_breakthrough`  | AI Breakthrough                   | en-US-AriaNeural            | 13:00 |
| `cybersecurity`    | Cybersecurity Bite                | en-US-DavisNeural           | 17:00 |
| `viral_science`    | Viral Science / Future Tech       | en-US-JennyNeural           | 21:00 |

Edit `config/niches.yaml` to retune sources, voices, palettes, schedule.

## Provider stack

| Component  | Primary                        | Fallback 1                  | Fallback 2 |
|---|---|---|---|
| LLM        | NVIDIA NIM `gpt-oss-120b`      | Cerebras `gpt-oss-120b`     | Groq / OpenRouter |
| Images     | NIM SDXL                       | Pollinations.ai (no key)    | HuggingFace FLUX |
| Video (SVD)| NIM Stable-Video-Diffusion     | HuggingFace SVD-XT-1.1      | — |
| TTS        | Camb.ai MARS Instruct          | edge-tts (no key)           | gTTS |
| Captions   | faster-whisper (CPU)           | —                           | — |
| Stock B-roll | Pexels API                   | Pixabay API                 | — |
| Upload     | YouTube Data API v3 OAuth      | —                           | — |

The pipeline self-heals: if a provider is rate-limited or down, the next one in the chain runs.

## Setup

### 1. Fork / clone this repo (recommended: keep it **public** for unlimited GitHub Actions minutes)

### 2. Get free API keys

| Provider | URL | Required? |
|---|---|---|
| NVIDIA NIM    | https://build.nvidia.com         | recommended |
| Cerebras      | https://cloud.cerebras.ai        | recommended (LLM fallback, 14,400 req/day) |
| Groq          | https://console.groq.com         | optional (LLM fallback) |
| Camb.ai       | https://studio.camb.ai           | recommended (primary TTS) |
| Pexels        | https://pexels.com/api           | recommended (B-roll) |
| Pixabay       | https://pixabay.com/api/docs     | optional (B-roll) |
| HuggingFace   | https://huggingface.co/settings/tokens | optional (image/video fallback) |
| OpenRouter    | https://openrouter.ai            | optional (LLM emergency fallback) |

You only need **one LLM provider** to start; more = better resilience.

### 3. YouTube OAuth

1. In Google Cloud Console: create a project, enable **YouTube Data API v3**.
2. Configure the OAuth consent screen (External; add yourself as a test user).
3. Create OAuth 2.0 **Desktop app** credentials, download the JSON.
4. Save the JSON as `scripts/client_secret.json` (this file is gitignored).
5. Run locally:

   ```bash
   pip install google-auth-oauthlib
   python scripts/generate_youtube_token.py
   ```

   Grant access in the browser. Three values are printed:
   ```
   YOUTUBE_CLIENT_ID=...
   YOUTUBE_CLIENT_SECRET=...
   YOUTUBE_REFRESH_TOKEN=...
   ```

### 4. Add GitHub Secrets

`Settings → Secrets and variables → Actions → New repository secret`

| Name | Required |
|---|---|
| `NVIDIA_NIM_API_KEY`     | ★ at least one LLM key |
| `CEREBRAS_API_KEY`       |   |
| `GROQ_API_KEY`           |   |
| `OPENROUTER_API_KEY`     |   |
| `CAMB_API_KEY`           | primary TTS narration |
| `HUGGINGFACE_API_KEY`    | optional |
| `PEXELS_API_KEY`         | recommended |
| `PIXABAY_API_KEY`        | optional |
| `YOUTUBE_CLIENT_ID`      | for upload |
| `YOUTUBE_CLIENT_SECRET`  | for upload |
| `YOUTUBE_REFRESH_TOKEN`  | for upload |

### 5. (Optional) Drop royalty-free music tracks

Place CC0 MP3s under:
```
assets/music/chill/
assets/music/corporate_upbeat/
assets/music/futuristic/
assets/music/tense/
assets/music/epic/
```
The orchestrator picks one randomly per niche. If the folder is empty, the video has narration only.

## Running

### A. Local — single video, no upload

```bash
pip install -r requirements.txt
sudo apt-get install ffmpeg fonts-dejavu-core   # or brew install ffmpeg
cp .env.example .env   # fill in the keys you have
set -a; . .env; set +a

# Build one Tech News video, no SVD, no upload
python -m pipeline.orchestrator --slot tech_news --no-upload --skip-svd
```

Outputs land in `runs/YYYY-MM-DD/<slot>/`:
- `*.mp4`              – final 1080×1920 Short
- `thumbnail.jpg`      – 1280×720
- `script.json`        – LLM script
- `result.json`        – stage summary
- `audio/narration.mp3`, `visuals/scene_*/…`

### B. CI — manual test (no upload)

GitHub → **Actions** tab → **Manual Test (single video, no upload)** → **Run workflow** → pick a slot. The MP4 is uploaded as a workflow artifact you can download.

### C. CI — daily batch (auto)

Once secrets are configured, the **Daily Batch** workflow runs every day at **00:00 UTC** and produces+uploads all 5 videos. YouTube schedules each one to go live at its niche's `schedule_utc` time.

You can also trigger it manually from the Actions tab.

## Project layout

```
.github/workflows/
  daily-batch.yml         # cron 00:00 UTC, all 5 slots, uploads to YouTube
  manual-test.yml         # workflow_dispatch, single slot, artifact only
config/
  niches.yaml             # 5 slot definitions + global video specs
  schedule.yaml           # cron + dedup window + per-stage timeouts
prompts/
  topic_select.txt        # LLM prompt to pick the day's topic
  research.txt            # LLM prompt to build a research brief
  script.txt              # LLM prompt to produce the structured script JSON
pipeline/
  orchestrator.py         # entrypoint: runs all slots or one
  ledger.py               # 30-day topic dedup + provider usage counters
  utils.py
  providers/
    llm.py                # NIM → Cerebras → Groq → OpenRouter
    image.py              # NIM SDXL → Pollinations → HF FLUX
    video.py              # NIM SVD → HF SVD-XT
    tts.py                # Camb.ai -> edge-tts -> gTTS
    stock.py              # Pexels → Pixabay
    youtube.py            # OAuth refresh-token uploader, publishAt scheduling
  stages/
    discover.py           # HN, Reddit, RSS, Wikipedia OTD, GitHub trending, Dev.to
    research.py           # topic selection + source enrichment
    script.py             # script JSON
    audio.py              # per-segment + master narration MP3
    visuals.py            # SDXL stills + SVD animation + stock fallback
    captions.py           # faster-whisper → SRT (3-word cues)
    thumbnail.py          # SDXL bg + PIL text overlay
    assemble.py           # FFmpeg concat + Ken Burns + caption burn + music duck
scripts/
  generate_youtube_token.py
runs/                     # per-day output dirs, kept on disk for review
ledger.json               # auto-committed by daily workflow
```

## Tuning knobs

| What | Where |
|---|---|
| Niches, voices, sources, palettes, schedule times | `config/niches.yaml` |
| Cron schedule, dedup window, per-stage timeouts   | `config/schedule.yaml` |
| Video resolution, fps, scene count, SVD count, captions size | `config/niches.yaml > video:` |
| Prompt wording / style | `prompts/*.txt` |
| Add a new niche | append a slot to `niches.yaml > slots:`, then add it to the `manual-test.yml` choice list |

## Cost & quota notes

- **GitHub Actions:** public repo = unlimited minutes; private = 2,000 min/mo (~12 daily runs/min × 30 days fits easily).
- **YouTube quota:** 1 upload + thumbnail ≈ 1,650 units. 5 uploads/day = 8,250 / 10,000 units. Comfortable.
- **Camb.ai:** set `CAMB_API_KEY` for the primary natural narration path. Without it, the pipeline falls back to edge-tts and gTTS.
- **NIM trial credits** deplete over weeks. Once they do, Pollinations + Cerebras + edge-tts keep the pipeline alive at zero cost.
- **YouTube channel limit:** ~100 uploads/day (undocumented). Won't matter at 5/day.

## Disclaimer

Videos in this pipeline are AI-assisted. The default script template encourages original
framing rather than copy-paste regurgitation, and all imagery is generated or sourced from
CC0 stock providers. Add an "AI-assisted" note in your channel description and disclose in
each video's description to stay aligned with YouTube's responsible-use guidance.
