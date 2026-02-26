# Deploying Svara-TTS to fal.ai

## Overview

Svara-TTS (`kenpath/svara-tts-v1`) is deployed as a serverless GPU endpoint on fal.ai using the `fal.App` class pattern. The deployment collapses the model's three-process architecture (FastAPI + vLLM + SNAC) into a single-process serverless function.

**Status:** ⏳ Serverless access pending approval. Request submitted at [fal.ai/dashboard/serverless-get-started](https://fal.ai/dashboard/serverless-get-started).

---

## Files

| File               | Purpose                                                                            |
| ------------------ | ---------------------------------------------------------------------------------- |
| `svara_tts_fal.py` | Complete fal.App deployment — container config, model loading, inference, endpoint |

## Architecture

```
Request (text, voice, emotion)
    │
    ▼
┌─────────────────────────────────────────────┐
│  fal.App (SvaraTTS)  —  GPU-A100 (40GB)     │
│                                              │
│  setup():                                    │
│    ├── Load kenpath/svara-tts-v1 (3B LLM)   │
│    ├── Load hubertsiuzdak/snac_24khz         │
│    └── Warmup inference                      │
│                                              │
│  run():                                      │
│    ├── Build Orpheus prompt with tokens      │
│    │   [128259] + text_tokens + [128009...]  │
│    ├── model.generate() → SNAC audio tokens  │
│    ├── Redistribute into 3 SNAC layers       │
│    ├── snac.decode() → 24kHz waveform        │
│    └── Return WAV via fal CDN                │
└─────────────────────────────────────────────┘
    │
    ▼
CDN URL (audio/wav, 24kHz mono)
```

## Container Image

Based on `falai/base:3.11-12.1.0` (official fal base with Python 3.11 + CUDA 12.1):

- **System deps:** `ffmpeg`, `libsndfile1`, `curl`, `git`
- **Python deps:** `torch==2.6.0`, `transformers==4.51.3`, `accelerate==1.6.0`, `snac`, `soundfile`, `numpy`, `scipy`, `hf-transfer`, `sentencepiece`
- **fal runtime deps (installed last):** `boto3==1.35.74`, `protobuf==4.25.1`, `pydantic==2.10.6`

## Configuration

| Setting           | Value       | Rationale                                   |
| ----------------- | ----------- | ------------------------------------------- |
| `machine_type`    | `GPU-A100`  | 40GB VRAM, sufficient for 3B params in BF16 |
| `keep_alive`      | `300`       | 5 min warm after last request               |
| `min_concurrency` | `0`         | Scale to zero when idle                     |
| `max_concurrency` | `5`         | Up to 5 concurrent runners                  |
| `app_name`        | `svara-tts` | URL slug                                    |

## API Schema

### Input (`TTSInput`)

| Field         | Type    | Default     | Description                              |
| ------------- | ------- | ----------- | ---------------------------------------- |
| `text`        | `str`   | _required_  | Text to synthesize (19 Indian languages) |
| `voice`       | `str`   | `hi_female` | Voice ID: `{lang_code}_{gender}`         |
| `emotion`     | `str?`  | `null`      | `happy`, `sad`, `anger`, `fear`, `clear` |
| `temperature` | `float` | `0.7`       | Sampling temperature (0.1–1.5)           |
| `max_tokens`  | `int`   | `2048`      | Max SNAC tokens (100–4000)               |
| `seed`        | `int`   | `-1`        | Reproducibility seed (-1 = random)       |

**Supported languages:** Hindi, English, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, Sindhi, Nepali, Sinhala, Sanskrit, Kashmiri, Dogri

### Output (`TTSOutput`)

| Field   | Type   | Description                      |
| ------- | ------ | -------------------------------- |
| `audio` | `File` | WAV file on fal CDN (24kHz mono) |

## Deploy Commands

```bash
# 1. Install CLI
pipx install fal --python python3.11

# 2. Authenticate
fal auth login

# 3. Test (temporary URL, public)
fal run svara_tts_fal.py::SvaraTTS

# 4. Deploy (permanent URL, private)
fal deploy svara_tts_fal.py::SvaraTTS --auth private
```

## Calling the API

```bash
# cURL
curl https://fal.run/shashankanil/svara-tts \
  -H 'Authorization: Key $FAL_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"text": "नमस्ते दुनिया", "voice": "hi_female", "emotion": "happy"}'
```

```python
# Python
import fal_client

result = fal_client.subscribe(
    "shashankanil/svara-tts",
    arguments={
        "text": "नमस्ते दुनिया",
        "voice": "hi_female",
        "emotion": "happy",
    },
)
print(result["audio"]["url"])  # CDN URL to WAV
```

## Pricing

- **GPU-A100 (40GB):** $0.99/hr ($0.0003/sec)
- **GPU-H100 (80GB):** $1.89/hr ($0.0005/sec)
- At ~3s per inference: **~$0.0008/request** on A100

## Next Steps

Once fal Serverless access is approved:

1. Run `fal run svara_tts_fal.py::SvaraTTS` to test
2. Verify audio output quality
3. Deploy to production with `fal deploy`
4. Optionally set `min_concurrency=1` to eliminate cold starts
