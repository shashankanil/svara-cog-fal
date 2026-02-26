# Deploying Svara-TTS to Replicate

## Overview

Svara-TTS (`kenpath/svara-tts-v1`) deployed as a Replicate model using Cog. Same inference pipeline as the fal.ai version — 3B Llama LLM generating SNAC audio tokens, decoded into 24kHz WAV.

---

## Files

| File                   | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| `replicate/cog.yaml`   | Cog build config — GPU, Python 3.11, deps, pget      |
| `replicate/predict.py` | Predictor class — model loading, inference, endpoint |

## Architecture

Same as fal deployment — Orpheus token protocol → `model.generate()` → SNAC 3-layer decode → WAV. See `FAL_DEPLOY.md` for the full architecture diagram.

## API Schema

### Input

| Field         | Type    | Default                       | Description                       |
| ------------- | ------- | ----------------------------- | --------------------------------- |
| `text`        | `str`   | `"नमस्ते, यह एक परीक्षण है।"` | Text to synthesize                |
| `voice`       | `str`   | `hi_female`                   | `{lang_code}_{gender}`            |
| `emotion`     | `str`   | `none`                        | `none/happy/sad/anger/fear/clear` |
| `temperature` | `float` | `0.7`                         | Sampling temperature (0.1–1.5)    |
| `max_tokens`  | `int`   | `2048`                        | Max SNAC tokens (100–4000)        |
| `seed`        | `int`   | `-1`                          | Reproducibility seed              |

### Output

Returns a `Path` to a WAV file (24kHz mono). Replicate auto-hosts it as a downloadable URL.

## Deploy Commands

```bash
# 1. Install Cog
brew install cog

# 2. Test locally (requires Docker + NVIDIA GPU)
cd replicate
cog predict -i text="Hello world" -i voice="en_female"

# 3. Create model on Replicate
#    Visit https://replicate.com/create
#    Name: svara-tts
#    Hardware: Nvidia L40S (recommended) or A100 80GB

# 4. Login and push
cog login
cog push r8.im/shashankanil/svara-tts

# 5. Run via API
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Prefer: wait" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "नमस्ते दुनिया", "voice": "hi_female", "emotion": "happy"}}' \
  https://api.replicate.com/v1/models/shashankanil/svara-tts/predictions
```

```python
# Python client
import replicate

output = replicate.run(
    "shashankanil/svara-tts",
    input={
        "text": "नमस्ते दुनिया",
        "voice": "hi_female",
        "emotion": "happy",
    },
)
# output is a URL: https://replicate.delivery/.../output.wav
```

## GPU Pricing

| GPU              | VRAM      | Cost/hr   | Best for         |
| ---------------- | --------- | --------- | ---------------- |
| Nvidia T4        | 16 GB     | $0.81     | Dev/testing only |
| **Nvidia L40S**  | **48 GB** | **$3.51** | **Recommended**  |
| Nvidia A100 80GB | 80 GB     | $5.04     | High throughput  |
| Nvidia H100      | 80 GB     | $5.49     | Fastest speed    |

## Notes

- **Cold starts:** 30s–several minutes. Use Deployments with `min_instances >= 1` for warm instances.
- **No access gate:** Unlike fal.ai, Replicate is fully self-serve.
- **Local testing:** Requires Docker. `cog predict` runs in the exact same container as production.
- **Weight caching:** Use `cog build --separate-weights` to cache model weights in a separate Docker layer.
