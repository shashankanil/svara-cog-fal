# Deploying svara-TTS from Hugging Face to fal.ai and Replicate

**Both fal.ai and Replicate support deploying custom TTS models with GPU inference, but they differ significantly in abstraction level, pricing, and developer workflow.** svara-TTS — a 3B-parameter Orpheus-style Llama model that generates speech via discrete SNAC audio tokens across 19 Indian languages — can run on either platform with some adaptation. fal.ai offers a Python-native class-based approach with tighter cold-start control and lower GPU costs (A100 at **$0.99/hr**), while Replicate uses Docker-container packaging via Cog with a larger ecosystem of pre-built models and broader GPU selection (A100 at **$5.04/hr**). The model's 3B parameter size in BF16 requires at least **16 GB VRAM**, making an A100 or L40S the minimum practical GPU for production.

---

## Understanding the svara-TTS architecture

Before packaging for either platform, you need to understand what you're deploying. svara-TTS v1 (`kenpath/svara-tts-v1`) is not a traditional vocoder-based TTS system. It's a **3B-parameter decoder-only transformer** built on the Llama 3.2 architecture, fine-tuned through a chain: `meta-llama/Llama-3.2-3B-Instruct` → Orpheus pretrained → Hindi fine-tune → LoRA adapter for 19 Indic languages. The model treats speech as a sequence of discrete audio tokens from the **SNAC codec** (`hubertsiuzdak/snac_24khz`), predicting them autoregressively just like an LLM predicts text tokens.

The inference pipeline has three stages: (1) format the input as a prompt like `"Hindi (Male): आज का दिन बहुत अच्छा है <happy>"` with special boundary tokens `[128259]` and `[128009, 128260]`, (2) the LLM generates SNAC tokens autoregressively (7 tokens per audio frame, starting at token ID 128266), and (3) these tokens are redistributed into three hierarchical SNAC layers and decoded into a **24 kHz mono WAV** waveform. The HF Space likely uses a Gradio-based `app.py` that loads the model via `transformers` or `unsloth`, while the production inference repo (`svara-tts-inference`) uses vLLM + FastAPI with Docker.

Key dependencies you'll need on either platform: `torch` with CUDA, `transformers`, `snac`, `numpy`, `soundfile`, and optionally `vllm` for high-throughput serving. System packages include `ffmpeg` and `libsndfile1`.

---

## Deploying to fal.ai: Python-native serverless functions

fal.ai uses a **class-based Python pattern** where you extend `fal.App`, define GPU requirements and dependencies as class attributes, load your model in `setup()`, and expose inference endpoints with `@fal.endpoint()`. The platform has a dedicated TTS deployment tutorial at `docs.fal.ai/serverless/tutorials/deploy-text-to-speech-model`.

**Note:** fal Serverless (Private Apps) is currently an **enterprise feature in private beta** — you must request access at `fal.ai/dashboard/serverless-get-started`.

### Installation and authentication

```bash
pip install fal
fal auth login          # Opens browser for OAuth
# Or set API key directly:
export FAL_KEY="your-api-key-here"
```

### Complete fal function for svara-TTS

```python
import fal
from fal.container import ContainerImage
from fal.toolkit import File
from pydantic import BaseModel, Field
from typing import Literal

# Custom Docker image for system dependencies
dockerfile_str = """
FROM python:3.11
RUN apt-get update && apt-get install -y ffmpeg libsndfile1
"""
custom_image = ContainerImage.from_dockerfile_str(dockerfile_str)

class SvaraInput(BaseModel):
    text: str = Field(
        description="Text to convert to speech",
        examples=["आज का दिन बहुत अच्छा है"],
    )
    language: Literal[
        "Hindi (Male)", "Hindi (Female)", "Tamil (Male)", "Tamil (Female)",
        "Bengali (Male)", "Bengali (Female)", "Telugu (Male)", "Telugu (Female)",
        "Kannada (Male)", "Kannada (Female)", "Indian English (Male)", "Indian English (Female)",
        # ... add all 38 voice profiles
    ] = Field(default="Hindi (Female)", description="Speaker voice profile")
    emotion: Literal["<happy>", "<sad>", "<anger>", "<fear>", "<clear>"] = Field(
        default="<clear>", description="Emotion/style tag"
    )

class SvaraOutput(BaseModel):
    audio: File = Field(description="Generated speech audio (WAV, 24kHz)")

class SvaraTTS(fal.App):
    image = custom_image
    machine_type = "GPU-A100"       # 40GB VRAM, sufficient for 3B params in BF16
    keep_alive = 300                # Keep warm for 5 minutes after last request
    max_concurrency = 2
    app_name = "svara-tts"
    requirements = [
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "transformers==4.42.4",
        "accelerate==0.33.0",
        "snac==1.2.0",
        "soundfile==0.12.1",
        "numpy==1.26.4",
        "hf-transfer==0.1.9",
    ]

    def setup(self):
        import os, torch
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC

        # Load the LLM (svara-tts-v1 with merged LoRA weights)
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained("kenpath/svara-tts-v1")
        self.model = AutoModelForCausalLM.from_pretrained(
            "kenpath/svara-tts-v1",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

        # Load SNAC decoder
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)
        self.warmup()

    def warmup(self):
        """Run a dummy inference to warm caches."""
        import torch
        prompt = "Hindi (Female): नमस्ते <clear>"
        self._generate_audio(prompt, max_new_tokens=100)

    def _redistribute_codes(self, code_list):
        import torch
        layer_1, layer_2, layer_3 = [], [], []
        num_frames = (len(code_list) + 1) // 7
        for i in range(num_frames):
            base = 7 * i
            if base + 6 >= len(code_list):
                break
            layer_1.append(code_list[base])
            layer_2.append(code_list[base + 1] - 4096)
            layer_3.append(code_list[base + 2] - (2 * 4096))
            layer_3.append(code_list[base + 3] - (3 * 4096))
            layer_2.append(code_list[base + 4] - (4 * 4096))
            layer_3.append(code_list[base + 5] - (5 * 4096))
            layer_3.append(code_list[base + 6] - (6 * 4096))
        codes = [
            torch.tensor(layer_1).unsqueeze(0).to(self.device),
            torch.tensor(layer_2).unsqueeze(0).to(self.device),
            torch.tensor(layer_3).unsqueeze(0).to(self.device),
        ]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat.squeeze().cpu().numpy()

    def _generate_audio(self, prompt_text, max_new_tokens=2000):
        import torch
        start_token = torch.tensor([[128259]], dtype=torch.int64, device=self.device)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64, device=self.device)
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        with torch.inference_mode():
            output = self.model.generate(
                modified_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )

        # Extract audio tokens (IDs >= 128266)
        generated = output[0][modified_ids.shape[1]:]
        audio_tokens = [t.item() - 128266 for t in generated if t.item() >= 128266]
        return self._redistribute_codes(audio_tokens)

    @fal.endpoint("/")
    def run(self, request: SvaraInput) -> SvaraOutput:
        import tempfile, soundfile as sf
        prompt = f"{request.language}: {request.text} {request.emotion}"
        audio_np = self._generate_audio(prompt)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, 24000)
            return SvaraOutput(
                audio=File.from_path(f.name, content_type="audio/wav", repository="cdn")
            )
```

### Deploy commands

```bash
# Test locally (creates ephemeral deployment)
fal run svara_tts.py::SvaraTTS

# Deploy permanently
fal deploy svara_tts.py::SvaraTTS --auth private

# Call the deployed endpoint
curl https://fal.run/your-username/svara-tts \
  -H 'Authorization: Key $FAL_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"text": "नमस्ते दुनिया", "language": "Hindi (Female)", "emotion": "<happy>"}'
```

### fal.ai GPU pricing and cold starts

fal.ai bills **per GPU-second** of actual compute time. For svara-TTS, the relevant tiers are:

- **GPU-A100** (40 GB VRAM): **$0.99/hr** ($0.0003/sec) — best value for a 3B model
- **GPU-H100** (80 GB VRAM): **$1.89/hr** ($0.0005/sec) — faster generation
- **GPU-A6000** (48 GB VRAM): available for lighter workloads

Cold starts on fal.ai are typically **5–10 seconds** for custom deployments. The `keep_alive` attribute (set in seconds) keeps runners warm between requests. Setting `min_concurrency = 1` eliminates cold starts entirely but incurs continuous cost. The persistent `/data` mount caches downloaded model weights across runner restarts, so subsequent cold boots only reload from local storage.

**Key docs:**

- Quick start: `docs.fal.ai/serverless/getting-started/quick-start`
- TTS tutorial: `docs.fal.ai/serverless/tutorials/deploy-text-to-speech-model`
- Custom containers: `docs.fal.ai/serverless/development/use-custom-container-image`
- Machine types: `docs.fal.ai/serverless/deployment-operations/machine-types`
- Pricing: `fal.ai/pricing`

---

## Deploying to Replicate: Docker containers via Cog

Replicate uses **Cog**, an open-source tool that packages ML models into standardized Docker containers with auto-generated APIs. You define your environment in `cog.yaml` and your prediction logic in `predict.py`, then push the container to Replicate's registry.

### Install Cog and initialize

```bash
# Install Cog
sudo curl -o /usr/local/bin/cog -L \
  https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Initialize project
mkdir svara-tts-replicate && cd svara-tts-replicate
cog init
```

### cog.yaml for svara-TTS

```yaml
build:
  gpu: true
  python_version: "3.11"
  system_packages:
    - "ffmpeg"
    - "libsndfile1"
    - "git"
  python_packages:
    - "torch==2.3.1"
    - "torchaudio==2.3.1"
    - "transformers==4.42.4"
    - "accelerate==0.33.0"
    - "snac==1.2.0"
    - "soundfile==0.12.1"
    - "numpy==1.26.4"
    - "huggingface_hub==0.23.0"
    - "sentencepiece==0.2.0"
    - "protobuf==5.27.3"
  run:
    - curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && chmod +x /usr/local/bin/pget
predict: "predict.py:Predictor"
image: "r8.im/your-username/svara-tts"
```

### predict.py for svara-TTS

```python
import os
import tempfile
import numpy as np
import torch
import soundfile as sf
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load svara-TTS model and SNAC decoder into GPU memory."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC

        self.device = "cuda"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("kenpath/svara-tts-v1")
        self.model = AutoModelForCausalLM.from_pretrained(
            "kenpath/svara-tts-v1",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

        # Load SNAC audio decoder
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)
        print("svara-TTS loaded successfully.")

    def _redistribute_codes(self, code_list):
        """Convert flat SNAC token list into 3 hierarchical layers and decode."""
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list) + 1) // 7):
            base = 7 * i
            if base + 6 >= len(code_list):
                break
            layer_1.append(code_list[base])
            layer_2.append(code_list[base + 1] - 4096)
            layer_3.append(code_list[base + 2] - 2 * 4096)
            layer_3.append(code_list[base + 3] - 3 * 4096)
            layer_2.append(code_list[base + 4] - 4 * 4096)
            layer_3.append(code_list[base + 5] - 5 * 4096)
            layer_3.append(code_list[base + 6] - 6 * 4096)
        codes = [
            torch.tensor(layer_1).unsqueeze(0).to(self.device),
            torch.tensor(layer_2).unsqueeze(0).to(self.device),
            torch.tensor(layer_3).unsqueeze(0).to(self.device),
        ]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat.squeeze().cpu().numpy()

    def predict(
        self,
        text: str = Input(
            description="Text to convert to speech",
            default="नमस्ते, यह एक परीक्षण है।",
        ),
        language: str = Input(
            description="Speaker profile: 'Language (Gender)' format",
            default="Hindi (Female)",
            choices=[
                "Hindi (Male)", "Hindi (Female)",
                "Tamil (Male)", "Tamil (Female)",
                "Bengali (Male)", "Bengali (Female)",
                "Telugu (Male)", "Telugu (Female)",
                "Kannada (Male)", "Kannada (Female)",
                "Indian English (Male)", "Indian English (Female)",
                # Add remaining profiles...
            ],
        ),
        emotion: str = Input(
            description="Emotion/style tag",
            default="<clear>",
            choices=["<happy>", "<sad>", "<anger>", "<fear>", "<clear>"],
        ),
        temperature: float = Input(
            description="Sampling temperature",
            default=0.7, ge=0.1, le=1.5,
        ),
        max_tokens: int = Input(
            description="Maximum SNAC tokens to generate (controls max audio length)",
            default=2000, ge=100, le=4000,
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=-1,
        ),
    ) -> Path:
        """Generate speech from text using svara-TTS."""
        if seed >= 0:
            torch.manual_seed(seed)

        # Build prompt in Orpheus format
        prompt_text = f"{language}: {text} {emotion}"
        start_token = torch.tensor([[128259]], dtype=torch.int64, device=self.device)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64, device=self.device)
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        # Generate SNAC audio tokens
        with torch.inference_mode():
            output = self.model.generate(
                modified_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )

        # Extract and decode audio tokens
        generated = output[0][modified_ids.shape[1]:]
        audio_tokens = [t.item() - 128266 for t in generated if t.item() >= 128266]
        audio_np = self._redistribute_codes(audio_tokens)

        # Write output WAV
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(output_path), audio_np, samplerate=24000)
        return output_path
```

### Build, test, and push

```bash
# 1. Test locally
cog predict -i text="Hello world" -i language="Indian English (Female)" -i emotion="<happy>"

# 2. Or start local server
cog serve -p 8080
curl http://localhost:8080/predictions -X POST \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "नमस्ते", "language": "Hindi (Female)"}}'

# 3. Create model on Replicate (visit https://replicate.com/create)

# 4. Login and push
cog login
cog push r8.im/your-username/svara-tts

# 5. Run via API
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Prefer: wait" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "आज मौसम बहुत अच्छा है", "language": "Hindi (Male)"}}' \
  https://api.replicate.com/v1/models/your-username/svara-tts/predictions
```

### Replicate GPU pricing

Replicate bills **per second** of compute. For a 3B-parameter TTS model, the practical options are:

| GPU               | VRAM  | Cost/hr   | Best for                                     |
| ----------------- | ----- | --------- | -------------------------------------------- |
| Nvidia T4         | 16 GB | **$0.81** | Development/testing only (tight for 3B BF16) |
| Nvidia L40S       | 48 GB | **$3.51** | Recommended production choice                |
| Nvidia A100 80 GB | 80 GB | **$5.04** | High throughput, largest batches             |
| Nvidia H100       | 80 GB | **$5.49** | Fastest generation speed                     |

Cold starts on Replicate can take **30 seconds to several minutes** for large models, since the entire Docker image must be fetched and `setup()` must run. Mitigate this by using **Deployments** with `min_instances >= 1` to keep instances warm, or by baking model weights into the Docker image at build time rather than downloading in `setup()`.

**Key docs:**

- Push a model: `replicate.com/docs/guides/build/push-a-model`
- Cog reference: `github.com/replicate/cog`
- Model best practices: `replicate.com/docs/guides/build/model-best-practices`
- Pricing: `replicate.com/pricing`
- Deployments: `replicate.com/docs/topics/deployments`

---

## fal.ai vs Replicate for TTS deployment

The two platforms target overlapping but distinct use cases. Here's how they compare specifically for deploying an audio generation model like svara-TTS.

### Cost per inference favors fal.ai significantly

For a 3B TTS model on an A100, fal.ai charges **$0.99/hr** versus Replicate's **$5.04/hr** — roughly a **5× cost difference** on equivalent hardware. If a typical svara-TTS inference takes 3 seconds, that's **$0.0008 per request** on fal.ai versus **$0.0042 per request** on Replicate. At 100,000 requests/month, this translates to ~$80 on fal.ai versus ~$420 on Replicate. fal.ai's H100 at $1.89/hr is still cheaper than Replicate's L40S at $3.51/hr, despite being a faster GPU.

### Developer experience differs in philosophy

fal.ai uses a **Python-native approach** — your entire deployment is a single `.py` file with a class definition. Dependencies are specified as a list of pip packages right in the class. This feels natural for Python developers and requires no Docker knowledge for simple deployments (though custom `ContainerImage` is available for system deps). The downside: fal Serverless is currently in **private beta for enterprise customers**, which limits accessibility.

Replicate uses a **Docker-first approach** via Cog. You maintain two files (`cog.yaml` + `predict.py`), and Cog builds a full Docker container. This is more explicit and portable — you can run `cog predict` locally with the exact same environment as production. The trade-off is heavier tooling setup (Docker required locally). Replicate's ecosystem is more mature, with thousands of public models and a community marketplace.

### Cold starts and latency

fal.ai reports cold starts of **5–10 seconds** for custom models, with `keep_alive` (in seconds) and `min_concurrency` to keep runners warm. The `/data` persistent mount means model weights survive across cold boots, so restarts only need to reload from local disk.

Replicate cold starts can reach **30 seconds to several minutes** for large models because the full container image must be pulled. Replicate mitigates this with Deployments (always-on instances) and by encouraging weight-baking into the Docker image. For models that qualify, Replicate has achieved sub-second cold boots for fine-tunes, but custom models typically face longer waits.

For production TTS workloads where latency matters, fal.ai's faster cold starts and granular `keep_alive` control provide a meaningful advantage.

### Platform maturity and ecosystem

Replicate has the larger ecosystem — thousands of public models, a prediction marketplace, webhook support, and recently joined Cloudflare (acquired in 2025). It has stronger community tooling, more tutorials, and broader GPU selection including multi-GPU configurations up to 8× H100. Replicate also handles output file hosting automatically (files available via HTTPS URLs for 1 hour).

fal.ai is more focused on media generation (images, video, audio) and has been growing rapidly in the AI inference space. Its built-in CDN upload via `File.from_path(..., repository="cdn")` handles audio output hosting natively. fal.ai's `fal.toolkit` provides specialized utilities like `Audio`, `Image`, and `Video` types purpose-built for media workloads.

### Summary comparison table

| Factor                | fal.ai                  | Replicate                              |
| --------------------- | ----------------------- | -------------------------------------- |
| **A100 cost**         | $0.99/hr                | $5.04/hr                               |
| **H100 cost**         | $1.89/hr                | $5.49/hr                               |
| **Cold start**        | 5–10 sec                | 30 sec–several min                     |
| **Packaging**         | Python class + pip list | cog.yaml + predict.py (Docker)         |
| **Local testing**     | `fal run` (remote)      | `cog predict` (local Docker)           |
| **Custom Docker**     | Yes (ContainerImage)    | Yes (native)                           |
| **TTS-specific docs** | Yes (Kokoro tutorial)   | Community examples (cog-dia, cog-xtts) |
| **Access**            | Enterprise private beta | Open to all                            |
| **Output hosting**    | Built-in CDN            | Auto-hosted URLs (1 hr expiry)         |
| **Ecosystem**         | Growing, media-focused  | Mature, broad marketplace              |

---

## Conclusion

For svara-TTS specifically — a 3B-parameter LLM-based model with SNAC decoding — **fal.ai offers better economics and lower latency** if you can get enterprise access. The **$0.99/hr A100 pricing** is hard to beat, and the Python-native workflow with built-in TTS tutorials makes deployment straightforward. The main barrier is the private beta access requirement.

**Replicate is the safer, more accessible choice** with a proven ecosystem, portable Docker-based packaging, and no access gates. The cost premium is substantial (~5× on equivalent GPUs), but Replicate's local testing via `cog predict`, its marketplace visibility, and Cloudflare integration may justify it depending on your distribution strategy. For either platform, the critical optimization is keeping instances warm — a 3B model reload on cold start adds significant latency that will degrade user experience for a real-time TTS application. Set `keep_alive = 300` on fal.ai or `min_instances = 1` on Replicate Deployments to avoid this in production.
