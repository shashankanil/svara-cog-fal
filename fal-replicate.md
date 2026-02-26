# Deploying Svara-TTS to fal.ai and Replicate

**Svara-TTS can be deployed as a serverless GPU API on both fal.ai and Replicate, though each platform requires a fundamentally different packaging approach.** The model is a 3B-parameter Llama-based TTS system that generates discrete SNAC audio codec tokens—not a conventional TTS pipeline—so deployment involves orchestrating both an LLM inference engine and an audio decoder. Replicate offers a more mature, self-serve workflow with Cog and per-second billing, while fal.ai provides a more Pythonic developer experience but currently gates its Serverless product behind enterprise access. Below are concrete, production-ready deployment guides for both platforms, tailored to the svara-tts architecture.

## Understanding svara-tts before packaging it

The HF Space at `kenpath/svara-tts` runs a **three-tier Docker stack**: a FastAPI API server on port 8080, a vLLM server on port 8000 serving `kenpath/svara-tts-v1` (the LLM), and a SNAC neural audio codec decoder (`hubertsiuzdak/snac_24khz`). The model descends from `meta-llama/Llama-3.2-3B-Instruct` through the Orpheus TTS lineage, generating **7 discrete audio tokens per frame** across 3 hierarchical SNAC codebook layers that are decoded into **24 kHz mono PCM audio**. The full inference repo lives at `github.com/Kenpath/svara-tts-inference` and includes an orchestrator, SNAC decoder module, voice configuration for 38 voice profiles across 19 languages, and supervisord for process management.

For deployment to fal.ai or Replicate, you need to collapse this multi-process architecture into a single container that loads both the LLM and the SNAC decoder, runs inference via the `transformers` library (or vLLM as a library), and returns audio. The key dependencies are **vLLM** (or transformers with `generate()`), **SNAC** (`snac` package), **PyTorch with CUDA**, and **ffmpeg** for audio format conversion.

## Deploying to fal.ai with the App pattern

fal.ai uses a class-based `fal.App` pattern. You define a Python class with a `setup()` method for model loading and `@fal.endpoint` decorated methods for inference. Audio output uses `fal.toolkit.File` which auto-uploads to fal's CDN.

**Install and authenticate:**

```bash
pip install fal
fal auth login
```

**Create `svara_tts_fal.py`:**

```python
import fal
from fal.container import ContainerImage
from pydantic import BaseModel, Field
from fal.toolkit import File
from typing import Literal, Optional

# Custom container for system deps (espeak, ffmpeg, etc.)
DOCKER_STRING = """
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    vllm>=0.6.0 \
    snac \
    transformers>=4.44.0 \
    soundfile \
    numpy \
    scipy \
    hf-transfer

# fal runtime deps (MUST be installed last)
RUN pip install --no-cache-dir \
    boto3==1.35.74 \
    protobuf==4.25.1 \
    pydantic==2.10.6
"""

class TTSInput(BaseModel):
    text: str = Field(
        description="Text to synthesize into speech.",
        examples=["नमस्ते, यह एक परीक्षण है।"],
    )
    voice: str = Field(
        default="hi_female",
        description="Voice ID: {lang_code}_{gender}, e.g. hi_male, en_female, ta_male.",
    )
    emotion: Optional[Literal["happy", "sad", "anger", "fear", "clear"]] = Field(
        default=None,
        description="Optional emotion tag to apply to the speech.",
    )

class TTSOutput(BaseModel):
    audio: File = Field(description="Generated audio file (WAV, 24kHz).")

class SvaraTTS(fal.App):
    kind = "container"
    image = ContainerImage.from_dockerfile_str(DOCKER_STRING)
    machine_type = "GPU-A100"      # 40GB VRAM, sufficient for 3B model
    keep_alive = 600               # 10 min warm after last request
    min_concurrency = 0            # Scale to zero when idle
    max_concurrency = 5
    app_name = "svara-tts"

    def setup(self):
        """Load both the LLM and SNAC decoder once at startup."""
        import os, torch
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC

        self.device = "cuda"
        model_id = "kenpath/svara-tts-v1"

        # Load tokenizer and LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

        # Load SNAC decoder
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

        # Key token IDs for the Orpheus protocol
        self.tokenizer_length = 128256
        self.start_of_header = 128259
        self.end_tokens = [128009, 128260, 128261, 128257]
        self.end_of_speech = 128258
        self.audio_start = 128266

        # Warmup inference
        self._synthesize("warmup", "en_female", None)
        print("Svara-TTS loaded and warmed up.")

    def _build_prompt(self, text: str, voice: str, emotion: str = None):
        """Construct the Orpheus-style token prompt."""
        import torch
        if emotion:
            text = f"{text}<{emotion}>"

        # Map voice ID to speaker string
        lang_map = {
            "hi": "Hindi", "en": "English", "bn": "Bengali", "ta": "Tamil",
            "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
            "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
            "ur": "Urdu", "sd": "Sindhi", "ne": "Nepali", "si": "Sinhala",
            "sa": "Sanskrit", "ks": "Kashmiri", "doi": "Dogri",
        }
        parts = voice.split("_")
        lang_code, gender = parts[0], parts[1].capitalize() if len(parts) > 1 else "Female"
        speaker = f"{lang_map.get(lang_code, 'Hindi')} ({gender})"

        prompt = f"{speaker}: {text}"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_ids = [self.start_of_header] + input_ids + self.end_tokens
        return torch.tensor([all_ids], dtype=torch.long).to(self.device)

    def _decode_snac_tokens(self, token_ids):
        """Decode Orpheus audio tokens to waveform via SNAC."""
        import torch
        codes = [t - self.audio_start for t in token_ids if t >= self.audio_start]
        codes = [c for c in codes if c < 7 * 4096]  # filter valid

        if len(codes) < 7:
            return None

        n_frames = len(codes) // 7
        layer_1, layer_2, layer_3 = [], [], []
        for i in range(n_frames):
            base = 7 * i
            layer_1.append(codes[base])
            layer_2.append(codes[base + 1] - 4096)
            layer_3.append(codes[base + 2] - 2 * 4096)
            layer_3.append(codes[base + 3] - 3 * 4096)
            layer_2.append(codes[base + 4] - 4 * 4096)
            layer_3.append(codes[base + 5] - 5 * 4096)
            layer_3.append(codes[base + 6] - 6 * 4096)

        codes_tensor = [
            torch.tensor(layer_1, device=self.device).unsqueeze(0),
            torch.tensor(layer_2, device=self.device).unsqueeze(0),
            torch.tensor(layer_3, device=self.device).unsqueeze(0),
        ]
        with torch.no_grad():
            audio = self.snac.decode(codes_tensor)
        return audio.squeeze().cpu().numpy()

    def _synthesize(self, text, voice, emotion):
        """Full synthesis pipeline."""
        import torch
        input_ids = self._build_prompt(text, voice, emotion)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=self.end_of_speech,
            )
        generated = output[0][input_ids.shape[1]:].tolist()
        return self._decode_snac_tokens(generated)

    @fal.endpoint("/")
    def run(self, input: TTSInput) -> TTSOutput:
        import tempfile, soundfile as sf
        audio = self._synthesize(input.text, input.voice, input.emotion)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 24000)
            return TTSOutput(
                audio=File.from_path(f.name, content_type="audio/wav", repository="cdn")
            )
```

**Deploy and test:**

```bash
# Development (temporary URL)
fal run svara_tts_fal.py::SvaraTTS --auth public

# Production (permanent URL)
fal deploy svara_tts_fal.py::SvaraTTS --auth private

# Call from Python
import fal_client
result = fal_client.subscribe(
    "your-username/svara-tts",
    arguments={"text": "नमस्ते दुनिया", "voice": "hi_female", "emotion": "happy"},
)
print(result["audio"]["url"])  # CDN URL to WAV file
```

**fal.ai GPU pricing** starts at **$0.99/hr for A100** and **$1.89/hr for H100**. For a 3B-param model like svara-tts, an A100 (40GB VRAM) is sufficient. Note that fal Serverless is currently an **enterprise feature**—you must request access at their Serverless Get Started page.

## Deploying to Replicate with Cog

Replicate uses **Cog**, an open-source container framework that wraps your model in a standardized Docker image with a `cog.yaml` config and a `predict.py` predictor class. Audio output is returned as a `cog.Path` object that Replicate automatically serves as a downloadable URL.

**Install Cog:**

```bash
# macOS
brew install cog

# Linux
sh <(curl -fsSL https://cog.run/install.sh)
```

**Create `cog.yaml`:**

```yaml
build:
  gpu: true
  python_version: "3.11"
  python_packages:
    - "torch==2.4.0"
    - "transformers>=4.44.0"
    - "snac"
    - "soundfile>=0.12"
    - "numpy>=1.26"
    - "scipy>=1.11"
    - "hf-transfer>=0.1.6"
    - "accelerate>=0.33"
  system_packages:
    - "ffmpeg"
    - "libsndfile1-dev"
  run:
    - curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_linux_x86_64" && chmod +x /usr/local/bin/pget

image: "r8.im/your-username/svara-tts"
predict: "predict.py:Predictor"
```

**Create `predict.py`:**

```python
import os
import time
import torch
import numpy as np
from cog import BasePredictor, Input, Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the LLM and SNAC decoder. Runs once on cold start."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC

        model_id = "kenpath/svara-tts-v1"
        self.device = "cuda"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Loading LLM (3B params)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

        print("Loading SNAC decoder...")
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

        # Orpheus protocol constants
        self.start_of_header = 128259
        self.end_tokens = [128009, 128260, 128261, 128257]
        self.end_of_speech = 128258
        self.audio_start = 128266

        # Language map for voice ID resolution
        self.lang_map = {
            "hi": "Hindi", "en": "English", "bn": "Bengali", "ta": "Tamil",
            "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
            "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
            "ur": "Urdu", "sd": "Sindhi", "ne": "Nepali", "si": "Sinhala",
            "sa": "Sanskrit", "ks": "Kashmiri", "doi": "Dogri",
        }

        # Warmup to JIT-compile CUDA kernels
        print("Running warmup inference...")
        self._synthesize("warmup", "en_female", None)
        print("Setup complete.")

    def _build_prompt(self, text, voice, emotion):
        parts = voice.split("_")
        lang_code = parts[0]
        gender = parts[1].capitalize() if len(parts) > 1 else "Female"
        speaker = f"{self.lang_map.get(lang_code, 'Hindi')} ({gender})"

        if emotion and emotion != "none":
            text = f"{text}<{emotion}>"

        prompt = f"{speaker}: {text}"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_ids = [self.start_of_header] + input_ids + self.end_tokens
        return torch.tensor([all_ids], dtype=torch.long).to(self.device)

    def _decode_snac_tokens(self, token_ids):
        codes = [t - self.audio_start for t in token_ids if t >= self.audio_start]
        codes = [c for c in codes if c < 7 * 4096]
        if len(codes) < 7:
            return np.zeros(24000, dtype=np.float32)  # 1s silence fallback

        n_frames = len(codes) // 7
        layer_1, layer_2, layer_3 = [], [], []
        for i in range(n_frames):
            b = 7 * i
            layer_1.append(codes[b])
            layer_2.append(codes[b + 1] - 4096)
            layer_3.append(codes[b + 2] - 2 * 4096)
            layer_3.append(codes[b + 3] - 3 * 4096)
            layer_2.append(codes[b + 4] - 4 * 4096)
            layer_3.append(codes[b + 5] - 5 * 4096)
            layer_3.append(codes[b + 6] - 6 * 4096)

        codes_tensor = [
            torch.tensor(layer_1, device=self.device).unsqueeze(0),
            torch.tensor(layer_2, device=self.device).unsqueeze(0),
            torch.tensor(layer_3, device=self.device).unsqueeze(0),
        ]
        with torch.no_grad():
            audio = self.snac.decode(codes_tensor)
        return audio.squeeze().cpu().numpy()

    def _synthesize(self, text, voice, emotion):
        input_ids = self._build_prompt(text, voice, emotion)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=self.end_of_speech,
            )
        generated = output[0][input_ids.shape[1]:].tolist()
        return self._decode_snac_tokens(generated)

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize into speech. Supports 19 languages.",
            default="नमस्ते, यह एक परीक्षण है।",
        ),
        voice: str = Input(
            description="Voice ID as {lang}_{gender}: hi_male, en_female, ta_male, etc.",
            default="hi_female",
        ),
        emotion: str = Input(
            description="Emotion tag to apply.",
            default="none",
            choices=["none", "happy", "sad", "anger", "fear", "clear"],
        ),
        seed: int = Input(
            description="Random seed for reproducibility. -1 for random.",
            default=-1,
            ge=-1,
        ),
    ) -> Path:
        """Synthesize speech from text and return a WAV file."""
        import soundfile as sf

        if seed >= 0:
            torch.manual_seed(seed)

        start = time.time()
        audio = self._synthesize(text, voice, emotion if emotion != "none" else None)
        elapsed = time.time() - start
        print(f"Synthesis took {elapsed:.2f}s for {len(text)} chars")

        output_path = "/tmp/output.wav"
        sf.write(output_path, audio, 24000)
        return Path(output_path)
```

**Build, test, and deploy:**

```bash
# Test locally
cog predict -i text="Hello world" -i voice="en_female"

# Build Docker image
cog build --separate-weights

# Create model on replicate.com/create, then push
cog login
cog push r8.im/your-username/svara-tts

# Call from Python
import replicate
output = replicate.run(
    "your-username/svara-tts",
    input={"text": "नमस्ते दुनिया", "voice": "hi_female", "emotion": "happy"}
)
# output is a URL: https://replicate.delivery/.../output.wav
```

For a 3B-param model, select the **Nvidia L40S** GPU tier (**$3.51/hr**, 48GB VRAM) as the best price-to-performance option, or **A100 80GB** ($5.04/hr) if you need more headroom for batching.

## Cold start optimization matters most for TTS APIs

Cold starts are the dominant latency concern for serverless TTS. Both platforms follow the same core principle: **load everything in `setup()`** and keep endpoints lightweight. But several svara-tts-specific optimizations apply:

**Bake weights into the image.** The svara-tts-v1 model is ~6GB in FP16. Downloading at cold start adds 30–60 seconds. On Replicate, place weights in a `weights/` subdirectory and build with `cog build --separate-weights`—this creates a separate Docker layer for weights that caches across rebuilds. On fal.ai, weights downloaded to `/data` persist across runner restarts automatically.

**Use `hf-transfer` for downloads.** If you choose runtime downloads, setting `HF_HUB_ENABLE_HF_TRANSFER=1` with the `hf-transfer` package speeds up HuggingFace Hub downloads by **5–10x** through parallel chunked transfers. On Replicate, the `pget` tool provides similar acceleration.

**Run warmup inference in setup.** The first `torch` inference on a new GPU triggers CUDA kernel compilation. A single dummy synthesis call in `setup()` ensures all kernels are JIT-compiled before the first real request. This typically saves **2–5 seconds** on first-request latency.

**Keep runners warm.** On fal.ai, set `keep_alive=600` (10 minutes) and optionally `min_concurrency=1` to always have one warm instance. On Replicate, private models maintain dedicated hardware that stays warm. Public models scale to zero but benefit from Replicate's boot-time optimizations.

**Consider vLLM vs transformers.** The HF Space uses vLLM for inference, which provides continuous batching and PagedAttention for higher throughput. For single-request serverless workloads, vanilla `transformers` with `model.generate()` is simpler to package and has lower memory overhead. For high-throughput production use, integrate vLLM as a library (not a separate server) within the same process.

## Platform comparison favors Replicate for most TTS use cases

| Factor                  | fal.ai                           | Replicate                                |
| ----------------------- | -------------------------------- | ---------------------------------------- |
| **Setup complexity**    | Pythonic, class-based            | Cog framework, Docker-based              |
| **Self-serve access**   | Enterprise only (request access) | Fully self-serve                         |
| **GPU pricing (A100)**  | ~$0.99/hr                        | ~$5.04/hr (80GB)                         |
| **Billing model**       | Per-second, active only          | Per-second (public), always-on (private) |
| **Audio output**        | `fal.toolkit.File` → CDN URL     | `cog.Path` → delivery URL                |
| **Cold start control**  | `min_concurrency`, `keep_alive`  | Hardware stays warm (private)            |
| **Custom containers**   | Full Dockerfile support          | `cog.yaml` + system packages             |
| **Community/ecosystem** | Growing, fewer examples          | Large, many TTS models deployed          |
| **Streaming support**   | WebSocket via `@fal.realtime`    | `Iterator[Path]` output type             |

**Replicate is the better choice for most teams** deploying svara-tts today. Its self-serve access, battle-tested Cog packaging system, large ecosystem of deployed TTS models (providing reference implementations), and straightforward per-second billing make it the lower-friction option. The L40S at **$0.000975/sec** offers excellent value for a 3B-param model, and the `--separate-weights` build pattern keeps iteration cycles fast.

**fal.ai excels on raw cost** if you can get enterprise access—its A100 pricing at ~$0.99/hr is roughly **5x cheaper** than Replicate's equivalent tier. The pure-Python `fal.App` pattern also means no new toolchain to learn. For teams already using fal.ai or needing aggressive cost optimization at scale, it's the stronger choice.

## Conclusion

The core technical challenge in deploying svara-tts is collapsing its three-process architecture (FastAPI + vLLM + SNAC) into a single-process serverless function. Both platforms handle this well: load the LLM and SNAC decoder in `setup()`, run `model.generate()` to produce audio tokens, decode them through SNAC's 3-layer codebook, and return a 24kHz WAV file. The critical optimization insight is that **the LLM inference dominates latency**—the SNAC decode step is negligible—so GPU selection and model quantization (FP16 or even INT8) matter more than any other tuning. For immediate deployment, start with Replicate's L40S tier and the `predict.py` template above; for cost optimization at scale, pursue fal.ai enterprise access and leverage their significantly lower GPU rates.
