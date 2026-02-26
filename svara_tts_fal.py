import fal
from fal.container import ContainerImage
from fal.toolkit import File
from pydantic import BaseModel, Field
from typing import Literal, Optional

# Custom container based on fal's official PyTorch + HuggingFace template
DOCKERFILE_STR = """
FROM falai/base:3.11-12.1.0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support + model dependencies
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    accelerate==1.6.0 \
    transformers==4.51.3 \
    snac \
    soundfile>=0.12 \
    numpy>=1.26 \
    scipy>=1.11 \
    hf-transfer>=0.1.6 \
    sentencepiece>=0.2.0 \
    --extra-index-url \
    https://download.pytorch.org/whl/cu124

# IMPORTANT: Install fal-required packages LAST to avoid version conflicts
RUN pip install --no-cache-dir \
    boto3==1.35.74 \
    protobuf==4.25.1 \
    pydantic==2.10.6

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV HF_HUB_ENABLE_HF_TRANSFER=1
"""

# --- Language & Voice Configuration ---

LANG_MAP = {
    "hi": "Hindi",
    "en": "Indian English",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "sd": "Sindhi",
    "ne": "Nepali",
    "si": "Sinhala",
    "sa": "Sanskrit",
    "ks": "Kashmiri",
    "doi": "Dogri",
}

VOICE_CHOICES = [
    f"{code}_{gender}"
    for code in LANG_MAP
    for gender in ("male", "female")
]

# --- Input / Output Schemas ---


class TTSInput(BaseModel):
    text: str = Field(
        description="Text to synthesize into speech. Supports 19 Indian languages.",
        examples=["नमस्ते, यह एक परीक्षण है।"],
    )
    voice: str = Field(
        default="hi_female",
        description=(
            "Voice ID as {lang_code}_{gender}. "
            "Available lang codes: " + ", ".join(LANG_MAP.keys()) + ". "
            "Example: hi_female, en_male, ta_female."
        ),
    )
    emotion: Optional[Literal["happy", "sad", "anger", "fear", "clear"]] = Field(
        default=None,
        description="Optional emotion tag to apply to the speech.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.5,
        description="Sampling temperature. Lower = more deterministic.",
    )
    max_tokens: int = Field(
        default=2048,
        ge=100,
        le=4000,
        description="Maximum SNAC tokens to generate (controls max audio length).",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility. -1 for random.",
    )


class TTSOutput(BaseModel):
    audio: File = Field(description="Generated audio file (WAV, 24 kHz mono).")


# --- fal.App ---


class SvaraTTS(fal.App):
    image = ContainerImage.from_dockerfile_str(DOCKERFILE_STR)
    machine_type = "GPU-A100"  # 40 GB VRAM — sufficient for 3B params in BF16
    keep_alive = 300  # Keep runner warm for 5 min after last request
    min_concurrency = 0  # Scale to zero when idle
    max_concurrency = 5
    app_name = "svara-tts"

    # ---- lifecycle ----

    def setup(self):
        """Load both the LLM and SNAC decoder once at cold start."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC

        self.device = "cuda"
        model_id = "kenpath/svara-tts-v1"

        print("Loading tokenizer…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Loading LLM (3B params)…")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

        print("Loading SNAC decoder…")
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

        # Orpheus protocol token constants
        self.start_of_header = 128259
        self.end_tokens = [128009, 128260, 128261, 128257]
        self.end_of_speech = 128258
        self.audio_start = 128266

        # Warmup inference to JIT-compile CUDA kernels
        print("Running warmup inference…")
        self._synthesize("warmup", "en_female", None)
        print("✓ Svara-TTS loaded and warmed up.")

    # ---- internal helpers ----

    def _build_prompt(self, text: str, voice: str, emotion: str | None = None):
        """Construct the Orpheus-style token prompt."""
        import torch

        if emotion:
            text = f"{text}<{emotion}>"

        parts = voice.split("_")
        lang_code = parts[0]
        gender = parts[1].capitalize() if len(parts) > 1 else "Female"
        speaker = f"{LANG_MAP.get(lang_code, 'Hindi')} ({gender})"

        prompt = f"{speaker}: {text}"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_ids = [self.start_of_header] + input_ids + self.end_tokens
        return torch.tensor([all_ids], dtype=torch.long).to(self.device)

    def _decode_snac_tokens(self, token_ids: list[int]):
        """Decode Orpheus audio tokens to waveform via SNAC 3-layer codebook."""
        import torch

        codes = [t - self.audio_start for t in token_ids if t >= self.audio_start]
        codes = [c for c in codes if c < 7 * 4096]  # filter valid range

        if len(codes) < 7:
            return None

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

    def _synthesize(
        self,
        text: str,
        voice: str,
        emotion: str | None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ):
        """Full synthesis pipeline: prompt → generate → decode."""
        import torch

        input_ids = self._build_prompt(text, voice, emotion)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.end_of_speech,
            )
        generated = output[0][input_ids.shape[1] :].tolist()
        return self._decode_snac_tokens(generated)

    # ---- endpoint ----

    @fal.endpoint("/")
    def run(self, request: TTSInput) -> TTSOutput:
        """Synthesize speech from text and return a WAV file on fal CDN."""
        import time
        import tempfile
        import torch
        import numpy as np
        import soundfile as sf

        if request.seed >= 0:
            torch.manual_seed(request.seed)

        start = time.time()
        audio = self._synthesize(
            text=request.text,
            voice=request.voice,
            emotion=request.emotion,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        elapsed = time.time() - start

        if audio is None:
            # Fallback: 1 second of silence if generation produced no valid tokens
            audio = np.zeros(24000, dtype=np.float32)

        print(f"Synthesis took {elapsed:.2f}s for {len(request.text)} chars")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 24000)
            return TTSOutput(
                audio=File.from_path(f.name, content_type="audio/wav", repository="cdn")
            )
