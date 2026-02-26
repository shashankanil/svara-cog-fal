"""
Svara-TTS Replicate Predictor

Deploys kenpath/svara-tts-v1 (3B Llama-based TTS) as a Replicate model.
Generates 24kHz mono WAV audio from text across 19 Indian languages
using the Orpheus token protocol and SNAC 3-layer audio codec.
"""

import os
import time
import tempfile

import numpy as np
import torch
import soundfile as sf
from cog import BasePredictor, Input, Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Language code → display name mapping for all 19 supported languages
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


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the LLM and SNAC decoder into GPU memory. Runs once on cold start."""
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

        # Warmup to JIT-compile CUDA kernels
        print("Running warmup inference…")
        self._synthesize("warmup", "en_female", None)
        print("✓ Setup complete.")

    # ---- internal helpers ----

    def _build_prompt(self, text: str, voice: str, emotion: str | None = None):
        """Construct the Orpheus-style token prompt."""
        if emotion and emotion != "none":
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
        """Decode Orpheus audio tokens → waveform via SNAC 3-layer codebook."""
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
        generated = output[0][input_ids.shape[1]:].tolist()
        return self._decode_snac_tokens(generated)

    # ---- Cog predict endpoint ----

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize into speech. Supports 19 Indian languages.",
            default="नमस्ते, यह एक परीक्षण है।",
        ),
        voice: str = Input(
            description=(
                "Voice ID as {lang_code}_{gender}. "
                "Codes: hi, en, bn, ta, te, mr, gu, kn, ml, pa, or, as, ur, sd, ne, si, sa, ks, doi. "
                "Example: hi_female, en_male, ta_female."
            ),
            default="hi_female",
        ),
        emotion: str = Input(
            description="Emotion tag to apply to the speech.",
            default="none",
            choices=["none", "happy", "sad", "anger", "fear", "clear"],
        ),
        temperature: float = Input(
            description="Sampling temperature. Lower = more deterministic.",
            default=0.7,
            ge=0.1,
            le=1.5,
        ),
        max_tokens: int = Input(
            description="Maximum SNAC tokens to generate (controls max audio length).",
            default=2048,
            ge=100,
            le=4000,
        ),
        seed: int = Input(
            description="Random seed for reproducibility. -1 for random.",
            default=-1,
            ge=-1,
        ),
    ) -> Path:
        """Synthesize speech from text and return a WAV file."""
        if seed >= 0:
            torch.manual_seed(seed)

        start = time.time()
        audio = self._synthesize(
            text=text,
            voice=voice,
            emotion=emotion if emotion != "none" else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start
        print(f"Synthesis took {elapsed:.2f}s for {len(text)} chars")

        output_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(output_path), audio, 24000)
        return output_path
