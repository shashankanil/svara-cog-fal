"""
Svara-TTS Replicate Predictor

Deploys kenpath/svara-tts-v1 (3B Llama-based TTS) as a Replicate model.
Generates 24kHz mono WAV audio from text across 19 Indian languages
using the official Svara-TTS inference pipeline.
"""

import os
import re
import time
import tempfile

import numpy as np
import torch
import soundfile as sf
from cog import BasePredictor, Input, Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

BOS_TOKEN = 128000
END_OF_TURN = 128009
AUDIO_TOKEN = 156939
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262

AUDIO_TOKENS_START = 128266
AUDIO_VOCAB_SIZE = 4096

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

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


def _extract_custom_token_numbers(text: str):
    """Extract custom token numbers from model output text."""
    for m in _TOKEN_RE.findall(text or ""):
        try:
            n = int(m)
            if n != 0:
                yield n
        except Exception:
            continue


def _raw_to_code_id(raw_num: int, good_idx: int) -> int:
    """Convert raw number to code id using band offset rule."""
    return raw_num - 10 - ((good_idx % 7) * 4096)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the LLM and SNAC decoder into GPU memory."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC

        self.device = "cuda"
        model_id = "kenpath/svara-tts-v1"

        print("Loading tokenizer…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading LLM (3B params)…")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded, vocab size: {self.model.config.vocab_size}")

        print("Loading SNAC decoder…")
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

        self.end_of_speech = END_OF_SPEECH

        print("Running warmup inference…")
        self._synthesize("hello", "en_female", None)
        print("✓ Setup complete.")

    def _build_prompt_string(self, text: str, voice: str, emotion: str | None = None) -> str:
        """Build prompt string using official Svara-TTS format."""
        if emotion and emotion != "none":
            text = f"{text}<{emotion}>"

        parts = voice.split("_")
        lang_code = parts[0]
        gender = parts[1].capitalize() if len(parts) > 1 else "Female"
        speaker = f"{LANG_MAP.get(lang_code, 'Hindi')} ({gender})"

        prompt = f"{speaker}: {text}"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        special_tokens = [
            BOS_TOKEN,
            START_OF_HUMAN,
            AUDIO_TOKEN,
        ] + prompt_tokens + [
            END_OF_HUMAN,
            END_OF_TURN,
            START_OF_AI,
            START_OF_SPEECH,
        ]

        return self.tokenizer.decode(special_tokens, skip_special_tokens=False)

    def _decode_audio_tokens(self, window: list) -> np.ndarray:
        """Decode a 28-code window into audio using SNAC."""
        if not window or len(window) < 7:
            return np.zeros(24000, dtype=np.float32)

        F = len(window) // 7
        frame = window[: F * 7]

        t = torch.tensor(frame, dtype=torch.int32, device=self.device)
        t = t.view(F, 7)

        codes_0 = t[:, 0].reshape(1, -1)
        codes_1 = t[:, [1, 4]].reshape(1, -1)
        codes_2 = t[:, [2, 3, 5, 6]].reshape(1, -1)

        if (
            torch.any((codes_0 < 0) | (codes_0 > 4096)) or
            torch.any((codes_1 < 0) | (codes_1 > 4096)) or
            torch.any((codes_2 < 0) | (codes_2 > 4096))
        ):
            return np.zeros(24000, dtype=np.float32)

        with torch.no_grad():
            audio = self.snac.decode([codes_0, codes_1, codes_2])
            audio = audio[:, :, 2048:4096]

        return audio.squeeze().cpu().numpy()

    def _synthesize(
        self,
        text: str,
        voice: str,
        emotion: str | None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Generate audio using text-based generation with token extraction."""
        prompt_str = self._build_prompt_string(text, voice, emotion)
        print(f"DEBUG: Prompt: {prompt_str[:200]}...")

        codes = []
        good = 0

        from transformers import AutoModelForCausalLM
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        with torch.no_grad():
            from transformers import AutoModelForCausalLM
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                eos_token_id=self.end_of_speech,
            )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        print(f"DEBUG: Generated text length: {len(generated_text)}")

        for raw in _extract_custom_token_numbers(generated_text):
            code = _raw_to_code_id(raw, good)
            if code > 0:
                codes.append(code)
                good += 1

        print(f"DEBUG: Extracted {len(codes)} audio codes")

        if len(codes) < 28:
            print("DEBUG: Not enough codes, returning silence")
            return np.zeros(24000, dtype=np.float32)

        return self._decode_audio_tokens(codes)

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
            description="Maximum SNAC tokens to generate.",
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
            np.random.seed(seed)

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
        print(f"Audio: shape={audio.shape}, min={audio.min():.3f}, max={audio.max():.3f}")

        output_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(output_path), audio, 24000)
        return output_path
