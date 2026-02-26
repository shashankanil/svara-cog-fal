"""
Svara-TTS Replicate Predictor

Deploys kenpath/svara-tts-v1 (3B Llama-based TTS) as a Replicate model.
Generates 24kHz mono WAV audio from text across 19+ Indian languages
using the Svara-TTS token protocol and SNAC 3-layer audio codec.

Reference: https://huggingface.co/spaces/kenpath/svara-tts (known-working Gradio demo)
"""

import os
import time
import tempfile

import numpy as np
import torch
import soundfile as sf
from cog import BasePredictor, Input, Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR = 24_000  # SNAC sample rate

# Language map: code -> label used in the prompt
LANG_MAP = {
    "hi": "Hindi",
    "en": "English",
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

# Emotion/style tags supported by the model
VALID_STYLES = {"neutral", "formal", "chat", "clear", "happy", "surprise", "sad", "fear", "anger", "disgust"}


# ---------------------------------------------------------------------------
# Token protocol helpers (matches Gradio demo exactly)
# ---------------------------------------------------------------------------
def parse_output(generated_ids: torch.Tensor) -> list[int]:
    """Extract SNAC codes from generated token IDs.

    1. Find the LAST occurrence of START_OF_SPEECH (128257)
    2. Crop everything after it
    3. Remove END_OF_SPEECH (128258) tokens
    4. Trim to multiple of 7
    5. Subtract 128266 from each token to get raw SNAC codes
    """
    token_to_find = 128257    # START_OF_SPEECH
    token_to_remove = 128258  # END_OF_SPEECH

    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        cropped_tensor = generated_ids[:, token_indices[1][-1] + 1:]
    else:
        cropped_tensor = generated_ids

    # Remove END_OF_SPEECH tokens
    row = cropped_tensor[0]
    row = row[row != token_to_remove]

    # Trim to multiple of 7 (7 codes per frame)
    trimmed_row = row[: (row.size(0) // 7) * 7]

    # Convert to raw SNAC codes
    code_list = [int(t.item()) - 128266 for t in trimmed_row]
    return code_list


def redistribute_codes(code_list: list[int], snac_model, device: str) -> np.ndarray:
    """Decode SNAC codes to audio waveform.

    Each frame is 7 codes mapped to 3 SNAC layers:
      codes[0]           -> layer 1 (coarsest, 1 code/frame)
      codes[1], codes[4] -> layer 2 (mid, 2 codes/frame)
      codes[2], codes[3], codes[5], codes[6] -> layer 3 (finest, 4 codes/frame)

    Band offsets are subtracted to bring each code back to [0, 4096):
      layer 1: codes[0] (no offset)
      layer 2: codes[1] - 4096, codes[4] - 4*4096
      layer 3: codes[2] - 2*4096, codes[3] - 3*4096, codes[5] - 5*4096, codes[6] - 6*4096
    """
    layer_1, layer_2, layer_3 = [], [], []

    num_frames = (len(code_list) + 1) // 7
    for i in range(num_frames):
        base = 7 * i
        layer_1.append(code_list[base + 0])
        layer_2.append(code_list[base + 1] - 4096)
        layer_3.append(code_list[base + 2] - (2 * 4096))
        layer_3.append(code_list[base + 3] - (3 * 4096))
        layer_2.append(code_list[base + 4] - (4 * 4096))
        layer_3.append(code_list[base + 5] - (5 * 4096))
        layer_3.append(code_list[base + 6] - (6 * 4096))

    codes = [
        torch.tensor(layer_1, device=device).unsqueeze(0),
        torch.tensor(layer_2, device=device).unsqueeze(0),
        torch.tensor(layer_3, device=device).unsqueeze(0),
    ]

    with torch.inference_mode():
        audio = snac_model.decode(codes).detach().squeeze().cpu().numpy()

    return audio


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the LLM and SNAC decoder into GPU memory."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC

        self.device = "cuda"
        model_id = "kenpath/svara-tts-v1"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Loading LLM (3B params)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded, vocab size: {self.model.config.vocab_size}")

        print("Loading SNAC decoder...")
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

        print("Running warmup inference...")
        self._synthesize("hello", "en_female", None)
        print("Setup complete.")

    # -----------------------------------------------------------------------
    # Prompt building (matches Gradio demo process_prompt exactly)
    # -----------------------------------------------------------------------
    def _build_prompt(
        self, text: str, voice: str, emotion: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build prompt token IDs and attention mask.

        Format: [128259] + tokenizer(prompt).input_ids + [128009, 128260]
          128259 = START_OF_HUMAN
          128009 = END_OF_TURN
          128260 = END_OF_HUMAN
        """
        # Parse voice into language label + gender
        parts = voice.split("_")
        lang_code = parts[0]
        gender = parts[1].capitalize() if len(parts) > 1 else "Female"
        lang_label = LANG_MAP.get(lang_code, "Hindi")

        # Build the text prompt with optional emotion tag
        tail = ""
        if emotion and emotion in VALID_STYLES and emotion != "neutral":
            tail = f" <{emotion}>"

        prompt = f"{lang_label} ({gender}): {text}{tail}"

        # Tokenize (the Gradio demo uses tokenizer(prompt) which adds no BOS by default)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Wrap with special tokens: START_OF_HUMAN ... END_OF_TURN END_OF_HUMAN
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        attention_mask = torch.ones_like(modified_input_ids)

        return modified_input_ids.to(self.device), attention_mask.to(self.device)

    # -----------------------------------------------------------------------
    # Full synthesis pipeline (matches Gradio demo generate_speech exactly)
    # -----------------------------------------------------------------------
    def _synthesize(
        self,
        text: str,
        voice: str,
        emotion: str | None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        max_tokens: int = 2048,
    ) -> np.ndarray:
        """prompt -> model.generate() -> parse_output() -> redistribute_codes() -> audio"""

        # 1) Build prompt
        input_ids, attention_mask = self._build_prompt(text, voice, emotion)
        print(f"Prompt length: {input_ids.shape[1]} tokens")

        # 2) Generate (matching Gradio demo parameters)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,  # END_OF_SPEECH
            )

        total_tokens = generated_ids.shape[1]
        new_tokens = total_tokens - input_ids.shape[1]
        print(f"Generated {new_tokens} new tokens (total sequence: {total_tokens})")

        # 3) Parse output: find START_OF_SPEECH, crop, remove END_OF_SPEECH, subtract 128266
        code_list = parse_output(generated_ids)
        print(f"Extracted {len(code_list)} SNAC codes ({len(code_list) // 7} frames)")

        if not code_list:
            print("WARNING: No audio codes extracted, returning silence")
            return np.zeros(SR, dtype=np.float32)

        # Debug: print code range
        print(f"Code range: min={min(code_list)}, max={max(code_list)}")

        # 4) Decode ALL codes at once via SNAC (NO sliding window, NO slicing)
        audio = redistribute_codes(code_list, self.snac, self.device)
        print(f"Audio: {audio.shape[0]} samples ({audio.shape[0] / SR:.2f}s), "
              f"range=[{audio.min():.4f}, {audio.max():.4f}]")

        return audio

    # -----------------------------------------------------------------------
    # Cog predict endpoint
    # -----------------------------------------------------------------------
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
            description="Emotion/style tag to apply to the speech.",
            default="none",
            choices=["none", "neutral", "formal", "chat", "clear", "happy", "surprise", "sad", "fear", "anger", "disgust"],
        ),
        temperature: float = Input(
            description="Sampling temperature. Higher = more expressive prosody.",
            default=0.7,
            ge=0.1,
            le=1.5,
        ),
        top_p: float = Input(
            description="Top-p (nucleus sampling). 0.6-0.8 for natural, 0.8-1.0 for expressive.",
            default=0.8,
            ge=0.1,
            le=1.0,
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty. >= 1.1 recommended to prevent loops.",
            default=1.1,
            ge=0.9,
            le=2.0,
        ),
        max_tokens: int = Input(
            description="Maximum new tokens to generate. Typical: 900-1200 for most sentences.",
            default=2048,
            ge=100,
            le=4096,
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
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start
        print(f"Synthesis took {elapsed:.2f}s for {len(text)} chars")

        # Normalize to prevent quiet audio
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95

        audio = np.clip(audio, -1.0, 1.0)

        output_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(output_path), audio, SR)
        return output_path
