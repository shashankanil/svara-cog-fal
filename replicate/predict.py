from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging
import os
import re
import subprocess
import tempfile
import time
import wave
from typing import List, Optional
from uuid import uuid4

import numpy as np
import torch
import torchaudio
from cog import BasePredictor, Input, Path
from snac import SNAC
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", MODEL_ID)
SNAC_MODEL_ID = os.getenv("SNAC_MODEL", "hubertsiuzdak/snac_24khz")

SAMPLE_RATE = 24000
TOTAL_CODEBOOKS = 7
SUBCODEBOOK_SIZE = 4096
AUDIO_TOKENS_START = 128266

BOS_TOKEN = 128000
END_OF_TURN = 128009
AUDIO_TOKEN = 156939
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262

AUDIO_TOKEN_OFFSETS = [AUDIO_TOKENS_START + (i * SUBCODEBOOK_SIZE) for i in range(TOTAL_CODEBOOKS)]

TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

VOICE_ID_TO_SPEAKER = {
    "hi_male": "Hindi (Male)",
    "hi_female": "Hindi (Female)",
    "bn_male": "Bengali (Male)",
    "bn_female": "Bengali (Female)",
    "mr_male": "Marathi (Male)",
    "mr_female": "Marathi (Female)",
    "te_male": "Telugu (Male)",
    "te_female": "Telugu (Female)",
    "kn_male": "Kannada (Male)",
    "kn_female": "Kannada (Female)",
    "bh_male": "Bhojpuri (Male)",
    "bh_female": "Bhojpuri (Female)",
    "mag_male": "Magahi (Male)",
    "mag_female": "Magahi (Female)",
    "hne_male": "Chhattisgarhi (Male)",
    "hne_female": "Chhattisgarhi (Female)",
    "mai_male": "Maithili (Male)",
    "mai_female": "Maithili (Female)",
    "as_male": "Assamese (Male)",
    "as_female": "Assamese (Female)",
    "brx_male": "Bodo (Male)",
    "brx_female": "Bodo (Female)",
    "doi_male": "Dogri (Male)",
    "doi_female": "Dogri (Female)",
    "gu_male": "Gujarati (Male)",
    "gu_female": "Gujarati (Female)",
    "ml_male": "Malayalam (Male)",
    "ml_female": "Malayalam (Female)",
    "pa_male": "Punjabi (Male)",
    "pa_female": "Punjabi (Female)",
    "ta_male": "Tamil (Male)",
    "ta_female": "Tamil (Female)",
    "en_male": "English (Male)",
    "en_female": "English (Female)",
    "ne_male": "Nepali (Male)",
    "ne_female": "Nepali (Female)",
    "sa_male": "Sanskrit (Male)",
    "sa_female": "Sanskrit (Female)",
}

DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    "। ",
    ". ",
    "? ",
    "! ",
    "… ",
    ",",
    " ",
    "",
]


class ImmediateFuture:
    def __init__(self, result: bytes):
        self._result = result

    def result(self) -> bytes:
        return self._result


class SvaraMapper:
    def __init__(self, window_size: int = 28):
        if window_size < 7 or window_size % 7 != 0:
            raise ValueError(f"window_size must be a multiple of 7, got {window_size}")
        self.window_size = window_size
        self.codes: List[int] = []
        self.good = 0

    def feed_raw(self, raw_num: int) -> Optional[List[int]]:
        code = raw_num - 10 - ((self.good % TOTAL_CODEBOOKS) * SUBCODEBOOK_SIZE)
        if code <= 0:
            return None
        self.codes.append(code)
        self.good += 1
        if self.good % TOTAL_CODEBOOKS == 0 and self.good >= self.window_size:
            return self.codes[-self.window_size :]
        return None


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def detect_max_workers(snac_device: str) -> int:
    if snac_device == "cpu":
        cpu_count = os.cpu_count() or 4
        return max(2, cpu_count // 2)

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            if vram_gb >= 16 and props.major >= 8:
                return 4
        except Exception:
            pass

    return 2


def extract_custom_token_numbers(text: str) -> List[int]:
    out: List[int] = []
    for match in TOKEN_RE.findall(text or ""):
        try:
            value = int(match)
            if value != 0:
                out.append(value)
        except Exception:
            continue
    return out


def hard_split(text: str, max_len: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end
        if start >= len(text) or (overlap > 0 and start == end - overlap and end == len(text)):
            break

    return chunks


def split_text_recursive(text: str, max_len: int, overlap: int, separators: List[str]) -> List[str]:
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []

    for separator in separators:
        if separator == "":
            break

        if separator in text:
            parts = text.split(separator)
            current_chunk = ""

            for i, part in enumerate(parts):
                part_with_sep = f"{part}{separator}" if i < len(parts) - 1 else part

                if len(part_with_sep) > max_len:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""

                    remaining = separators[separators.index(separator) + 1 :]
                    if remaining:
                        chunks.extend(split_text_recursive(part_with_sep, max_len, overlap, remaining))
                    else:
                        chunks.extend(hard_split(part_with_sep, max_len, overlap))
                    continue

                if len(current_chunk) + len(part_with_sep) > max_len:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        if overlap > 0 and len(current_chunk) >= overlap:
                            current_chunk = current_chunk[-overlap:] + part_with_sep
                        else:
                            current_chunk = part_with_sep
                    else:
                        current_chunk = part_with_sep
                else:
                    current_chunk += part_with_sep

            if current_chunk:
                chunks.append(current_chunk.strip())

            return [c for c in chunks if c]

    return hard_split(text, max_len, overlap)


def chunk_text(text: str, max_len: int = 200, overlap: int = 0) -> List[str]:
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    return [c.strip() for c in split_text_recursive(text, max_len, overlap, DEFAULT_SEPARATORS) if c.strip()]


def crossfade_pcm(a: bytes, b: bytes, overlap_ms: int = 50, sample_rate: int = SAMPLE_RATE) -> bytes:
    overlap_samples = int(sample_rate * overlap_ms / 1000)
    if overlap_samples <= 0:
        return a + b

    arr_a = np.frombuffer(a, dtype=np.int16).astype(np.float32)
    arr_b = np.frombuffer(b, dtype=np.int16).astype(np.float32)

    if len(arr_a) < overlap_samples or len(arr_b) < overlap_samples:
        return a + b

    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

    blended = arr_a[-overlap_samples:] * fade_out + arr_b[:overlap_samples] * fade_in

    merged = np.concatenate(
        [
            arr_a[:-overlap_samples],
            blended,
            arr_b[overlap_samples:],
        ]
    )

    return np.clip(merged, -32768, 32767).astype(np.int16).tobytes()


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"

        self.model_id = os.getenv("VLLM_MODEL", MODEL_ID)
        self.tokenizer_model = os.getenv("TOKENIZER_MODEL", TOKENIZER_MODEL)
        self.snac_model_id = os.getenv("SNAC_MODEL", SNAC_MODEL_ID)
        self.snac_device = os.getenv("SNAC_DEVICE", self.device).lower()
        if self.snac_device == "cuda" and not torch.cuda.is_available():
            logger.warning("SNAC_DEVICE=cuda requested but CUDA is unavailable; falling back to cpu")
            self.snac_device = "cpu"
        if self.snac_device == "mps" and not torch.backends.mps.is_available():
            logger.warning("SNAC_DEVICE=mps requested but MPS is unavailable; falling back to cpu")
            self.snac_device = "cpu"

        self.snac_window_size = int(os.getenv("SNAC_WINDOW_SIZE", "28"))
        if self.snac_window_size < 7 or self.snac_window_size % 7 != 0:
            raise ValueError(f"SNAC_WINDOW_SIZE must be a multiple of 7, got {self.snac_window_size}")

        self.max_chunk_chars = int(os.getenv("DEFAULT_CHUNK_SIZE", "200"))
        self.crossfade_ms = int(os.getenv("CROSSFADE_MS", "50"))
        self.concurrent_decode = env_bool("CONCURRENT_DECODE", True)
        self.max_workers = detect_max_workers(self.snac_device)

        logger.info("Initializing Replicate predictor")
        logger.info("vLLM model=%s tokenizer=%s", self.model_id, self.tokenizer_model)
        logger.info(
            "SNAC model=%s device=%s window_size=%d workers=%d",
            self.snac_model_id,
            self.snac_device,
            self.snac_window_size,
            self.max_workers,
        )

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, token=hf_token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)

        self.bos_token = self.token_id_or_default("<|begin_of_text|>", BOS_TOKEN)
        self.end_of_turn = self.token_id_or_default("<|eot_id|>", END_OF_TURN)
        self.audio_token = self.token_id_or_default("<|audio|>", AUDIO_TOKEN)
        self.start_of_speech = self.token_id_or_default("<custom_token_1>", START_OF_SPEECH)
        self.end_of_speech = self.token_id_or_default("<custom_token_2>", END_OF_SPEECH)
        self.start_of_human = self.token_id_or_default("<custom_token_3>", START_OF_HUMAN)
        self.end_of_human = self.token_id_or_default("<custom_token_4>", END_OF_HUMAN)
        self.start_of_ai = self.token_id_or_default("<custom_token_5>", START_OF_AI)
        self.end_of_ai = self.token_id_or_default("<custom_token_6>", END_OF_AI)

        engine_args = AsyncEngineArgs(
            model=self.model_id,
            trust_remote_code=env_bool("VLLM_TRUST_REMOTE_CODE", True),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            quantization=os.getenv("VLLM_QUANTIZATION") or None,
            enforce_eager=env_bool("VLLM_ENFORCE_EAGER", False),
            attention_backend=os.getenv("VLLM_ATTENTION_BACKEND") or None,
            kv_cache_dtype=os.getenv("VLLM_KV_CACHE_DTYPE", "auto"),
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)

        self.snac_model = SNAC.from_pretrained(self.snac_model_id).to(self.snac_device)
        self.snac_model.eval()

        if env_bool("SNAC_COMPILE", True):
            try:
                self.snac_model = torch.compile(self.snac_model)
                logger.info("SNAC decoder compiled with torch.compile")
            except Exception as err:
                logger.warning("SNAC compile skipped: %s", err)

        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up SNAC decoder")
        self.decode_window_to_pcm16([1] * self.snac_window_size)
        logger.info("SNAC warmup complete")

    def token_id_or_default(self, token_text: str, default: int) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token_text)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return default

    def resolve_speaker_id(self, voice: str) -> str:
        candidate = (voice or "").strip()
        if not candidate:
            raise ValueError("voice must be provided")

        lookup = VOICE_ID_TO_SPEAKER.get(candidate.lower())
        if lookup:
            return lookup

        return candidate

    def scalar_id(self, token_id: int) -> torch.Tensor:
        return torch.tensor([[token_id]], dtype=torch.int64)

    def human_turn(self, text_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.scalar_id(self.start_of_human),
                self.scalar_id(self.audio_token),
                text_ids,
                self.scalar_id(self.end_of_human),
                self.scalar_id(self.end_of_turn),
            ],
            dim=1,
        )

    def audio_turn(self, audio_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.scalar_id(self.start_of_ai),
                self.scalar_id(self.start_of_speech),
                audio_ids,
                self.scalar_id(self.end_of_speech),
                self.scalar_id(self.end_of_ai),
                self.scalar_id(self.end_of_turn),
            ],
            dim=1,
        )

    def final_generation_prefix(self) -> torch.Tensor:
        return torch.cat([self.scalar_id(self.start_of_ai), self.scalar_id(self.start_of_speech)], dim=1)

    def build_prompt_string(
        self,
        text: str,
        voice: str,
        audio_tokens: Optional[List[int]] = None,
        reference_transcript: Optional[str] = None,
    ) -> str:
        blocks: List[torch.Tensor] = [self.scalar_id(self.bos_token)]

        if audio_tokens is not None:
            if reference_transcript and reference_transcript.strip():
                transcript_ids = self.tokenizer(
                    reference_transcript,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids
                blocks.append(self.human_turn(transcript_ids))

            audio_ids = torch.tensor([audio_tokens], dtype=torch.int64)
            blocks.append(self.audio_turn(audio_ids))

            text_ids = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            blocks.append(self.human_turn(text_ids))
            blocks.append(self.final_generation_prefix())

        else:
            speaker_id = self.resolve_speaker_id(voice)
            prompt = f"{speaker_id}: {text}"
            text_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            blocks.append(self.human_turn(text_ids))
            blocks.append(self.final_generation_prefix())

        full_ids = torch.cat(blocks, dim=1).view(-1)
        return self.tokenizer.decode(full_ids.tolist(), skip_special_tokens=False)

    def decode_window_to_pcm16(self, window: List[int]) -> bytes:
        if not window or len(window) < TOTAL_CODEBOOKS:
            return b""

        frames = len(window) // TOTAL_CODEBOOKS
        frame = window[: frames * TOTAL_CODEBOOKS]

        tensor = torch.tensor(frame, dtype=torch.int32, device=self.snac_device).view(frames, TOTAL_CODEBOOKS)
        codes_0 = tensor[:, 0].reshape(1, -1)
        codes_1 = tensor[:, [1, 4]].reshape(1, -1)
        codes_2 = tensor[:, [2, 3, 5, 6]].reshape(1, -1)

        if (
            torch.any((codes_0 < 0) | (codes_0 > 4096))
            or torch.any((codes_1 < 0) | (codes_1 > 4096))
            or torch.any((codes_2 < 0) | (codes_2 > 4096))
        ):
            return b""

        with torch.inference_mode():
            audio = self.snac_model.decode([codes_0, codes_1, codes_2])[:, :, 2048:4096]

        audio_np = audio.detach().float().cpu().numpy().reshape(-1)
        return (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def decode_full_codes_to_pcm16(self, codes: List[int]) -> bytes:
        if not codes or len(codes) < TOTAL_CODEBOOKS:
            return b""

        frames = len(codes) // TOTAL_CODEBOOKS
        frame = codes[: frames * TOTAL_CODEBOOKS]

        tensor = torch.tensor(frame, dtype=torch.int32, device=self.snac_device).view(frames, TOTAL_CODEBOOKS)
        codes_0 = tensor[:, 0].reshape(1, -1)
        codes_1 = tensor[:, [1, 4]].reshape(1, -1)
        codes_2 = tensor[:, [2, 3, 5, 6]].reshape(1, -1)

        if (
            torch.any((codes_0 < 0) | (codes_0 > 4096))
            or torch.any((codes_1 < 0) | (codes_1 > 4096))
            or torch.any((codes_2 < 0) | (codes_2 > 4096))
        ):
            return b""

        with torch.inference_mode():
            audio = self.snac_model.decode([codes_0, codes_1, codes_2])

        audio_np = audio.detach().float().cpu().numpy().reshape(-1)
        return (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def encode_reference_audio_to_tokens(self, reference_audio: Path) -> List[int]:
        waveform, sample_rate = torchaudio.load(str(reference_audio))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(dtype=torch.float32, device=self.snac_device)
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE).to(self.snac_device)
            waveform = resampler(waveform)

        waveform = waveform.unsqueeze(0)

        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        all_tokens: List[int] = []
        coarse_frames = codes[0].shape[1]

        for i in range(coarse_frames):
            c0 = int(codes[0][0][i].item()) + AUDIO_TOKEN_OFFSETS[0]
            c1 = int(codes[1][0][2 * i].item()) + AUDIO_TOKEN_OFFSETS[1]
            c2 = int(codes[2][0][4 * i].item()) + AUDIO_TOKEN_OFFSETS[2]
            c3 = int(codes[2][0][4 * i + 1].item()) + AUDIO_TOKEN_OFFSETS[3]
            c4 = int(codes[1][0][2 * i + 1].item()) + AUDIO_TOKEN_OFFSETS[4]
            c5 = int(codes[2][0][4 * i + 2].item()) + AUDIO_TOKEN_OFFSETS[5]
            c6 = int(codes[2][0][4 * i + 3].item()) + AUDIO_TOKEN_OFFSETS[6]
            all_tokens.extend([c0, c1, c2, c3, c4, c5, c6])

        return all_tokens

    async def synthesize_chunk_pcm16(
        self,
        text: str,
        voice: str,
        sampling_params: SamplingParams,
        audio_tokens: Optional[List[int]] = None,
        reference_transcript: Optional[str] = None,
    ) -> bytes:
        request_id = f"req-{uuid4().hex}"
        prompt = self.build_prompt_string(
            text=text,
            voice=voice,
            audio_tokens=audio_tokens,
            reference_transcript=reference_transcript,
        )

        logger.info("request_id=%s prompt_chars=%d", request_id, len(prompt))

        stream = self.llm.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        mapper = SvaraMapper(window_size=self.snac_window_size)
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        pending: List[concurrent.futures.Future] = []
        pcm_parts: List[bytes] = []
        window_count = 0

        prev_text = ""
        req_start = time.time()
        first_window_at: Optional[float] = None

        if self.concurrent_decode:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        try:
            async for out in stream:
                if not out.outputs:
                    continue

                current_text = out.outputs[0].text
                if len(current_text) < len(prev_text):
                    delta = current_text
                else:
                    delta = current_text[len(prev_text) :]
                prev_text = current_text

                for raw_num in extract_custom_token_numbers(delta):
                    window = mapper.feed_raw(raw_num)
                    if window is None:
                        continue

                    if first_window_at is None:
                        first_window_at = time.time()

                    if executor is not None:
                        pending.append(executor.submit(self.decode_window_to_pcm16, window))
                    else:
                        pending.append(ImmediateFuture(self.decode_window_to_pcm16(window)))
                    window_count += 1

                    while len(pending) > 2:
                        pcm = pending.pop(0).result()
                        if pcm:
                            pcm_parts.append(pcm)

            for future in pending:
                pcm = future.result()
                if pcm:
                    pcm_parts.append(pcm)

        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        if first_window_at is None:
            ttft = "NA"
        else:
            ttft = f"{(first_window_at - req_start) * 1000.0:.2f}"

        total_ms = (time.time() - req_start) * 1000.0
        logger.info(
            "request_id=%s ttft_ms=%s total_ms=%.2f windows=%d codes=%d",
            request_id,
            ttft,
            total_ms,
            window_count,
            len(mapper.codes),
        )

        if pcm_parts:
            return b"".join(pcm_parts)

        if mapper.codes:
            logger.info("request_id=%s falling back to full-buffer decode", request_id)
            return self.decode_full_codes_to_pcm16(mapper.codes)

        return b""

    def stitch_chunks(self, chunks_pcm: List[bytes]) -> bytes:
        non_empty = [c for c in chunks_pcm if c]
        if not non_empty:
            return b""
        if len(non_empty) == 1:
            return non_empty[0]

        overlap_bytes = int(SAMPLE_RATE * self.crossfade_ms / 1000) * 2
        if overlap_bytes <= 0:
            return b"".join(non_empty)

        out = bytearray(non_empty[0])

        for chunk in non_empty[1:]:
            if len(out) < overlap_bytes or len(chunk) < overlap_bytes:
                out.extend(chunk)
                continue

            tail = bytes(out[-overlap_bytes:])
            head = chunk[:overlap_bytes]
            blended = crossfade_pcm(tail, head, overlap_ms=self.crossfade_ms, sample_rate=SAMPLE_RATE)

            del out[-overlap_bytes:]
            out.extend(blended)
            out.extend(chunk[overlap_bytes:])

        return bytes(out)

    def pcm16_to_wav_bytes(self, pcm16_bytes: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16_bytes)
        return buf.getvalue()

    def encode_output(self, pcm16_bytes: bytes, response_format: str) -> tuple[bytes, str]:
        fmt = (response_format or "wav").lower().strip()
        if fmt not in {"wav", "mp3", "opus", "aac", "pcm"}:
            raise ValueError(f"Unsupported response_format '{response_format}'")

        if fmt == "pcm":
            return pcm16_bytes, ".pcm"

        if fmt == "wav":
            return self.pcm16_to_wav_bytes(pcm16_bytes), ".wav"

        cmd = [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-loglevel",
            "error",
        ]

        if fmt == "mp3":
            cmd.extend(["-f", "mp3", "pipe:1"])
            suffix = ".mp3"
        elif fmt == "opus":
            cmd.extend(["-c:a", "libopus", "-f", "opus", "pipe:1"])
            suffix = ".opus"
        else:
            cmd.extend(["-c:a", "aac", "-f", "adts", "pipe:1"])
            suffix = ".aac"

        proc = subprocess.run(cmd, input=pcm16_bytes, capture_output=True, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg encoding failed for format={fmt}: {err}")

        return proc.stdout, suffix

    async def predict_async(
        self,
        input_text: str,
        voice: str,
        response_format: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_tokens: int,
        seed: int,
        chunk_size: int,
        reference_audio: Optional[Path],
        reference_transcript: str,
    ) -> tuple[bytes, str]:
        audio_tokens: Optional[List[int]] = None
        if reference_audio is not None:
            logger.info("Encoding reference audio from %s", reference_audio)
            audio_tokens = self.encode_reference_audio_to_tokens(reference_audio)
            logger.info("Reference audio encoded: %d tokens", len(audio_tokens))

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[self.end_of_speech],
            seed=seed if seed >= 0 else None,
        )

        effective_chunk_size = chunk_size if chunk_size > 0 else self.max_chunk_chars
        chunks = chunk_text(input_text, max_len=effective_chunk_size)
        logger.info("Input chars=%d chunk_size=%d chunks=%d", len(input_text), effective_chunk_size, len(chunks))

        pcm_chunks: List[bytes] = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("Synthesizing chunk %d/%d chars=%d", idx, len(chunks), len(chunk))
            pcm = await self.synthesize_chunk_pcm16(
                text=chunk,
                voice=voice,
                sampling_params=sampling_params,
                audio_tokens=audio_tokens,
                reference_transcript=reference_transcript,
            )
            pcm_chunks.append(pcm)

        merged_pcm = self.stitch_chunks(pcm_chunks)
        if not merged_pcm:
            raise ValueError("No audio tokens decoded from model output")

        return self.encode_output(merged_pcm, response_format)

    def predict(
        self,
        input: str = Input(description="Text to synthesize.", default="Hello, this is a test."),
        voice: str = Input(
            description=(
                "Voice ID (e.g., hi_male, en_female) or direct speaker string "
                "(e.g., Hindi (Male))."
            ),
            default="hi_male",
        ),
        response_format: str = Input(
            description="Output audio format.",
            default="wav",
            choices=["wav", "mp3", "opus", "aac", "pcm"],
        ),
        temperature: float = Input(description="Sampling temperature.", default=0.75, ge=0.0, le=2.0),
        top_p: float = Input(description="Top-p nucleus sampling.", default=0.9, ge=0.0, le=1.0),
        top_k: int = Input(description="Top-k sampling.", default=40, ge=-1, le=200),
        repetition_penalty: float = Input(description="Repetition penalty.", default=1.1, ge=1.0, le=2.0),
        max_tokens: int = Input(description="Maximum generated tokens.", default=2048, ge=1, le=4096),
        seed: int = Input(description="Random seed (-1 for random).", default=-1, ge=-1),
        chunk_size: int = Input(
            description="Max characters per long-text chunk (crossfaded).",
            default=200,
            ge=50,
            le=1000,
        ),
        reference_audio: Optional[Path] = Input(
            description="Optional reference audio file for zero-shot voice cloning.",
            default=None,
        ),
        reference_transcript: str = Input(
            description="Optional transcript for reference audio.",
            default="",
        ),
    ) -> Path:
        input_text = (input or "").strip()
        if not input_text:
            raise ValueError("input text must not be empty")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            output_bytes, suffix = loop.run_until_complete(
                self.predict_async(
                    input_text=input_text,
                    voice=voice,
                    response_format=response_format,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    seed=seed,
                    chunk_size=chunk_size,
                    reference_audio=reference_audio,
                    reference_transcript=reference_transcript,
                )
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        output_path = Path(tempfile.mktemp(suffix=suffix))
        with open(output_path, "wb") as file:
            file.write(output_bytes)

        return output_path
