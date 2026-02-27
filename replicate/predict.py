import asyncio
import base64
import io
import json
import os
import time
import wave
from typing import AsyncGenerator
from uuid import uuid4

import numpy as np
import torch
from cog import BasePredictor, ConcatenateIterator, Input
from snac import SNAC
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_ID = os.getenv("VLLM_MODEL", "kenpath/voice-svara-tts-v1-fft-v0.5")
SNAC_MODEL_ID = os.getenv("SNAC_MODEL", "hubertsiuzdak/snac_24khz")

SAMPLE_RATE = 24000
SUBCODEBOOK_SIZE = 4096
TOTAL_CODEBOOKS = 7
SUPPORTED_VOICES = [
    ("aaradhya", "Aaradhya"),
    ("abhiram", "Abhiram"),
    ("aditya", "Aditya"),
    ("akshat", "Akshat"),
    ("akshay", "Akshay"),
    ("ananya", "Ananya"),
    ("anika", "Anika"),
    ("anirudh", "Anirudh"),
    ("anjali", "Anjali"),
    ("ayesha", "Ayesha"),
    ("bunty", "Bunty"),
    ("chanchal", "Chanchal"),
    ("danish", "Danish"),
    ("digpal", "Digpal"),
    ("hansika", "Hansika"),
    ("kanika", "Kanika"),
    ("karan", "Karan"),
    ("karthik", "Karthik"),
    ("kishan", "Kishan"),
    ("likhitha", "Likhitha"),
    ("madhu", "Madhu"),
    ("maheshwari", "Maheshwari"),
    ("manav", "Manav"),
    ("monica", "Monica"),
    ("mumtaz", "Mumtaz"),
    ("prakash", "Prakash"),
    ("priya", "Priya"),
    ("rahul", "Rahul"),
    ("rajesh", "Rajesh"),
    ("ranbir", "Ranbir"),
    ("riya", "Riya"),
    ("sagar", "Sagar"),
    ("samisha", "Samisha"),
    ("sapna", "Sapna"),
    ("shalini", "Shalini"),
    ("shashank", "Shashank"),
    ("shivam", "Shivam"),
    ("shivani", "Shivani"),
    ("siddharth", "Siddharth"),
    ("simran", "Simran"),
    ("smrithi", "Smrithi"),
    ("sneha", "Sneha"),
    ("tanvi", "Tanvi"),
    ("tanya", "Tanya"),
    ("tripti", "Tripti"),
    ("vasanth", "Vasanth"),
    ("vidya", "Vidya"),
    ("viraj", "Viraj"),
    ("vishal", "Vishal"),
    ("yash", "Yash"),
]
VOICE_NAME_BY_ID = {voice_id: name for voice_id, name in SUPPORTED_VOICES}
VOICE_ID_CHOICES = [voice_id for voice_id, _ in SUPPORTED_VOICES]


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Keep token fallback behavior aligned with the original Modal script.
        self.start_token = self._token_id_or_default("<custom_token_7>", 128259)
        self.end_tokens = [
            self._token_id_or_default("<|eot_id|>", 128009),
            self._token_id_or_default("<custom_token_8>", 128260),
            self._token_id_or_default("<custom_token_9>", 128261),
            self._token_id_or_default("<custom_token_5>", 128257),
        ]

        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            trust_remote_code=False,
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            quantization=os.getenv("VLLM_QUANTIZATION") or None,
            enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "true").lower() == "true",
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)

        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_ID).to(self.device)
        self.snac_model.eval()

    def _token_id_or_default(self, token_text: str, default: int) -> int:
        tok_id = self.tokenizer.convert_tokens_to_ids(token_text)
        if isinstance(tok_id, int) and tok_id >= 0:
            return tok_id
        return default

    def _format_prompt_string(self, voice_name: str, text: str) -> str:
        formatted = f"<|audio|> {voice_name}: {text}<|eot_id|>"
        wrapped = "<custom_token_3>" + formatted + "<custom_token_4><custom_token_5>"
        prompt_tokens = self.tokenizer(wrapped, return_tensors="pt")
        start_token = torch.tensor([[self.start_token]], dtype=torch.int64)
        end_tokens = torch.tensor([self.end_tokens], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        return self.tokenizer.decode(all_input_ids[0])

    def _turn_token_into_id(self, token_string: str, index: int):
        token_string = token_string.strip()
        last_token_start = token_string.rfind("<custom_token_")
        if last_token_start == -1:
            return None
        last_token = token_string[last_token_start:]
        if not (last_token.startswith("<custom_token_") and last_token.endswith(">")):
            return None
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % TOTAL_CODEBOOKS) * SUBCODEBOOK_SIZE)
        except ValueError:
            return None

    def _decode_recent_multiframe_to_pcm16(self, multiframe):
        if len(multiframe) < TOTAL_CODEBOOKS:
            return b""

        codes_0 = torch.tensor([], device=self.device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.device, dtype=torch.int32)

        num_frames = len(multiframe) // TOTAL_CODEBOOKS
        frame = multiframe[: num_frames * TOTAL_CODEBOOKS]
        for j in range(num_frames):
            i = TOTAL_CODEBOOKS * j
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=self.device, dtype=torch.int32)])
            codes_1 = torch.cat(
                [
                    codes_1,
                    torch.tensor([frame[i + 1]], device=self.device, dtype=torch.int32),
                    torch.tensor([frame[i + 4]], device=self.device, dtype=torch.int32),
                ]
            )
            codes_2 = torch.cat(
                [
                    codes_2,
                    torch.tensor([frame[i + 2]], device=self.device, dtype=torch.int32),
                    torch.tensor([frame[i + 3]], device=self.device, dtype=torch.int32),
                    torch.tensor([frame[i + 5]], device=self.device, dtype=torch.int32),
                    torch.tensor([frame[i + 6]], device=self.device, dtype=torch.int32),
                ]
            )

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        if (
            torch.any(codes[0] < 0)
            or torch.any(codes[0] > 4096)
            or torch.any(codes[1] < 0)
            or torch.any(codes[1] > 4096)
            or torch.any(codes[2] < 0)
            or torch.any(codes[2] > 4096)
        ):
            return b""

        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)

        audio_slice = audio_hat[:, :, 2048:4096]
        audio_np = audio_slice.detach().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _decode_full_buffer_to_pcm16(self, buffer):
        if len(buffer) < TOTAL_CODEBOOKS:
            return b""

        num_frames = len(buffer) // TOTAL_CODEBOOKS
        frame = buffer[: num_frames * TOTAL_CODEBOOKS]

        codes_0 = []
        codes_1 = []
        codes_2 = []

        for j in range(num_frames):
            i = TOTAL_CODEBOOKS * j
            codes_0.append(frame[i])
            codes_1.extend([frame[i + 1], frame[i + 4]])
            codes_2.extend([frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])

        codes = [
            torch.tensor([codes_0], device=self.device, dtype=torch.int32),
            torch.tensor([codes_1], device=self.device, dtype=torch.int32),
            torch.tensor([codes_2], device=self.device, dtype=torch.int32),
        ]

        if (
            torch.any(codes[0] < 0)
            or torch.any(codes[0] > 4096)
            or torch.any(codes[1] < 0)
            or torch.any(codes[1] > 4096)
            or torch.any(codes[2] < 0)
            or torch.any(codes[2] > 4096)
        ):
            return b""

        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)

        audio_np = audio_hat.detach().cpu().numpy().reshape(-1)
        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _pcm16_to_wav_bytes(self, pcm16_bytes: bytes):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16_bytes)
        return buf.getvalue()

    async def _generate_stream_events(
        self, request_id: str, prompt_string: str, sampling_params: SamplingParams
    ) -> AsyncGenerator[str, None]:
        stream = self.llm.generate(
            prompt=prompt_string,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        req_start = time.time()
        buffer = []
        count = 0
        first_emit_at = None
        chunk_index = 0

        async for out in stream:
            token = self._turn_token_into_id(out.outputs[0].text, count)
            if token is None:
                continue
            if token > 0:
                buffer.append(token)
                count += 1

            if count % TOTAL_CODEBOOKS == 0 and count > 27:
                audio_samples = self._decode_recent_multiframe_to_pcm16(buffer[-28:])
                if audio_samples:
                    if first_emit_at is None:
                        first_emit_at = time.time()
                        print(
                            f"[ttft] req_id={request_id} mode=stream "
                            f"ttft_ms={(first_emit_at - req_start) * 1000.0:.2f}"
                        )

                    payload = {
                        "event": "chunk",
                        "chunk_index": chunk_index,
                        "wav_chunk_b64": base64.b64encode(self._pcm16_to_wav_bytes(audio_samples)).decode("ascii"),
                        "sampling_rate": SAMPLE_RATE,
                    }
                    chunk_index += 1
                    yield json.dumps(payload) + "\n"

        if first_emit_at is None:
            print(f"[ttft] req_id={request_id} mode=stream ttft_ms=NA")

        print(f"[timing] req_id={request_id} mode=stream total_ms={(time.time() - req_start) * 1000.0:.2f}")
        final = {
            "event": "done",
            "num_audio_tokens": len(buffer),
            "sampling_rate": SAMPLE_RATE,
        }
        yield json.dumps(final) + "\n"

    async def _generate_non_stream_event(self, request_id: str, prompt_string: str, sampling_params: SamplingParams):
        stream = self.llm.generate(
            prompt=prompt_string,