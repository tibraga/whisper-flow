""" transcriber """

import os
import asyncio

import torch
import numpy as np

import whisper
from whisper import Whisper

from faster_whisper import WhisperModel

models = {}


def get_model(file_name="tiny.en.pt") -> Whisper:
    """load models from disk"""
    if file_name not in models:
        path = os.path.join(os.path.dirname(__file__), f"./models/{file_name}")
        models[file_name] = whisper.load_model(path).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return models[file_name]


def transcribe_pcm_chunks(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks list"""
    arr = (
        np.frombuffer(b"".join(chunks), np.int16).flatten().astype(np.float32) / 32768.0
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Inicializando whisper...")
        # Run on GPU with FP16
    asr_pipeline = WhisperModel(
        "deepdml/faster-whisper-large-v3-turbo-ct2", device=device, compute_type="float16"
    )
    print("Inicializando whisper...OK")
    return asr_pipeline.transcribe(
            arr, word_timestamps=True, language="pt", vad_filter=True
        )
    # return model.transcribe(
    #     arr,
    #     fp16=False,
    #     language=lang,
    #     logprob_threshold=log_prob,
    #     temperature=temperature,
    # )


async def transcribe_pcm_chunks_async(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks async"""
    return await asyncio.get_running_loop().run_in_executor(
        None, transcribe_pcm_chunks, model, chunks, lang, temperature, log_prob
    )
