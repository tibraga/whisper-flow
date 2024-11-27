""" transcriber """

import os
import asyncio

import torch
import numpy as np

import whisper
from whisper import Whisper

from faster_whisper import WhisperModel

models = {}


# Run on GPU with FP16
asr_pipeline = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16"
)


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

    # Defina um limiar para o silêncio (ajuste conforme necessário)
    threshold = 0.01

    # Obtenha o valor absoluto do array de áudio
    abs_arr = np.abs(arr)

    # Encontre os índices onde o áudio não é silencioso
    non_silent_indices = np.where(abs_arr > threshold)[0]

    if len(non_silent_indices) > 0:
        # Determine os índices de início e fim
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1] + 1  # +1 para incluir o último sample
        # Recorte o array para remover o silêncio
        trimmed_arr = arr[start_index:end_index]
    else:
        # O array é totalmente silencioso
        trimmed_arr = np.array([])

    # Agora você pode usar 'trimmed_arr' para a transcrição

    device = "cuda" if torch.cuda.is_available() else "cpu"


    segments, info = asr_pipeline.transcribe(
            trimmed_arr, word_timestamps=True, language="pt", vad_filter=True
        )
    segments = list(segments)  # The transcription will actually run here.

    flattened_words = [
            word for segment in segments for word in segment.words
        ]

    to_return = {
        "language": info.language,
        "language_probability": info.language_probability,
        "text": " ".join([s.text.strip() for s in segments]),
        "words": [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            }
            for w in flattened_words
        ],
    }
    return to_return
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
