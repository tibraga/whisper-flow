""" transcriber """

import os
import asyncio

import torch
import numpy as np
import librosa

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

    # Parâmetros ajustados para 16 kHz
    frame_length = 1024
    hop_length = 256

    # Remover silêncio
    trimmed_arr, index = librosa.effects.trim(
        arr,
        top_db=20,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Verificar se o áudio não está vazio após a remoção de silêncio
    if trimmed_arr.size > 0:
        # Transcrever usando o modelo Whisper
        # Por exemplo, se o modelo espera um arquivo de áudio ou um array numpy específico
        # transcricao = whisper.transcribe(trimmed_arr, sample_rate=16000)
        print("O áudio após a remoção de silêncio NÃO está vazio.")
    else:
        print("O áudio após a remoção de silêncio está vazio.")
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
