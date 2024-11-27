""" transcriber """

import os
import asyncio

import torch
import numpy as np

import whisper
from whisper import Whisper

from faster_whisper import WhisperModel

from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks
import torchaudio

import io
import wave

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
    # arr = (
    #     np.frombuffer(b"".join(chunks), np.int16).flatten().astype(np.float32) / 32768.0
    # )

    audio_bytes = b''.join(chunks)

    # Defina os parâmetros do áudio
    sample_width = 2      # Largura de amostra em bytes (2 bytes para 16 bits)
    n_channels = 1        # Número de canais (1 para mono, ajuste se for estéreo)
    sample_rate = 16000   # Taxa de amostragem em Hz (ajuste conforme o seu áudio)

    # Crie um buffer em memória para escrever o arquivo WAV
    buffer = io.BytesIO()

    # Escreva os dados de áudio no buffer como um arquivo WAV
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

    # Mova o ponteiro do buffer para o início
    buffer.seek(0)

    # Carregue o áudio usando torchaudio.load a partir do buffer
    waveform, sr = torchaudio.load(buffer)
    
    vad_model = load_silero_vad()
    speech_segments = get_speech_timestamps(waveform, vad_model, sampling_rate = sample_rate)
    cleaned_audio = collect_chunks(speech_segments, waveform)

    # Agora você pode usar 'trimmed_arr' para a transcrição

    device = "cuda" if torch.cuda.is_available() else "cpu"


    segments, info = asr_pipeline.transcribe(
            cleaned_audio, word_timestamps=True, language="pt", vad_filter=True
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
