""" transcriber """

import os
import asyncio

import torch
import numpy as np

import whisper
from whisper import Whisper

from faster_whisper import WhisperModel

from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks, read_audio
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

def save_audio_to_file(
    audio_data, file_name, audio_dir="audio_files", audio_format="wav"
):
    """
    Saves the audio data to a file.

    :param audio_data: The audio data to save.
    :param file_name: The name of the file.
    :param audio_dir: Directory where audio files will be saved.
    :param audio_format: Format of the audio file.
    :return: Path to the saved audio file.
    """

    os.makedirs(audio_dir, exist_ok=True)

    file_path = os.path.join(audio_dir, file_name)

    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(1)  # Assuming mono audio
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_data)

    print(f"file_path: {file_path}")

    return file_path

def transcribe_pcm_chunks(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks list"""
    arr = (
        np.frombuffer(b"".join(chunks), np.int16).flatten().astype(np.float32) / 32768.0
    )

    # audio_bytes = b''.join(chunks)
    # path_wav = save_audio_to_file(audio_bytes, "teste.wav")
    # wav = read_audio(path_wav)
    # vad_model = load_silero_vad()
    # speech_segments = get_speech_timestamps(wav, vad_model)
    # cleaned_audio = torch.tensor([], dtype=torch.float32)
    # if len(speech_segments) > 0:
    #     cleaned_audio = collect_chunks(speech_segments, wav)

    # cleaned_audio_array = cleaned_audio.numpy()

    # Agora você pode usar 'trimmed_arr' para a transcrição

    vad_parameters = {
        "threshold": 0.1,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "window_size_samples": 512,
        "speech_pad_ms": 30
    }

    segments, info = asr_pipeline.transcribe(
            arr, word_timestamps=True, without_timestamps=True, language="pt", vad_filter=True, vad_parameters=vad_parameters
        )
    segments = list(segments)  # The transcription will actually run here.
    print(segments)
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
