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
import uuid

models = {}

language_codes = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "czech": "cs",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish": "fi",
    "faroese": "fo",
    "french": "fr",
    "galician": "gl",
    "gujarati": "gu",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "croatian": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "indonesian": "id",
    "icelandic": "is",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "georgian": "ka",
    "kazakh": "kk",
    "khmer": "km",
    "kannada": "kn",
    "korean": "ko",
    "latin": "la",
    "luxembourgish": "lb",
    "lingala": "ln",
    "lao": "lo",
    "lithuanian": "lt",
    "latvian": "lv",
    "malagasy": "mg",
    "maori": "mi",
    "macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "marathi": "mr",
    "malay": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nepali": "ne",
    "dutch": "nl",
    "norwegian nynorsk": "nn",
    "norwegian": "no",
    "occitan": "oc",
    "punjabi": "pa",
    "polish": "pl",
    "pashto": "ps",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "sindhi": "sd",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "sundanese": "su",
    "swedish": "sv",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "turkmen": "tk",
    "tagalog": "tl",
    "turkish": "tr",
    "tatar": "tt",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yiddish": "yi",
    "yoruba": "yo",
    "chinese": "zh",
    "cantonese": "yue",
    "multi": None
}

# Run on GPU with FP16
asr_pipeline_default = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16"
)
asr_pipeline_largev3 = WhisperModel(
    "large-v3", device="cuda", compute_type="float16"
)

# deprecated
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

    return file_path

def transcribe_pcm_chunks(
    model: Whisper, chunks: list, transcription_language="portuguese", translation_language=None, temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks list"""
    # arr = (
    #     np.frombuffer(b"".join(chunks), np.int16).flatten().astype(np.float32) / 32768.0
    # )

    audio_bytes = b''.join(chunks)
    file_name = f"{str(uuid.uuid4())}.wav"
    path_wav = save_audio_to_file(audio_bytes, file_name)
    wav = read_audio(path_wav)
    vad_model = load_silero_vad()
    speech_segments = get_speech_timestamps(wav, vad_model)
    cleaned_audio = torch.tensor([], dtype=torch.float32)
    if len(speech_segments) > 0:
        cleaned_audio = collect_chunks(speech_segments, wav)
    cleaned_audio_array = cleaned_audio.numpy()

    language_transcribe = (
            None
            if transcription_language is None
            else language_codes.get(transcription_language.lower())
        )
    task = (
        "transcribe" if translation_language is None
        else "translate"
    )
    asr_pipeline = (
        asr_pipeline_default if translation_language is None
        else asr_pipeline_largev3
    )
    segments, info = asr_pipeline.transcribe(
            cleaned_audio_array, word_timestamps=True, language=language_transcribe, vad_filter=True, task=task
        )
    segments = list(segments)  # The transcription will actually run here.
    
    os.remove(path_wav)

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
    model: Whisper, chunks: list, transcription_language="portuguese", translation_language=None, temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks async"""
    return await asyncio.get_running_loop().run_in_executor(
        None, transcribe_pcm_chunks, model, chunks, transcription_language, translation_language, temperature, log_prob
    )
