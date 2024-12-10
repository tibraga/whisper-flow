""" test scenario module """

import time
import uuid
import asyncio
from queue import Queue
from typing import Callable
import Levenshtein

from whisperflow.corrections.mixed_correction_strategy import MixedCorrectionStrategy
from whisperflow.corrections.medical_spell_checker import MedicalSpellChecker

def get_all(queue: Queue) -> list:
    """get_all from queue"""
    res = []
    while queue and not queue.empty():
        res.append(queue.get())
    return res


async def transcribe(
    should_stop: list,
    queue: Queue,
    transcriber: Callable[[list], str],
    segment_closed: Callable[[dict], None],
    transcription_language: str,
    translation_language: str,
):
    """the transcription loop"""
    window, prev_result, cycles = [], {}, 0

    while not should_stop[0]:
        start = time.time()
        await asyncio.sleep(0.01)
        window.extend(get_all(queue))

        if not window:
            continue

        result = {
            "is_partial": True,
            "data": await transcriber(window),
            "time": (time.time() - start) * 1000,
        }
        if should_close_segment_improved(result, prev_result, result.get("time"), cycles):
            if not translation_language:
                mixed_strategy = MixedCorrectionStrategy(language=transcription_language)
                # Inicializa o controlador com a estratégia mista
                spell_checker = MedicalSpellChecker(strategy=mixed_strategy)
                text_corrected = spell_checker.correct_text(result["data"]["text"])
                result["data"]["text"] = text_corrected

            window.clear()
            prev_result, cycles = {}, 0
            result["is_partial"] = False
        elif result["data"]["text"] == prev_result.get("data", {}).get("text", ""):
            cycles += 1
        else:
            cycles = 0
            prev_result = result

        print(f"result: {result}")

        if result["data"]["text"]:
            await segment_closed(result)

def should_close_segment_improved(result: dict, prev_result: dict, time_processing: float, cycles, max_cycles=1):
    """return if segment should be closed improved"""
    words1 = result["data"]["text"].split()
    words2 = prev_result.get("data", {}).get("text", "").split()

    # Verifica se está levando muito tempo para processar, então interrompo logo.
    if time_processing > 2000:
        return True
    # Verifica se as frases têm o mesmo número de palavras
    if len(words1) != len(words2):
        return False
    else:
        semelhantes = True
        for p1, p2 in zip(words1, words2):
            distancia = Levenshtein.distance(p1, p2)
            if distancia > 2:
                semelhantes = False
                break
        return cycles >= max_cycles and semelhantes

def should_close_segment(result: dict, prev_result: dict, cycles, max_cycles=1):
    """return if segment should be closed"""
    return cycles >= max_cycles and result["data"]["text"] == prev_result.get(
        "data", {}
    ).get("text", "")


class TrancribeSession:  # pylint: disable=too-few-public-methods
    """transcription state"""

    def __init__(self, transcribe_async, send_back_async, transcription_language, translation_language) -> None:
        """ctor"""
        self.id = uuid.uuid4()  # pylint: disable=invalid-name
        self.queue = Queue()
        self.should_stop = [False]
        self.transcription_language = transcription_language
        self.translation_language = translation_language
        self.task = asyncio.create_task(
            transcribe(self.should_stop, self.queue, transcribe_async, send_back_async, self.transcription_language, self.translation_language)
        )

    def add_chunk(self, chunk: bytes):
        """add new chunk"""
        self.queue.put_nowait(chunk)

    async def stop(self):
        """stop session"""
        self.should_stop[0] = True
        await self.task