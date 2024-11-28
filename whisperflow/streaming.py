""" test scenario module """

import time
import uuid
import asyncio
from queue import Queue
from typing import Callable
import Levenshtein


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
        # if should_close_segment(result, prev_result, cycles):
        #     window, prev_result, cycles = [], {}, 0
        #     result["is_partial"] = False
        # elif result["data"]["text"] == prev_result.get("data", {}).get("text", ""):
        #     cycles += 1
        # else:
        #     cycles = 0
        #     prev_result = result

        if should_close_segment_improved(result["data"]["text"], prev_result.get("data", {}).get("text", ""), result.get("time")):
            window, prev_result = [], {}
            result["is_partial"] = False
        else:
            prev_result = result

        print(f"result: {result}")

        if result["data"]["text"]:
            await segment_closed(result)

def should_close_segment_improved(result: dict, prev_result: dict, time_processing: float):
    """return if segment should be closed improved"""
    words1 = result.split()
    words2 = prev_result.split()

    # Verifica se está levando muito tempo para processar, então interrompo logo.
    if time_processing > 2000:
        return False
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
        return semelhantes

def should_close_segment(result: dict, prev_result: dict, cycles, max_cycles=1):
    """return if segment should be closed"""
    return cycles >= max_cycles and result["data"]["text"] == prev_result.get(
        "data", {}
    ).get("text", "")


class TrancribeSession:  # pylint: disable=too-few-public-methods
    """transcription state"""

    def __init__(self, transcribe_async, send_back_async) -> None:
        """ctor"""
        self.id = uuid.uuid4()  # pylint: disable=invalid-name
        self.queue = Queue()
        self.should_stop = [False]
        self.task = asyncio.create_task(
            transcribe(self.should_stop, self.queue, transcribe_async, send_back_async)
        )

    def add_chunk(self, chunk: bytes):
        """add new chunk"""
        self.queue.put_nowait(chunk)

    async def stop(self):
        """stop session"""
        self.should_stop[0] = True
        await self.task


"Não, são bem simples. O eletrocardiograma e o ecocardiograma serão feitos no laboratório de cardiologia. E o exame de sangue pode ser feito em qualquer laboratório. Vou encaminhar os guias para você."
"Não, são bem simples. O eletrocardiograma e o ecocardiograma serão feitos no laboratório de cardiologia. E o exame de sangue pode ser feito em qualquer laboratório. Vou encaminhar os guias para você."
