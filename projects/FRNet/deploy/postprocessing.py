from time import time

import numpy.typing as npt
import numpy as np


class Postprocessing:

    def __init__(self, score_threshold: float = 0.0, ignored_index: int = 16):
        self.score_threshold = score_threshold
        self.ignored_index = ignored_index

    def postprocess(self, predictions: npt.ArrayLike) -> npt.ArrayLike:
        t_start = time()
        result = np.where(
            np.max(predictions, axis=1) >= self.score_threshold,
            np.argmax(predictions, axis=1), self.ignored_index)
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        print(f'Postprocessing latency: {latency} ms')
        return result
