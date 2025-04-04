import numpy as np
import numpy.typing as npt

TRAIN_SPLIT = 0.85
TEST_SPLIT = (0.85, 0.95)
VAL_SPLIT = 0.95


def compute_kl_divergence(p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> float:

    epsilon = 1e-10
    # p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    valid_mask = p > 1e-10

    log_multiply = p[valid_mask] * np.log(p[valid_mask] / q[valid_mask])

    return np.sum(log_multiply)
