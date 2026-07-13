"""BEVFusion detection-head output contract.

The transformation from the detection head's raw output dict to the
``(bbox_pred, score, label)`` triple is the output contract that the ONNX graph bakes in.
The PyTorch reference pipeline must produce the *identical* triple for PyTorch↔ONNX parity
to be meaningful, so both call this single function rather than each keeping a copy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def head_dict_to_detection_outputs(outputs: dict) -> tuple:
    """Turn the detection-head output dict into the ``(bbox_pred, score, label)`` outputs.

    Args:
        outputs: Detection-head output dict with ``heatmap``, ``query_labels``,
            ``query_heatmap_score``, and the ``center``/``height``/``dim``/``rot``/``vel`` regressions.

    Returns:
        Tuple ``(bbox_pred [10, num_proposals], score [num_proposals], label [num_proposals])``.
    """
    score = outputs["heatmap"].sigmoid()
    one_hot = F.one_hot(outputs["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
    score = score * outputs["query_heatmap_score"] * one_hot
    score = score[0].max(dim=0)[0]

    bbox_pred = torch.cat(
        [outputs["center"][0], outputs["height"][0], outputs["dim"][0], outputs["rot"][0], outputs["vel"][0]],
        dim=0,
    )

    return bbox_pred, score, outputs["query_labels"][0]
