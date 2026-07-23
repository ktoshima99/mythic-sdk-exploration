"""nuScenes mAP benchmark for BEVFormer-style multi-view 3D detection.

Pure-numpy implementation of the mAP portion of the nuScenes detection metric:
BEV-center-distance matching at four distance thresholds (0.5, 1.0, 2.0, 4.0 m),
averaged across the 10 detection classes. No nuscenes-devkit / shapely / scipy
dependencies — matches the algorithm in `nuscenes.eval.detection.algo`
(`accumulate` + `calc_ap`).

NDS and the TP metrics (ATE/ASE/AOE/AVE/AAE) are out of scope.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vnnort.data.container import (
    InputData,
    MultiViewDetection3DInput,
    MultiViewDetection3DOutput,
    OutputData,
)
from vnnort.data.datasets.nuscenes import NUSCENES_DETECTION_CLASSES
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

logger = logging.getLogger(__name__)

# nuScenes detection challenge constants (see nuscenes.eval.detection.constants).
DIST_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
N_RECALL_BINS: int = 101  # DetectionMetricData.nelem in the devkit
MIN_RECALL: float = 0.1
MIN_PRECISION: float = 0.1


def _calc_ap(precision_at_recall_bins: NDArray[np.float64]) -> float:
    """Integrate AP over recall ∈ (MIN_RECALL, 1] with a precision floor of MIN_PRECISION.

    Mirrors `nuscenes.eval.detection.algo.calc_ap`: drop the first `round(100 *
    MIN_RECALL) + 1` recall bins (here, 11), subtract the precision floor and
    clip at zero, average the remaining 90 bins, divide by `(1 - MIN_PRECISION)`.
    """
    keep_from = round(100 * MIN_RECALL) + 1
    cropped = precision_at_recall_bins[keep_from:].astype(np.float64)
    cropped = np.maximum(cropped - MIN_PRECISION, 0.0)
    return float(cropped.mean()) / (1.0 - MIN_PRECISION)


def _accumulate_per_class(
    pred_records: list[tuple[float, NDArray[np.float32], str]],
    gt_centers_by_sample: dict[str, NDArray[np.float32]],
    dist_th: float,
) -> NDArray[np.float64]:
    """Greedy BEV-center matching → interpolated precision at 101 recall bins.

    `pred_records` is a list of (score, center_xy, sample_token) for a single class
    across all samples. `gt_centers_by_sample` maps sample_token → (G, 2) GT centers
    for the same class.

    Matching: walk predictions in descending score order; each prediction takes the
    nearest unmatched same-sample same-class GT within `dist_th` meters in BEV. If
    no GT is within the threshold, the prediction is a false positive.
    """
    npos = sum(int(arr.shape[0]) for arr in gt_centers_by_sample.values())
    if npos == 0 or len(pred_records) == 0:
        return np.zeros(N_RECALL_BINS, dtype=np.float64)

    pred_records_sorted = sorted(pred_records, key=lambda r: -r[0])

    taken: dict[str, NDArray[np.bool_]] = {
        token: np.zeros(arr.shape[0], dtype=bool) for token, arr in gt_centers_by_sample.items()
    }

    tp_flags = np.zeros(len(pred_records_sorted), dtype=np.float64)
    fp_flags = np.zeros(len(pred_records_sorted), dtype=np.float64)
    for i, (_score, center, token) in enumerate(pred_records_sorted):
        gt_centers = gt_centers_by_sample.get(token)
        if gt_centers is None or gt_centers.shape[0] == 0:
            fp_flags[i] = 1.0
            continue
        dists = np.linalg.norm(gt_centers - center[None, :], axis=1)
        dists_avail = np.where(taken[token], np.inf, dists)
        best = int(np.argmin(dists_avail))
        if dists_avail[best] < dist_th:
            tp_flags[i] = 1.0
            taken[token][best] = True
        else:
            fp_flags[i] = 1.0

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)
    rec = tp_cum / float(npos)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    rec_interp = np.linspace(0.0, 1.0, N_RECALL_BINS)
    interp: NDArray[np.float64] = np.interp(rec_interp, rec, prec, right=0.0)
    return interp


class BevformerBenchmark(BenchmarkBase):
    """nuScenes mAP for surround-view 3D detection.

    Accumulates per-(class, sample) prediction and GT BEV centers via `update`, and
    computes mAP via BEV-center matching at 4 distance thresholds in `compute`.
    Predictions and GT must be in the same frame (LiDAR for BEVFormer).
    """

    def setup_and_reset(self) -> None:
        """Reset all accumulators (called before each `run`)."""
        # class_idx → sample_token → (G, 2) centers
        self._gt_centers: dict[int, dict[str, list[NDArray[np.float32]]]] = defaultdict(lambda: defaultdict(list))
        # class_idx → list of (score, center_xy, sample_token)
        self._pred_records: dict[int, list[tuple[float, NDArray[np.float32], str]]] = defaultdict(list)

    def update(self, input_data: list[InputData], output_data: list[OutputData]) -> None:  # type: ignore[override]
        """Stash per-sample predictions and GT for later mAP computation."""
        for inp, out in zip(input_data, output_data):
            if not isinstance(inp, MultiViewDetection3DInput):
                raise TypeError(f"input_data entries must be MultiViewDetection3DInput, got {type(inp).__name__}")
            if not isinstance(out, MultiViewDetection3DOutput):
                raise TypeError(f"output_data entries must be MultiViewDetection3DOutput, got {type(out).__name__}")
            token = inp.sample_token

            if inp.gt_boxes is not None and inp.gt_labels is not None and inp.gt_boxes.shape[0] > 0:
                gt_centers = inp.gt_boxes[:, :2].astype(np.float32, copy=False)
                gt_labels = inp.gt_labels.astype(np.int64, copy=False)
                for class_id in np.unique(gt_labels):
                    mask = gt_labels == class_id
                    self._gt_centers[int(class_id)][token].append(gt_centers[mask])

            if out.boxes.shape[0] > 0:
                pred_centers = out.boxes[:, :2].astype(np.float32, copy=False)
                pred_scores = out.scores.astype(np.float32, copy=False)
                pred_labels = out.labels.astype(np.int64, copy=False)
                for i in range(out.boxes.shape[0]):
                    self._pred_records[int(pred_labels[i])].append(
                        (float(pred_scores[i]), pred_centers[i].copy(), token)
                    )

    def compute(self) -> dict[str, Any]:
        """Compute mAP plus per-class and per-(class, threshold) AP breakdowns."""
        per_class_per_threshold_ap: dict[str, dict[float, float]] = {}
        per_class_ap: dict[str, float] = {}

        for class_idx, class_name in enumerate(NUSCENES_DETECTION_CLASSES):
            gt_lists = self._gt_centers.get(class_idx, {})
            gt_by_sample = {token: np.concatenate(lst, axis=0) for token, lst in gt_lists.items() if lst}
            pred_records = self._pred_records.get(class_idx, [])

            aps: dict[float, float] = {}
            for dist_th in DIST_THRESHOLDS:
                prec_interp = _accumulate_per_class(pred_records, gt_by_sample, dist_th)
                aps[dist_th] = _calc_ap(prec_interp)
            per_class_per_threshold_ap[class_name] = aps
            per_class_ap[class_name] = float(np.mean(list(aps.values())))

        mAP = float(np.mean(list(per_class_ap.values())))
        return {
            "mAP": mAP,
            "per_class_ap": per_class_ap,
            "per_class_per_threshold_ap": per_class_per_threshold_ap,
        }
