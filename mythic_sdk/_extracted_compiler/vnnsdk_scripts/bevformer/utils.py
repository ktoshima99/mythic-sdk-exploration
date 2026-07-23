import numpy as np
import PIL.Image
from numpy.typing import NDArray

from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import MultiViewDetection3DInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase


# nuScenes circle-NMS distance thresholds (per class), from BEVFormer's
# bbox_coder config. Class order matches NUSCENES_DETECTION_CLASSES.
_NMS_DIST: dict[int, float] = {
    0: 2.0,  # car
    1: 3.0,  # truck
    2: 2.5,  # construction_vehicle
    3: 4.0,  # bus
    4: 3.0,  # trailer
    5: 1.0,  # barrier
    6: 1.5,  # motorcycle
    7: 1.0,  # bicycle
    8: 0.5,  # pedestrian
    9: 0.3,  # traffic_cone
}


def _denormalize_bbox(bboxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Decode raw 10-column bbox predictions to (cx, cy, cz, w, l, h, yaw, vx, vy).

    Mirrors `projects/mmdet3d_plugin/core/bbox/util.denormalize_bbox` in pure numpy.
    Input column order is (cx, cy, log_w, log_l, cz, log_h, sin_yaw, cos_yaw, vx, vy).
    """
    rot = np.arctan2(bboxes[..., 6:7], bboxes[..., 7:8])
    cx, cy, cz = bboxes[..., 0:1], bboxes[..., 1:2], bboxes[..., 4:5]
    w = np.exp(bboxes[..., 2:3])
    length = np.exp(bboxes[..., 3:4])
    h = np.exp(bboxes[..., 5:6])
    vx, vy = bboxes[..., 8:9], bboxes[..., 9:10]
    return np.concatenate([cx, cy, cz, w, length, h, rot, vx, vy], axis=-1)


def _circle_nms(
    bboxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    labels: NDArray[np.int64],
    dist_thrs: dict[int, float],
) -> NDArray[np.int64]:
    """Per-class circular NMS on BEV centers — same algorithm as the reference."""
    if len(bboxes) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(scores)[::-1]
    bboxes_s = bboxes[order]
    labels_s = labels[order]
    pts = bboxes_s[:, :2]
    suppressed = np.zeros(len(bboxes), dtype=bool)
    keep: list[int] = []
    for i in range(len(bboxes)):
        if suppressed[i]:
            continue
        keep.append(int(order[i]))
        radius = dist_thrs.get(int(labels_s[i]), 1.0)
        if i + 1 < len(bboxes):
            dists = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
            mask = (dists < radius) & (labels_s[i + 1 :] == labels_s[i])
            suppressed[i + 1 :][mask] = True
    return np.array(keep, dtype=np.int64)


def _post_process(
    all_cls_scores: NDArray[np.float32],
    all_bbox_preds: NDArray[np.float32],
    *,
    post_center_range: tuple[float, ...],
    max_num: int,
    num_classes: int,
    score_thr: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """BEVFormerHead.get_bboxes equivalent — decode, threshold, NMS, finalize.

    The BEVFormer head already maps cx/cy/cz through sigmoid + pc_range inside
    `reg_branches`, so the ONNX outputs centers in physical (LiDAR) coordinates;
    no further pc_range scaling is required here.

    Args:
        all_cls_scores (NDArray[np.float32]): (num_layers, B, num_query, num_classes)
            decoder-first class logits.
        all_bbox_preds (NDArray[np.float32]): (num_layers, B, num_query, 10) decoder-first,
            with (cx, cy, log_w, log_l, cz, log_h, sin_yaw, cos_yaw, vx, vy).
        post_center_range (tuple[float, ...]): Filter boxes whose center falls outside this box.
        max_num (int): Top-k cap before NMS.
        num_classes (int): Width of the class dimension.
        score_thr (float): Minimum sigmoid score after top-k.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
            `(boxes, scores, labels)` with boxes shaped (N, 9) =
            (cx, cy, cz, w, l, h, yaw, vx, vy), and scores/labels each (N,).
    """
    cls = all_cls_scores[-1][0]  # (num_query, num_classes)
    bpd = all_bbox_preds[-1][0]  # (num_query, 10)

    cls_sig = 1.0 / (1.0 + np.exp(-cls))  # sigmoid
    flat = cls_sig.reshape(-1)
    topk = np.argsort(flat)[::-1][:max_num]

    scores = flat[topk]
    labels = (topk % num_classes).astype(np.int64)
    boxes = _denormalize_bbox(bpd[topk // num_classes])

    pcr_post = np.asarray(post_center_range, dtype=np.float32)
    in_range = (boxes[:, :3] >= pcr_post[:3]).all(1) & (boxes[:, :3] <= pcr_post[3:]).all(1)
    mask = in_range & (scores > score_thr)
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    if len(scores) == 0:
        return (
            np.zeros((0, 9), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    keep = _circle_nms(boxes, scores, labels, _NMS_DIST)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Adjust z to box bottom and shrink dims slightly (matches reference final step).
    boxes[:, 2] -= boxes[:, 5] * 0.5
    boxes[:, 3:6] *= 0.9

    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)


class RandomNuscenesBevformerDataset(DatasetBase):
    """Eval-only stand-in for `NuscenesBevformerDataset` with random in-memory samples.

    Emits `MultiViewDetection3DInput` instances whose field shapes and dtypes match the
    real dataset, so the BEVFormer pipeline can run end-to-end without nuScenes on disk.
    """

    N_CAMERAS = 6
    IMAGE_HEIGHT = 900
    IMAGE_WIDTH = 1600

    def __init__(self, split: str = "val", length: int = 8, seed: int = 0) -> None:
        """Initialize a deterministic random dataset for the requested split."""
        if split not in {"val", "train"}:
            raise ValueError(f"split must be 'val' or 'train'; got {split!r}")
        self.split = split
        self._length = length
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of synthetic samples."""
        return self._length

    def __getitem__(self, index: int) -> MultiViewDetection3DInput:
        """Return a synthetic 6-camera sample."""
        if not 0 <= index < self._length:
            raise IndexError(index)

        images = [
            PIL.Image.fromarray(
                self._rng.integers(0, 256, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3), dtype=np.uint8),
                mode="RGB",
            )
            for _ in range(self.N_CAMERAS)
        ]
        lidar2img = self._rng.standard_normal((self.N_CAMERAS, 4, 4)).astype(np.float32)
        can_bus = self._rng.standard_normal(18).astype(np.float32)

        m = int(self._rng.integers(1, 8))
        gt_boxes = self._rng.standard_normal((m, 9)).astype(np.float32)
        gt_labels = self._rng.integers(0, 10, m, dtype=np.int32)
        gt_names = ["car"] * m

        return MultiViewDetection3DInput(
            images=images,
            lidar2img=lidar2img,
            can_bus=can_bus,
            is_first_in_scene=(index == 0),
            sample_token=f"random_{index}",
            scene_token="random_scene",
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_names=gt_names,
        )

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return the benchmark class used to evaluate this dataset."""
        from vnnort.inference.evaluation.bevformer_3d import BevformerBenchmark

        return BevformerBenchmark
