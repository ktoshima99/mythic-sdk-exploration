"""nuScenes dataset for BEVFormer multi-view 3D detection.

Self-contained alternative to the mmdet3d / mmcv dataset path used by the original
BEVFormer ONNX eval (see model_zoo/bevformer/bevformer_lib/onnx_eval.py and
projects/mmdet3d_plugin/datasets/nuscenes_dataset.py). Everything that is invariant
across BEVFormer-tiny/-small/-base lives here; the variant-specific resize /
normalize / pad / lidar2img-row scaling is the responsibility of the per-variant
preprocess.
"""

import logging
import os
import pickle
from typing import Any

import numpy as np
import PIL.Image
from numpy.typing import NDArray

from vnnort import get_env_variable
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import MultiViewDetection3DInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

logger = logging.getLogger(__name__)

VNNORT_NUSCENES_PATH = get_env_variable("VNNORT_NUSCENES_PATH")

# Camera order matches the BEVFormer reference (custom_utils.data_loading.CAMERA_TYPES).
CAMERA_TYPES: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# Class order MUST match BEVFormer training (`class_names` in its configs/_base_/
# datasets/nus-3d.py and bevformer_tiny.py). The model's `outputs_classes` channel
# axis is aligned with this ordering, so any divergence here silently misaligns
# predictions and GT in the benchmark. Note this differs from mmdet3d's default
# NuScenesDataset.CLASSES tuple (which swaps several pairs).
NUSCENES_DETECTION_CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
_NAME_TO_LABEL: dict[str, int] = {name: i for i, name in enumerate(NUSCENES_DETECTION_CLASSES)}

_VALID_SPLITS = {"val", "train"}


def _resolve_pkl_path(path_to_dataset: str, split: str) -> str:
    """Find the BEVFormer-style info pkl under `path_to_dataset` for the given split.

    Probes the two filename conventions in use: `nuscenes_infos_temporal_{split}.pkl`
    (full v1.0-trainval) and `nuscenes_mini_infos_temporal_{split}.pkl` (v1.0-mini).
    Returns the first one that exists.
    """
    candidates = [
        f"nuscenes_infos_temporal_{split}.pkl",
        f"nuscenes_mini_infos_temporal_{split}.pkl",
    ]
    for name in candidates:
        path = os.path.join(path_to_dataset, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No info pkl found under {path_to_dataset!r} for split={split!r}; " f"looked for {candidates}"
    )


def _quaternion_yaw(quat_wxyz: NDArray[np.float64]) -> float:
    """Yaw (radians) of a quaternion in (w, x, y, z) order.

    Matches nuscenes.eval.common.utils.quaternion_yaw: rotates the unit x-vector by
    the quaternion and atan2's the resulting xy components. Equivalent without
    pulling in pyquaternion / nuscenes-devkit.
    """
    w, x, y, z = (float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
    rx = 1.0 - 2.0 * (y * y + z * z)
    ry = 2.0 * (x * y + w * z)
    return float(np.arctan2(ry, rx))


def _compute_absolute_can_bus(info: dict[str, Any]) -> NDArray[np.float32]:
    """Overwrite the raw pkl can_bus with ego2global pose / yaw.

    Mirrors the in-place mutation that CustomNuScenesDataset.get_data_info performs
    on the can_bus right before BEVFormer consumes it.
    """
    can_bus = np.array(info["can_bus"], dtype=np.float64).copy()
    can_bus[:3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    rotation = np.asarray(info["ego2global_rotation"], dtype=np.float64).reshape(-1)
    can_bus[3:7] = rotation
    yaw_rad = _quaternion_yaw(rotation)
    yaw_deg = yaw_rad / np.pi * 180.0
    if yaw_deg < 0.0:
        yaw_deg += 360.0
    can_bus[-2] = yaw_deg / 180.0 * np.pi
    can_bus[-1] = yaw_deg
    return can_bus.astype(np.float32)


def _rebase_data_path(data_path: str, path_to_dataset: str) -> str:
    """Rewrite a pkl-stored image path to live under our `path_to_dataset`.

    nuScenes info pkls store `data_path` as whatever was relative at create-time
    (e.g. `./data/nuscenes/samples/CAM_FRONT/x.jpg`). Anchor on the canonical
    `samples/` or `sweeps/` directory and rejoin against the local root so the
    path resolves regardless of where the pkl was generated.
    """
    norm = data_path.replace("\\", "/")
    for anchor in ("samples/", "sweeps/"):
        idx = norm.find(anchor)
        if idx >= 0:
            return os.path.join(path_to_dataset, norm[idx:])
    if os.path.isabs(data_path):
        return data_path
    return os.path.join(path_to_dataset, data_path)


def _build_lidar2img(cam_info: dict[str, Any]) -> NDArray[np.float64]:
    """Per-camera 4x4 lidar2img matrix, identical to the BEVFormer reference math."""
    sensor2lidar_rotation = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64)
    sensor2lidar_translation = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64)
    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


class NuscenesBevformerDataset(DatasetBase):
    """nuScenes dataset for BEVFormer-style surround-view 3D detection.

    Reads a BEVFormer-style temporal info pkl (`nuscenes_infos_temporal_{split}.pkl`)
    from `path_to_dataset` and emits `MultiViewDetection3DInput` samples that match
    the input contract of the BEVFormer ONNX model without any mmdet/mmcv/mmdet3d
    dependency.

    The dataset assumes sequential iteration: it pre-bakes the can_bus delta versus
    the previous in-scene frame and exposes `is_first_in_scene` so downstream code
    can reset BEV-temporal state (`prev_bev`, `use_prev_bev`).
    """

    def __init__(self, path_to_dataset: str = VNNORT_NUSCENES_PATH, split: str = "val") -> None:
        """Load samples from `path_to_dataset/nuscenes_infos_temporal_{split}.pkl`.

        Args:
            path_to_dataset (str): Root directory containing the info pkl (and typically the
                raw `samples/` folder referenced by `cam_info['data_path']`).
            split (str): One of `'val'`, `'train'`.

        Raises:
            ValueError: If `split` is not one of the allowed values, or if the pkl
                payload has an unexpected shape.
        """
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}; got {split!r}")

        pkl_path = _resolve_pkl_path(path_to_dataset, split)
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "infos" in payload:
            data_infos = list(payload["infos"])
        elif isinstance(payload, list):
            data_infos = list(payload)
        else:
            raise ValueError(f"Unexpected pkl payload at {pkl_path}: {type(payload).__name__}")

        # NuScenesDataset.load_annotations sorts by timestamp so scene frames are contiguous.
        data_infos.sort(key=lambda x: int(x["timestamp"]))

        self.path_to_dataset = path_to_dataset
        self.split = split
        self.data_infos = data_infos

        n = len(data_infos)
        self._is_first_in_scene: list[bool] = [True] * n
        self._prev_pos: list[NDArray[np.float32]] = [np.zeros(3, dtype=np.float32) for _ in range(n)]
        self._prev_yaw_deg: list[float] = [0.0] * n

        prev_pos: NDArray[np.float32] = np.zeros(3, dtype=np.float32)
        prev_yaw_deg = 0.0
        prev_scene_token: str | None = None
        for i, info in enumerate(data_infos):
            scene_token = str(info["scene_token"])
            same_scene = scene_token == prev_scene_token
            self._is_first_in_scene[i] = not same_scene
            if same_scene:
                self._prev_pos[i] = prev_pos
                self._prev_yaw_deg[i] = prev_yaw_deg

            cb_abs = _compute_absolute_can_bus(info)
            prev_pos = cb_abs[:3].astype(np.float32).copy()
            prev_yaw_deg = float(cb_abs[-1])
            prev_scene_token = scene_token

    def __len__(self) -> int:
        """Return the number of samples loaded from the pkl."""
        return len(self.data_infos)

    def __getitem__(self, index: int) -> MultiViewDetection3DInput:
        """Build a `MultiViewDetection3DInput` for the given sample index."""
        info = self.data_infos[index]

        images: list[PIL.Image.Image] = []
        lidar2img_per_cam: list[NDArray[np.float64]] = []
        for cam in CAMERA_TYPES:
            cam_info = info["cams"][cam]
            data_path = _rebase_data_path(cam_info["data_path"], self.path_to_dataset)
            images.append(PIL.Image.open(data_path).convert("RGB"))
            lidar2img_per_cam.append(_build_lidar2img(cam_info))

        lidar2img = np.stack(lidar2img_per_cam, axis=0).astype(np.float32)

        can_bus = _compute_absolute_can_bus(info)
        if self._is_first_in_scene[index]:
            can_bus[:3] = 0.0
            can_bus[-1] = 0.0
        else:
            can_bus[:3] -= self._prev_pos[index]
            can_bus[-1] -= self._prev_yaw_deg[index]

        gt_boxes, gt_labels, gt_names = self._extract_gt(info)

        return MultiViewDetection3DInput(
            images=images,
            lidar2img=lidar2img,
            can_bus=can_bus,
            is_first_in_scene=bool(self._is_first_in_scene[index]),
            sample_token=str(info["token"]),
            scene_token=str(info["scene_token"]),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_names=gt_names,
        )

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return the (stub) benchmark class for BEVFormer-style 3D detection."""
        from vnnort.inference.evaluation.bevformer_3d import BevformerBenchmark

        return BevformerBenchmark

    @staticmethod
    def _extract_gt(
        info: dict[str, Any],
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.int32] | None, list[str] | None]:
        """Parse the pkl's gt_boxes / gt_names into the InputData fields.

        Drops boxes whose class name is not part of the BEVFormer 10-class detection
        set, matching the filtering NuScenesDataset.get_ann_info applies.
        Returns `(None, None, None)` when no GT is present (e.g. test split).
        """
        gt_boxes_raw = info.get("gt_boxes")
        gt_names_raw = info.get("gt_names")
        if gt_boxes_raw is None or gt_names_raw is None or len(gt_boxes_raw) == 0:
            return None, None, None

        gt_boxes = np.asarray(gt_boxes_raw, dtype=np.float32)
        # BEVFormer's get_ann_info concatenates the per-box velocity (vx, vy) onto the
        # (x, y, z, w, l, h, yaw) box, yielding the (M, 9) format the model expects.
        if gt_boxes.shape[1] == 7:
            gt_velocity = np.asarray(info.get("gt_velocity"), dtype=np.float32)
            if gt_velocity.shape != (gt_boxes.shape[0], 2):
                raise ValueError(f"gt_velocity shape {gt_velocity.shape} does not match gt_boxes {gt_boxes.shape}")
            nan_mask = np.isnan(gt_velocity).any(axis=1)
            gt_velocity[nan_mask] = 0.0
            gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=1)
        gt_names = [str(n) for n in gt_names_raw]
        gt_labels = np.array([_NAME_TO_LABEL.get(n, -1) for n in gt_names], dtype=np.int32)
        keep = gt_labels >= 0
        if not bool(keep.all()):
            gt_boxes = gt_boxes[keep]
            gt_names = [n for n, k in zip(gt_names, keep) if k]
            gt_labels = gt_labels[keep]

        if gt_boxes.shape[0] == 0:
            return None, None, None
        return gt_boxes, gt_labels, gt_names
