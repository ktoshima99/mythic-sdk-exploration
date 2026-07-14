from __future__ import annotations

from pathlib import Path


import cv2
import numpy as np

from .data_loading import unwrap_dc
from .processing import denormalize_image
from .visualization import _draw_boxes_on_image, visualize_frame
from rich.console import Console

console = Console()

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "configs"
    / "bevformer"
    / "bevformer_tiny.py"
)
GT_RESULTS_DIR = "gt-reference-results"



def inject_ann_info_for_gt(dataset) -> None:
    """Monkey-patch dataset.prepare_test_data to inject ann_info before the pipeline.

    The vanilla test pipeline calls get_data_info but not get_ann_info, so
    LoadAnnotations3D finds no 'ann_info' key and raises KeyError.  This patch
    mirrors the extra steps from prepare_train_data without touching any shared code.
    """
    def _patched(idx: int):
        input_dict = dataset.get_data_info(idx)
        if input_dict is None:
            return None
        dataset.pre_pipeline(input_dict)
        input_dict["ann_info"] = dataset.get_ann_info(idx)
        return dataset.pipeline(input_dict)

    dataset.prepare_test_data = _patched


def patch_pipeline_for_gt(cfg) -> None:
    """Mutate cfg.data.test pipeline in-place to include GT annotations.

    Inserts LoadAnnotations3D and updates DefaultFormatBundle3D / CustomCollect3D
    so that gt_bboxes_3d and gt_labels_3d flow through the test dataloader.
    """
    test_cfg = cfg.data.test

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _set(obj, key, val):
        if isinstance(obj, dict):
            obj[key] = val
        else:
            setattr(obj, key, val)

    pipeline = _get(test_cfg, "pipeline", [])

    # 1. Insert LoadAnnotations3D after LoadMultiViewImageFromFiles
    insert_idx = None
    for i, t in enumerate(pipeline):
        if _get(t, "type") == "LoadMultiViewImageFromFiles":
            insert_idx = i + 1
            break
    if insert_idx is not None:
        pipeline.insert(
            insert_idx,
            dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
        )

    # 2. Patch DefaultFormatBundle3D and CustomCollect3D inside MultiScaleFlipAug3D
    for t in pipeline:
        if _get(t, "type") != "MultiScaleFlipAug3D":
            continue
        for inner in _get(t, "transforms", []):
            inner_type = _get(inner, "type")
            if inner_type == "DefaultFormatBundle3D":
                _set(inner, "with_label", True)
            elif inner_type == "CustomCollect3D":
                _set(inner, "keys", ["img", "gt_bboxes_3d", "gt_labels_3d"])
        break


# ── GT extraction ─────────────────────────────────────────────────────────────


def extract_gt_result(data: dict) -> dict[str, np.ndarray]:
    """Build a visualize_frame-compatible result dict from dataloader GT fields.

    Returns boxes_3d (N, 9) in bottom-centre convention, scores_3d ones, labels_3d.
    """
    empty = dict(
        boxes_3d=np.zeros((0, 9), dtype=np.float32),
        scores_3d=np.zeros((0,), dtype=np.float32),
        labels_3d=np.zeros((0,), dtype=np.int64),
    )
    gt_boxes_raw = data.get("gt_bboxes_3d")
    gt_labels_raw = data.get("gt_labels_3d")
    if gt_boxes_raw is None or gt_labels_raw is None:
        return empty

    boxes_list = unwrap_dc(gt_boxes_raw)
    labels_list = unwrap_dc(gt_labels_raw)

    boxes_obj = boxes_list[0]   # LiDARInstance3DBoxes
    labels_t = labels_list[0]   # LongTensor

    boxes_np = (
        boxes_obj.tensor.cpu().numpy().astype(np.float32)
        if hasattr(boxes_obj, "tensor")
        else np.array(boxes_obj, dtype=np.float32)
    )
    labels_np = (
        labels_t.cpu().numpy().astype(np.int64)
        if hasattr(labels_t, "numpy")
        else np.array(labels_t, dtype=np.int64)
    )

    # Drop ignored annotations
    valid = labels_np >= 0
    boxes_np = boxes_np[valid]
    labels_np = labels_np[valid]

    if len(boxes_np) == 0:
        return empty

    # Pad to (N, 9) — GT boxes are (N, 7); fill vx/vy with zeros
    if boxes_np.shape[1] < 9:
        boxes_np = np.concatenate(
            [boxes_np, np.zeros((len(boxes_np), 9 - boxes_np.shape[1]), dtype=np.float32)],
            axis=1,
        )

    return dict(
        boxes_3d=boxes_np,
        scores_3d=np.ones(len(boxes_np), dtype=np.float32),
        labels_3d=labels_np,
    )


# ── Visualization wrappers ────────────────────────────────────────────────────


def build_cam_grid(
    img_np: np.ndarray,
    result: dict[str, np.ndarray],
    lidar2img: np.ndarray,
    img_norm: dict,
) -> np.ndarray:
    """Render 6-camera grid with GT boxes, without the BEV inset."""
    n_cams = img_np.shape[1]
    raw_imgs = [denormalize_image(img_np[0, c], img_norm) for c in range(n_cams)]
    boxes = result.get("boxes_3d", np.zeros((0, 9), dtype=np.float32))
    labels = result.get("labels_3d", np.zeros((0,), dtype=np.int64))

    cam_imgs = []
    for cam_idx, raw in enumerate(raw_imgs):
        if len(boxes) > 0 and lidar2img.shape[1] > cam_idx:
            raw = _draw_boxes_on_image(raw, boxes, labels, lidar2img[0, cam_idx])
        cam_imgs.append(raw)

    if len(cam_imgs) == 6:
        ordered = [
            cam_imgs[5], cam_imgs[0], cam_imgs[1],
            cv2.flip(cam_imgs[2], 1), cv2.flip(cam_imgs[3], 1), cv2.flip(cam_imgs[4], 1),
        ]
        return np.vstack([np.hstack(ordered[:3]), np.hstack(ordered[3:])])
    if cam_imgs:
        return np.hstack(cam_imgs)
    return np.full((480, 800, 3), 40, dtype=np.uint8)


def _render(
    img_np: np.ndarray,
    result: dict[str, np.ndarray],
    lidar2img: np.ndarray,
    img_norm: dict,
    pc_range: list[float],
    *,
    pad_with_shape: bool,
) -> np.ndarray:
    if pad_with_shape:
        return visualize_frame(img_np, result, lidar2img, img_norm, pc_range)
    return build_cam_grid(img_np, result, lidar2img, img_norm)

