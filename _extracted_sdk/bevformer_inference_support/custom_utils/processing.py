"""BEVFormer image pre/post-processing utilities.

Covers: per-class constants, InferenceConfig, config parsing, CLI argument
parsers, image normalization helpers, bbox post-processing, and NMS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import typer
from mmcv import Config
from mmcv.image import imdenormalize, imnormalize

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),
    1: (200, 200, 0),
    2: (0, 0, 200),
    3: (0, 140, 200),
    4: (200, 0, 200),
    5: (0, 200, 200),
    6: (100, 0, 128),
    7: (200, 130, 0),
    8: (50, 50, 200),
    9: (120, 120, 120),
}

# NMS distance thresholds per class (in metres)
_NMS_DIST: dict[int, float] = {
    0: 2.0,
    1: 3.0,
    2: 2.5,
    3: 4.0,
    4: 3.0,
    5: 1.0,
    6: 1.5,
    7: 1.0,
    8: 0.5,
    9: 0.3,
}


# ── Config dataclass ──────────────────────────────────────────────────────────


@dataclass()
class InferenceConfig:
    """Model and post-processing parameters parsed from an mmdet3d .py config."""

    bev_h: int
    bev_w: int
    embed_dims: int
    pc_range: list[float]
    post_center_range: list[float]
    max_num: int
    num_classes: int
    class_names: list[str]
    img_norm: dict
    input_modality: dict[str, bool]

    def modality_use(self, key: str) -> bool:
        """``cfg.input_modality`` access (e.g. ``use_lidar``, ``use_radar``, ``use_map``)."""
        return bool(self.input_modality.get(key, False))


def parse_config_py(config_path: Path) -> tuple[InferenceConfig, Config]:
    """Load an mmdet .py config (e.g. bevformer_tiny.py) and derive an InferenceConfig.

    Returns ``(infer_cfg, mmcfg)`` so callers that need the raw mmcv ``Config``
    (e.g. for the dataloader) don't have to re-parse it.
    """
    from .data_loading import load_mmcv_config

    cfg = load_mmcv_config(config_path)
    pts = cfg.model.pts_bbox_head
    im: dict[str, bool] = {}
    if hasattr(cfg, "input_modality") and cfg.input_modality is not None:
        im = {str(k): bool(v) for k, v in dict(cfg.input_modality).items()}
    return InferenceConfig(
        bev_h=int(pts.bev_h),
        bev_w=int(pts.bev_w),
        embed_dims=int(pts.transformer.embed_dims),
        pc_range=list(cfg.point_cloud_range),
        post_center_range=list(pts.bbox_coder.post_center_range),
        max_num=int(pts.bbox_coder.max_num),
        num_classes=int(pts.num_classes),
        class_names=list(cfg.class_names),
        img_norm=dict(cfg.img_norm_cfg),
        input_modality=im,
    ), cfg


# ── CLI option parsers ────────────────────────────────────────────────────────


def _parse_int_tuple(
    s: str | None, n: int, option: str, fmt: str
) -> tuple[int, ...] | None:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != n:
        raise typer.BadParameter(f"{option} must be in format '{fmt}'")
    return tuple(int(p) for p in parts)


def parse_crop(crop_str: str | None) -> tuple[int, int, int, int] | None:
    """Parse 'x,y,w,h' string into a crop tuple."""
    return _parse_int_tuple(crop_str, 4, "--crop", "x,y,w,h")


def parse_resize(resize_str: str | None) -> tuple[int, int] | None:
    """Parse 'w,h' string into a resize tuple."""
    return _parse_int_tuple(resize_str, 2, "--resize", "w,h")


# ── Image preprocessing ───────────────────────────────────────────────────────


def _img_norm_arrays(img_norm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) as float32 arrays from img_norm dict."""
    mean = np.array(img_norm["mean"], dtype=np.float32)
    std = np.array(img_norm["std"], dtype=np.float32)
    return mean, std


def preprocess_image(
    img_bgr: np.ndarray,
    img_norm: dict,
    crop: tuple[int, int, int, int] | None = None,
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Preprocess a BGR image: optional crop → optional resize → normalize (mmcv).

    Args:
        img_bgr: (H, W, 3) uint8 BGR array.
        img_norm: dict with keys 'mean', 'std', 'to_rgb'.
        crop: (x, y, w, h) pixel crop applied first.
        resize: (target_w, target_h) resize applied after crop.

    Returns:
        (3, H, W) float32 normalized array.
    """
    if crop is not None:
        x, y, w, h = crop
        img_bgr = img_bgr[y : y + h, x : x + w]

    if resize is not None:
        tw, th = resize
        if img_bgr.shape[:2] != (th, tw):
            img_bgr = cv2.resize(img_bgr, (tw, th))

    img = img_bgr.astype(np.float32)
    mean, std = _img_norm_arrays(img_norm)
    to_rgb = bool(img_norm["to_rgb"])
    img = imnormalize(img, mean, std, to_rgb=to_rgb)
    return img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)


def denormalize_image(img_chw: np.ndarray, img_norm: dict) -> np.ndarray:
    """Inverse of preprocess_image: (C, H, W) float → (H, W, 3) uint8 BGR (mmcv)."""
    mean, std = _img_norm_arrays(img_norm)
    to_rgb = bool(img_norm["to_rgb"])
    img = img_chw.transpose(1, 2, 0).astype(np.float32)
    img = imdenormalize(img, mean, std, to_bgr=to_rgb)
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_lidar2img_for_crop_resize(
    lidar2img: np.ndarray,
    *,
    crop: tuple[int, int, int, int] | None,
    resize: tuple[int, int] | None,
    tensor_hw: tuple[int, int],
) -> np.ndarray:
    """Apply the same 2D pixel affine as :func:`preprocess_image` (crop → resize) to ``lidar2img``.

    Mirrors ``CropResizeFlipImage`` / nuScenes: ``cam2img[:3,:3] = ida @ cam2img[:3,:3]`` so
    3D→2D projection matches the transformed image tensor.

    Args:
        lidar2img: ``(1, N, 4, 4)`` or ``(N, 4, 4)``.
        crop: ``(x, y, w, h)`` in pixels on the tensor **before** crop, or ``None``.
        resize: target ``(width, height)`` after crop, or ``None`` to keep crop size.
        tensor_hw: ``(H, W)`` per-camera plane of the batch **before** crop/resize.
    """
    if crop is None and resize is None:
        return lidar2img
    H, W = int(tensor_hw[0]), int(tensor_hw[1])
    x, y = 0, 0
    w, h = W, H
    if crop is not None:
        x, y, w, h = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
    tw, th = w, h
    if resize is not None:
        tw, th = int(resize[0]), int(resize[1])
    r_x = tw / max(w, 1)
    r_y = th / max(h, 1)
    ida = np.eye(3, dtype=np.float64)
    ida[0, 0] = r_x
    ida[1, 1] = r_y
    ida[0, 2] = -float(x) * r_x
    ida[1, 2] = -float(y) * r_y
    ida_f = ida.astype(np.float32)
    arr = np.asarray(lidar2img, dtype=np.float32).copy()
    if arr.ndim == 2:
        arr = arr[np.newaxis, np.newaxis]
    elif arr.ndim == 3:
        arr = arr[np.newaxis]
    for i in range(arr.shape[1]):
        arr[0, i, :3, :] = ida_f @ arr[0, i, :3, :]
    return arr


def apply_crop_resize_to_batch(
    img_np: np.ndarray,
    img_norm: dict,
    crop: tuple[int, int, int, int] | None,
    resize: tuple[int, int] | None,
) -> np.ndarray:
    """Apply crop+resize to a (1, N, C, H, W) batch by round-tripping through pixels."""
    if crop is None and resize is None:
        return img_np
    out = []
    for cam in range(img_np.shape[1]):
        bgr = denormalize_image(img_np[0, cam], img_norm)
        proc = preprocess_image(bgr, img_norm, crop=crop, resize=resize)
        out.append(proc)
    return np.stack(out, axis=0)[np.newaxis]  # (1, N, C, H, W)


# ── Post-processing ───────────────────────────────────────────────────────────


# Logic mirrors projects/mmdet3d_plugin/core/bbox/util.py::denormalize_bbox;
# numpy impl here so ONNX path does not depend on torch.
def _denormalize_bbox(bboxes: np.ndarray) -> np.ndarray:
    """Decode raw model bbox predictions to real-world coordinates."""
    rot = np.arctan2(bboxes[..., 6:7], bboxes[..., 7:8])
    cx, cy, cz = bboxes[..., 0:1], bboxes[..., 1:2], bboxes[..., 4:5]
    w = np.exp(bboxes[..., 2:3])
    length = np.exp(bboxes[..., 3:4])
    h = np.exp(bboxes[..., 5:6])
    if bboxes.shape[-1] > 8:
        vx, vy = bboxes[..., 8:9], bboxes[..., 9:10]
        return np.concatenate([cx, cy, cz, w, length, h, rot, vx, vy], axis=-1)
    return np.concatenate([cx, cy, cz, w, length, h, rot], axis=-1)


def _circle_nms(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    dist_thrs: dict[int, float],
) -> np.ndarray:
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


def post_process(
    all_cls_scores: np.ndarray,
    all_bbox_preds: np.ndarray,
    cfg: InferenceConfig,
    score_thr: float = 0.3,
) -> dict[str, np.ndarray]:
    """Decode, threshold, and NMS raw model outputs.

    Args:
        all_cls_scores: (num_layers, bs, num_query, num_classes) float32
        all_bbox_preds: (num_layers, bs, num_query, 10) float32
        cfg: InferenceConfig with pc_range, post_center_range, max_num, num_classes
        score_thr: score threshold

    Returns:
        dict with 'boxes_3d' (N, 9), 'scores_3d' (N,), 'labels_3d' (N,)
    """
    cls = all_cls_scores[-1][0]  # (num_query, num_classes)
    bpd = all_bbox_preds[-1][0]  # (num_query, 10)

    cls_sig = 1.0 / (1.0 + np.exp(-cls))  # sigmoid
    flat = cls_sig.reshape(-1)
    topk = np.argsort(flat)[::-1][: cfg.max_num]

    scores = flat[topk]
    labels = topk % cfg.num_classes
    boxes = _denormalize_bbox(bpd[topk // cfg.num_classes])

    pcr = np.array(cfg.post_center_range)
    in_range = (boxes[:, :3] >= pcr[:3]).all(1) & (boxes[:, :3] <= pcr[3:]).all(1)
    mask = in_range & (scores > score_thr)
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    if len(scores) == 0:
        return dict(
            boxes_3d=np.zeros((0, 9), dtype=np.float32),
            scores_3d=np.zeros((0,), dtype=np.float32),
            labels_3d=np.zeros((0,), dtype=np.int64),
        )

    keep = _circle_nms(boxes, scores, labels, _NMS_DIST)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Adjust z to bottom-center; shrink dims slightly (reduce over-sized predictions)
    boxes[:, 2] -= boxes[:, 5] * 0.5
    boxes[:, 3:6] *= 0.9

    return dict(
        boxes_3d=boxes.astype(np.float32),
        scores_3d=scores.astype(np.float32),
        labels_3d=labels.astype(np.int64),
    )
