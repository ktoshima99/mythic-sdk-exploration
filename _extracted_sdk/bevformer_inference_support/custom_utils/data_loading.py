"""BEVFormer data-loading utilities.

Temporal state tracking, batch extraction, mmdet3d dataloader builders,
mmdet3d config loading, dataset helpers, and image-scale extraction.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmcv import Config
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset

# ── Meta unwrapping ──────────────────────────────────────────────────────────


def unwrap_dc(value) -> list:
    """Unwrap a DataContainer or list-of-DataContainer to a plain Python list.

    Test dataloader: field is [DataContainer, ...] (one per GPU).
    Training dataloader: field is DataContainer directly.
    """
    if isinstance(value, list):
        return [item.data[0] if hasattr(item, "data") else item for item in value]
    if hasattr(value, "data"):
        inner = value.data[0]
        if isinstance(inner, list):
            return inner
        return [inner]
    raise TypeError(f"unwrap_dc: unexpected type {type(value)}")


def unwrap_meta(data: dict) -> dict | None:
    """Unwrap a dataloader img_metas batch to a plain dict."""
    raw = data.get("img_metas")
    if raw is None:
        return None
    meta = raw[0] if isinstance(raw, list) else raw
    if hasattr(meta, "data"):
        meta = meta.data[0]
    while isinstance(meta, (list, tuple)):
        meta = meta[0]
    if not isinstance(meta, dict):
        raise ValueError(
            f"unwrap_meta: expected dict after unwrapping, got {type(meta)}"
        )
    return meta


def resolve_lidar_top_sample_data_token(nusc, meta: dict | None) -> str | None:
    """Return ``LIDAR_TOP`` ``sample_data`` token from ``img_metas`` or nuScenes sample lookup.

    Many pkls omit ``lidar_sample_data_token``; ``sample_idx`` is the sample token and can be
    resolved via ``nusc.get('sample', ...)['data']['LIDAR_TOP']`` (same fallback as radar merge).
    """
    if meta is None or nusc is None:
        return None
    t = meta.get("lidar_sample_data_token")
    if t:
        return str(t)
    sid = meta.get("sample_idx")
    if not sid:
        return None
    try:
        sp = nusc.get("sample", str(sid))
        return str(sp["data"]["LIDAR_TOP"])
    except Exception:
        return None


def extract_scene_token(data: dict) -> str:
    meta = unwrap_meta(data)
    if meta is None:
        raise ValueError("extract_scene_token: missing img_metas")
    return str(meta["scene_token"])


def extract_sample_token(data: dict) -> str:
    meta = unwrap_meta(data)
    if "sample_idx" in meta:
        return str(meta["sample_idx"])
    return str(meta["sample_token"])


# ── Temporal state ────────────────────────────────────────────────────────────


@dataclass
class TemporalState:
    """Tracks inter-frame BEV embedding and ego-motion for temporal modelling."""

    prev_bev: np.ndarray | None = None
    prev_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    prev_angle: float = 0.0
    scene_token: str | None = None

    def reset(self) -> None:
        self.prev_bev = None
        self.prev_pos = np.zeros(3, np.float32)
        self.prev_angle = 0.0

    def update_can_bus_delta(self, can_bus_raw: np.ndarray) -> np.ndarray:
        """Compute ego-motion delta and update stored pose. Returns (1, 18) array."""
        cb = can_bus_raw.reshape(-1).copy()
        tmp_pos = cb[:3].copy()
        tmp_ang = float(cb[-1])
        delta = cb.copy()
        if self.prev_bev is not None:
            delta[:3] -= self.prev_pos
            delta[-1] -= self.prev_angle
        else:
            delta[:3] = 0.0
            delta[-1] = 0.0
        self.prev_pos = tmp_pos
        self.prev_angle = tmp_ang
        return delta.reshape(1, -1).astype(np.float32)


# ── Sample array extraction ───────────────────────────────────────────────────


def extract_sample_arrays(
    data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict | None]:
    """Extract ONNX-ready numpy arrays from a dataloader batch (batch_size=1).

    Returns:
        img_np:      (1, N_cams, C, H, W) float32
        lidar2img:   (1, N_cams, 4, 4) float32
        can_bus:     (1, 18) float32
        scene_token: str
        img_norm:    dict or None
    """
    img_raw = data["img"][0] if isinstance(data["img"], list) else data["img"]
    img_t: torch.Tensor = img_raw.data[0].float()
    if img_t.dim() == 4:
        img_t = img_t.unsqueeze(0)
    if img_t.dim() != 5:
        raise ValueError(
            f"extract_sample_arrays: expected img tensor with 5 dims after batch, got {img_t.dim()}"
        )
    img_np = img_t.numpy()

    meta = unwrap_meta(data)
    if meta is None:
        raise ValueError("extract_sample_arrays: could not unwrap img_metas")

    lidar2img = np.array(meta["lidar2img"], dtype=np.float32)
    if lidar2img.ndim == 2:
        lidar2img = lidar2img[np.newaxis, np.newaxis]
    elif lidar2img.ndim == 3:
        lidar2img = lidar2img[np.newaxis]

    can_bus = np.array(meta["can_bus"], dtype=np.float32)
    if can_bus.ndim == 1:
        can_bus = can_bus[np.newaxis]

    return (
        img_np,
        lidar2img,
        can_bus,
        str(meta["scene_token"]),
        meta["img_norm_cfg"],
    )


# ── Config helpers ────────────────────────────────────────────────────────────


def _apply_resolution_scale_override_to_config(cfg, resolution_scale: float) -> None:
    """Override RandomScaleImageMultiViewImage.scales in the test pipeline(s)."""

    def _get_type(obj) -> str | None:
        return obj.get("type") if isinstance(obj, dict) else getattr(obj, "type", None)

    def _get_transforms(obj) -> list:
        v = (
            obj.get("transforms")
            if isinstance(obj, dict)
            else getattr(obj, "transforms", None)
        )
        return v if isinstance(v, list) else []

    test_configs = [cfg.data.test] if isinstance(cfg.data.test, dict) else cfg.data.test
    for test_cfg in test_configs:
        pipeline = (
            test_cfg.get("pipeline")
            if isinstance(test_cfg, dict)
            else getattr(test_cfg, "pipeline", None)
        )
        if not isinstance(pipeline, list):
            continue
        for t in pipeline:
            if _get_type(t) != "MultiScaleFlipAug3D":
                continue
            for inner in _get_transforms(t):
                if _get_type(inner) == "RandomScaleImageMultiViewImage":
                    if isinstance(inner, dict):
                        inner["scales"] = [resolution_scale]
                    else:
                        setattr(inner, "scales", [resolution_scale])
                    break


def load_mmcv_config(config_path: Path) -> Config:
    """Load an mmdet3d .py config and import its plugin if declared.

    Shared preamble for both test and training dataloader builders.
    """
    cfg = Config.fromfile(str(config_path))
    if getattr(cfg, "plugin", False):
        plugin_dir = getattr(cfg, "plugin_dir", None)
        if plugin_dir:
            mod = ".".join(os.path.dirname(plugin_dir).split("/"))
        else:
            mod = ".".join(str(Path(config_path).parent).split("/"))
        importlib.import_module(mod)
    return cfg


def nuscenes_paths_from_test_cfg(cfg: Config) -> tuple[str, str, Path | None]:
    """Pull ``(version, data_root, cache_dir)`` out of ``cfg.data.test``.

    ``cache_dir`` is the parent of ``ann_file`` (where ``create_data.py`` writes the
    devkit cache via ``--out-dir``); ``None`` if no ``ann_file`` is configured.
    """
    t = cfg.data.test
    if isinstance(t, dict):
        version = str(t.get("version", "v1.0-trainval"))
        data_root = str(t["data_root"])
        ann_file = t.get("ann_file")
    else:
        version = str(getattr(t, "version", "v1.0-trainval"))
        data_root = str(t.data_root)
        ann_file = getattr(t, "ann_file", None)
    cache_dir = Path(ann_file).parent if ann_file else None
    return version, data_root, cache_dir


def load_nuscenes_cached(
    version: str,
    dataroot: str | Path,
    *,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
):
    """Return a NuScenes devkit instance — from the side pickle cache if fresh, else freshly built.

    The cache is produced by ``tools/create_data.py`` next to the info pkls (i.e. the
    ``out_path`` arg, which corresponds to ``ann_file``'s parent at inference time);
    pass that directory as ``cache_dir``. If ``cache_dir`` is ``None`` we look in
    ``dataroot`` for backward compatibility. Falls back transparently to
    ``NuScenes(version=..., dataroot=...)`` when the cache is absent or stale.
    """
    from .nuscenes_cache import load_cached

    cached = load_cached(
        version,
        cache_dir if cache_dir is not None else dataroot,
        raw_data_root=dataroot,
    )
    if cached is not None:
        return cached
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=str(dataroot), verbose=verbose)


# ── Dataloader builders ───────────────────────────────────────────────────────


def build_dataloader_from_mmcv_config(
    config: Path | Config,
    *,
    train: bool = False,
    dist: bool = False,
    resolution_scale_override: Optional[float] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[DataLoader, Dataset, dict | None, Config]:
    """Build an MMDet3D dataloader from a .py model config.

    Args:
        config_path: Path to .py config.
        train: If True, build from ``cfg.data.train``; else from ``cfg.data.test``.
        dist: Whether to use distributed loading.
        resolution_scale_override: Test only: override RandomScaleImageMultiViewImage.scales.
        shuffle: Train only: shuffle samples.
        seed: Train only: sampler seed.

    Returns:
        (data_loader, dataset, img_norm_cfg | None, mmcv_config)
        ``img_norm_cfg`` is only set for test mode (from top-level ``img_norm_cfg``).
    """
    if isinstance(config, Config):
        cfg = config
    else:
        cfg = load_mmcv_config(config)

    if train:
        dataset = build_dataset(cfg.data.train)
        loader = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=dist,
            shuffle=shuffle,
            seed=seed,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        return loader, dataset, None, cfg

    if resolution_scale_override is not None:
        _apply_resolution_scale_override_to_config(cfg, resolution_scale_override)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        spp = cfg.data.test.pop("samples_per_gpu", 1)
        if spp > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        for ds in cfg.data.test:
            ds.test_mode = True

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=dist,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    raw_norm = getattr(cfg, "img_norm_cfg", None)
    norm = dict(raw_norm) if raw_norm is not None else None
    return loader, dataset, norm, cfg


# ── Dataloader wrapper ────────────────────────────────────────────────────────


class BEVFormerTorchNetDataLoader:
    """Iterable wrapper around :func:`build_dataloader_from_mmcv_config` use in ``to_training`` conversion.

    Each iteration yields a tuple of tensors matching the ONNX model's input order::

        (img, can_bus, lidar2img, prev_bev)

    where ``img`` is ``(1, N_cams, C, H, W)``, ``can_bus`` is ``(1, 18)``,
    ``lidar2img`` is ``(1, N_cams, 4, 4)``, and ``prev_bev`` is
    ``(bev_h*bev_w, 1, embed_dims)`` initialised to zeros.

    This format is compatible with ``munc._stats_collector.collect_edge_data``
    which unpacks each batch via ``torchnet(*batch)``.
    """

    def __init__(
        self,
        config: "Path | Config",
        *,
        train: bool = False,
        dist: bool = False,
        resolution_scale_override: Optional[float] = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        loader, dataset, norm, cfg = build_dataloader_from_mmcv_config(
            config,
            train=train,
            dist=dist,
            resolution_scale_override=resolution_scale_override,
            shuffle=shuffle,
            seed=seed,
        )
        self._loader = loader

        head = cfg.model.pts_bbox_head
        self._bev_h: int = head.bev_h
        self._bev_w: int = head.bev_w
        self._embed_dims: int = head.transformer.embed_dims

    def __iter__(self):
        for batch in self._loader:
            img_np, lidar2img_np, can_bus_np, _, _ = extract_sample_arrays(batch)
            B = img_np.shape[0]
            prev_bev = torch.zeros(
                (self._bev_h * self._bev_w, B, self._embed_dims), dtype=torch.float32
            )
            yield (
                torch.from_numpy(np.ascontiguousarray(img_np)), 
                torch.from_numpy(np.ascontiguousarray(can_bus_np)),
                torch.from_numpy(np.ascontiguousarray(lidar2img_np)),
                prev_bev.permute(1, 0, 2),
                # torch.tensor( [1.0 if prev_bev is not None else 0.0], dtype=torch.float32 )
                torch.tensor( [1.0], dtype=torch.float32 )
            )

    def __len__(self) -> int:
        return len(self._loader)


# ── Dataset helpers ───────────────────────────────────────────────────────────


def precompute_scene_info(dataset) -> tuple[dict[str, int], int]:
    """Scan dataset.data_infos → (scene_token→frame_count, total_scene_count)."""
    counts: dict[str, int] = {}
    seen: list[str] = []
    for info in dataset.data_infos:
        tok = str(info["scene_token"])
        if tok not in counts:
            seen.append(tok)
        counts[tok] = counts.get(tok, 0) + 1
    return counts, len(seen)


# ── Image scale extraction ────────────────────────────────────────────────────


def extract_img_scale(mmcfg: Any) -> tuple[tuple[int, int], str] | None:
    """Extract the final (w, h) model input resolution from an mmdet config's test pipeline.

    Walks MultiScaleFlipAug3D.img_scale and applies inner transforms that change
    resolution (RandomScaleImageMultiViewImage, PadMultiViewImage).

    Returns:
        ((w, h), pipeline_description) or None if not found.
        pipeline_description is a human-readable string of the resize steps, e.g.
        "1600x900 → x0.5 → 800x450 → pad÷32 → 800x480".
    """
    try:
        for transform in mmcfg.data.test.pipeline:
            if getattr(transform, "type", None) != "MultiScaleFlipAug3D":
                continue
            scale = getattr(transform, "img_scale", None)
            if scale is None:
                continue
            if isinstance(scale, (list, tuple)) and len(scale) > 0:
                scale = scale[0] if isinstance(scale[0], (list, tuple)) else scale
            if not (isinstance(scale, (list, tuple)) and len(scale) == 2):
                continue
            w, h = int(scale[0]), int(scale[1])
            parts = [f"{w}x{h}"]

            for t in getattr(transform, "transforms", None) or []:
                t_type = getattr(t, "type", None)
                if t_type == "RandomScaleImageMultiViewImage":
                    scales = getattr(t, "scales", [1.0])
                    s = scales[0] if scales else 1.0
                    w, h = int(w * s), int(h * s)
                    parts += [f"(scale x{s:g})", f"{w}x{h}"]
                elif t_type == "PadMultiViewImage":
                    div = getattr(t, "size_divisor", 1)
                    w = math.ceil(w / div) * div
                    h = math.ceil(h / div) * div
                    parts += [f"(pad to ÷{div})", f"{w}x{h}"]

            pipeline = " → ".join(parts)
            return (w, h), pipeline
    except (AttributeError, KeyError, TypeError, IndexError):
        return None
    return None


# ── nuScenes sweeps (12Hz) — lazy API walk ───────────────────────────────────


CAMERA_TYPES: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def closest_can_bus_pose(nusc_can_bus, scene_name: str, timestamp: int) -> np.ndarray:
    """Closest CAN bus pose with ``utime <= timestamp`` (same units as nuScenes sample timestamps).

    Mirrors ``_get_can_bus_info`` in ``nuscenes_converter.py`` but parameterized by timestamp
    so sweeps and keyframes share one code path. Server-side scenes with no can_bus JSON
    return a zero vector (same fallback as the converter).
    """
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except Exception:
        return np.zeros(18, dtype=np.float32)

    last_pose = pose_list[0]
    for pose in pose_list:
        if pose["utime"] > timestamp:
            break
        last_pose = pose
    last_pose = dict(last_pose)
    last_pose.pop("utime", None)
    pos = last_pose.pop("pos")
    rotation = last_pose.pop("orientation")
    can_bus: list[float] = []
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in list(last_pose.keys()):
        can_bus.extend(last_pose[key])
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus, dtype=np.float32)


def earliest_sample_data_in_scene(nusc, scene_token: str, channel: str) -> str:
    scene_rec = nusc.get("scene", scene_token)
    sample_tok = scene_rec["first_sample_token"]
    sample = nusc.get("sample", sample_tok)
    sd_tok = sample["data"][channel]
    sd = nusc.get("sample_data", sd_tok)
    while sd["prev"] != "":
        sd = nusc.get("sample_data", sd["prev"])
    return sd["token"]


def _cam_front_token_chain(nusc, scene_token: str) -> list[str]:
    """All CAM_FRONT ``sample_data`` tokens in temporal order (keyframes + sweeps)."""
    t0 = earliest_sample_data_in_scene(nusc, scene_token, "CAM_FRONT")
    chain: list[str] = []
    sd = nusc.get("sample_data", t0)
    while True:
        chain.append(sd["token"])
        if sd["next"] == "":
            break
        sd = nusc.get("sample_data", sd["next"])
    return chain


def advance_sample_data_to_timestamp(nusc, sd_token: str, target_ts: int) -> str:
    """Latest ``sample_data`` on the same sensor chain with ``timestamp <= target_ts``."""
    sd = nusc.get("sample_data", sd_token)
    while sd["next"] != "":
        nxt = nusc.get("sample_data", sd["next"])
        if nxt["timestamp"] <= target_ts:
            sd = nxt
        else:
            break
    return sd["token"]


def build_sweeps_data_infos(nusc, nusc_can_bus, scene_tokens: list[str]) -> list[dict]:
    """Build ``data_infos`` entries matching ``_fill_trainval_infos`` layout at sweep cadence.

    Master clock is ``CAM_FRONT``. Other cameras and ``LIDAR_TOP`` are aligned to the latest
    ``sample_data`` with ``timestamp <=`` the master timestamp (cameras are not perfectly
    synchronous). ``sensor2lidar_*`` for each camera uses ``obtain_sensor2top`` with the
    reference LiDAR frame at that aligned LiDAR timestamp.

    Args:
        nusc: Initialized ``NuScenes``.
        nusc_can_bus: ``NuScenesCanBus`` for the same dataroot.
        scene_tokens: Scene tokens to include, **in order** (slice from the pkl scene order).
    """
    from pyquaternion import Quaternion

    from bevformer_lib.tools.data_converter.nuscenes_converter import obtain_sensor2top

    all_infos: list[dict] = []
    for scene_token in scene_tokens:
        scene_rec = nusc.get("scene", scene_token)
        scene_name = scene_rec["name"]
        master_chain = _cam_front_token_chain(nusc, scene_token)
        ptrs = {c: earliest_sample_data_in_scene(nusc, scene_token, c) for c in CAMERA_TYPES}
        lidar_ptr = earliest_sample_data_in_scene(nusc, scene_token, "LIDAR_TOP")

        for frame_idx, master_tok in enumerate(master_chain):
            master_sd = nusc.get("sample_data", master_tok)
            master_ts = int(master_sd["timestamp"])
            ptrs = {c: advance_sample_data_to_timestamp(nusc, ptrs[c], master_ts) for c in CAMERA_TYPES}
            lidar_ptr = advance_sample_data_to_timestamp(nusc, lidar_ptr, master_ts)

            lidar_sd = nusc.get("sample_data", lidar_ptr)
            cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            l2e_t = cs["translation"]
            l2e_r_mat = Quaternion(cs["rotation"]).rotation_matrix
            e2g_t = np.array(pose["translation"], dtype=np.float64)
            e2g_r_mat = Quaternion(pose["rotation"]).rotation_matrix

            lidar_path = str(nusc.get_sample_data_path(lidar_ptr))
            cams: dict[str, dict] = {}
            for cam in CAMERA_TYPES:
                ct = ptrs[cam]
                _, _, intrinsic = nusc.get_sample_data(ct)
                cam_info = obtain_sensor2top(
                    nusc, ct, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info.update(cam_intrinsic=intrinsic)
                cams[cam] = cam_info

            can_bus = closest_can_bus_pose(nusc_can_bus, scene_name, master_ts)
            info = {
                "lidar_path": lidar_path,
                "lidar_sample_data_token": lidar_ptr,
                "token": master_tok,
                "prev": "",
                "next": "",
                "can_bus": can_bus,
                "frame_idx": frame_idx,
                "sweeps": [],
                "cams": cams,
                "scene_token": scene_token,
                "lidar2ego_translation": cs["translation"],
                "lidar2ego_rotation": cs["rotation"],
                "ego2global_translation": pose["translation"],
                "ego2global_rotation": pose["rotation"],
                "timestamp": master_ts,
            }
            all_infos.append(info)
    return all_infos


def build_sweeps_dataloader(
    config: Path | Config,
    start_scene: int,
    end_scene: int | None = None,
    *,
    dist: bool = False,
    resolution_scale_override: Optional[float] = None,
) -> tuple["DataLoader", "Dataset", dict | None, Config, Any]:
    """Build a test :class:`DataLoader` over nuScenes **sweeps** (~12Hz) instead of keyframe pkl entries.

    Reuses the same ``cfg.data.test`` pipeline as :func:`build_dataloader_from_mmcv_config`.
    Scene indices ``[start_scene, end_scene)`` match the **ordered** scene list from the
    keyframe dataset constructed from the same config (so ``--start-scene`` is comparable
    between ``samples`` and ``sweeps`` modes).

    Returns ``(loader, dataset, img_norm, cfg, nusc)`` — the constructed ``NuScenes`` is
    threaded back so overlay code can reuse it instead of paying ~25 s for a second load.
    """
    _, ref_dataset, norm, cfg = build_dataloader_from_mmcv_config(
        config,
        train=False,
        dist=dist,
        resolution_scale_override=resolution_scale_override,
    )

    scene_order: list[str] = []
    seen: set[str] = set()
    for info in ref_dataset.data_infos:
        st = str(info["scene_token"])
        if st not in seen:
            seen.add(st)
            scene_order.append(st)

    n_scenes = len(scene_order)
    if start_scene < 0 or start_scene >= n_scenes:
        raise ValueError(f"start_scene={start_scene} out of range (n_scenes={n_scenes})")
    end = n_scenes if end_scene is None else min(end_scene, n_scenes)
    selected_scenes = scene_order[start_scene:end]

    version, data_root, cache_dir = nuscenes_paths_from_test_cfg(cfg)

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    nusc = load_nuscenes_cached(version, data_root, cache_dir=cache_dir)
    nusc_can = NuScenesCanBus(dataroot=data_root)
    sweeps_infos = build_sweeps_data_infos(nusc, nusc_can, selected_scenes)
    ref_dataset.data_infos = sweeps_infos

    loader = build_dataloader(
        ref_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=dist,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return loader, ref_dataset, norm, cfg, nusc


# ── Multi-sweep accumulation helpers ──────────────────────────────────────────


def load_nuscenes_lidar_xyz_multisweep(
    nusc,
    lidar_sd_token: str,
    n_sweeps: int = 1,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load N past LiDAR sweeps merged into the master LIDAR_TOP frame.

    Args:
        nusc: Initialized ``NuScenes``.
        lidar_sd_token: ``LIDAR_TOP`` ``sample_data`` token for the current frame.
        n_sweeps: Number of past sweeps to aggregate (1 = single frame).

    Returns:
        ``(xyz, depth)`` arrays of shape ``(N, 3)`` and ``(N,)`` respectively,
        both ``float32``, or ``(None, None)`` on failure.
    """
    try:
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
    except Exception:
        return None, None

    try:
        ref_sd = nusc.get("sample_data", lidar_sd_token)
        ref_cs = nusc.get("calibrated_sensor", ref_sd["calibrated_sensor_token"])
        ref_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])

        ref_sensor2ego = transform_matrix(
            ref_cs["translation"], Quaternion(ref_cs["rotation"]), inverse=False
        )
        ref_ego2global = transform_matrix(
            ref_pose["translation"], Quaternion(ref_pose["rotation"]), inverse=False
        )
        global_from_ref = ref_ego2global @ ref_sensor2ego
        ref_from_global = np.linalg.inv(global_from_ref)

        pts_all: list[np.ndarray] = []
        current_tok = lidar_sd_token
        for _ in range(n_sweeps):
            sd = nusc.get("sample_data", current_tok)
            path = nusc.get_sample_data_path(current_tok)
            pc = LidarPointCloud.from_file(path)
            pts = pc.points[:3, :].T.astype(np.float64)

            if current_tok != lidar_sd_token:
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                sensor2ego = transform_matrix(
                    cs["translation"], Quaternion(cs["rotation"]), inverse=False
                )
                ego2global = transform_matrix(
                    pose["translation"], Quaternion(pose["rotation"]), inverse=False
                )
                T = ref_from_global @ (ego2global @ sensor2ego)
                hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
                pts = (T @ hom.T).T[:, :3]

            pts_all.append(pts.astype(np.float32))
            prev_tok = sd.get("prev", "")
            if not prev_tok:
                break
            current_tok = prev_tok

        if not pts_all:
            return None, None

        xyz = np.concatenate(pts_all, axis=0)
        depth = np.linalg.norm(xyz, axis=1).astype(np.float32)
        return xyz, depth

    except Exception:
        return None, None


def accumulate_radar_points_lidar_frame(
    nusc,
    lidar_sd_token: str,
    radar_sd_by_channel: dict[str, str],
    n_sweeps: int = 5,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Accumulate N past radar sweeps per channel into the master LiDAR frame.

    Args:
        nusc: Initialized ``NuScenes``.
        lidar_sd_token: ``LIDAR_TOP`` ``sample_data`` token (reference frame).
        radar_sd_by_channel: dict mapping RADAR channel name → current ``sample_data`` token.
        n_sweeps: Number of past sweeps to walk per channel.

    Returns:
        ``(xyz, sensor_ids, vel_xy)`` concatenated across all channels, or three ``None``.
    """
    try:
        from nuscenes.utils.data_classes import RadarPointCloud
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
    except Exception:
        return None, None, None

    from .visualization import RADAR_CHANNELS, _sd_to_global_4x4

    try:
        T_lidar_global = np.linalg.inv(_sd_to_global_4x4(nusc, lidar_sd_token))
    except Exception:
        return None, None, None

    xyz_all, sid_all, vel_all = [], [], []

    for sid, ch in enumerate(RADAR_CHANNELS):
        current_tok = radar_sd_by_channel.get(ch)
        if not current_tok:
            continue
        for _ in range(n_sweeps):
            try:
                path = nusc.get_sample_data_path(current_tok)
                pc = RadarPointCloud.from_file(path)
                pts = pc.points
                n = pts.shape[1]
                if n > 0:
                    hom = np.vstack([pts[0:3], np.ones((1, n))])
                    T_glob_radar = _sd_to_global_4x4(nusc, current_tok)
                    T_lr = T_lidar_global @ T_glob_radar
                    li = (T_lr @ hom).T[:, :3].astype(np.float32)
                    xyz_all.append(li)
                    sid_all.append(np.full(n, sid, dtype=np.int32))
                    vxy = np.stack([pts[6], pts[7]], axis=0).astype(np.float64)
                    R = T_lr[:3, :3]
                    v3 = np.vstack([vxy, np.zeros((1, n), dtype=np.float64)])
                    v_l = (R @ v3).T[:, :2].astype(np.float32)
                    vel_all.append(v_l)
            except Exception:
                pass
            sd = nusc.get("sample_data", current_tok)
            prev_tok = sd.get("prev", "")
            if not prev_tok:
                break
            current_tok = prev_tok

    if not xyz_all:
        return None, None, None
    return (
        np.concatenate(xyz_all, axis=0),
        np.concatenate(sid_all, axis=0),
        np.concatenate(vel_all, axis=0),
    )
