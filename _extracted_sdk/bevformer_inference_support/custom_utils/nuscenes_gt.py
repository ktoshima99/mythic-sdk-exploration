"""Ground-truth box helpers for nuScenes (``nusc.get_boxes``)."""

from __future__ import annotations

import numpy as np
from mmdet3d.datasets import NuScenesDataset


def nusc_boxes_to_visualize_result(
    nusc, lidar_sample_data_token: str, class_names: list[str]
) -> dict[str, np.ndarray]:
    """Build a :func:`visualize_frame`-compatible dict using devkit-interpolated boxes.

    Uses ``nusc.get_boxes(lidar_sample_data_token)`` so returned boxes are already in
    **LiDAR** coordinates, matching ``lidar2img`` and the BEV renderer.

    For a sweep ``sample_data`` token between 2Hz keyframes, the devkit linearly interpolates
    box centers / sizes and **slerps** orientations per ``instance_token`` between adjacent
    keyframes. Objects absent from both bracketing keyframes are omitted.

    Args:
        nusc: Initialized ``NuScenes``.
        lidar_sample_data_token: ``LIDAR_TOP`` ``sample_data`` token (sweep or keyframe).
        class_names: Model class order (``cfg.class_names``).
    """
    empty = dict(
        boxes_3d=np.zeros((0, 9), dtype=np.float32),
        scores_3d=np.zeros((0,), dtype=np.float32),
        labels_3d=np.zeros((0,), dtype=np.int64),
    )
    boxes = nusc.get_boxes(lidar_sample_data_token)
    if not boxes:
        return empty

    centers = []
    wlh_list = []
    yaws = []
    labels = []
    for box in boxes:
        name = box.name
        if name in NuScenesDataset.NameMapping:
            name = NuScenesDataset.NameMapping[name]
        if name not in class_names:
            continue
        labels.append(class_names.index(name))
        centers.append(box.center)
        w, l, h = box.wlh
        wlh_list.append([w, l, h])
        yaws.append(box.orientation.yaw_pitch_roll[0])

    if not labels:
        return empty

    locs = np.asarray(centers, dtype=np.float32)
    dims = np.asarray(wlh_list, dtype=np.float32)
    rots = np.asarray(yaws, dtype=np.float32).reshape(-1, 1)
    # Same SECOND-style yaw packing as ``nuscenes_converter._fill_trainval_infos``.
    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1).astype(np.float32)
    pad = 9 - gt_boxes.shape[1]
    if pad > 0:
        gt_boxes = np.concatenate(
            [gt_boxes, np.zeros((len(gt_boxes), pad), dtype=np.float32)], axis=1
        )

    return dict(
        boxes_3d=gt_boxes,
        scores_3d=np.ones(len(labels), dtype=np.float32),
        labels_3d=np.asarray(labels, dtype=np.int64),
    )
