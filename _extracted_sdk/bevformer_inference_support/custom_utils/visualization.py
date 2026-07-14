"""BEVFormer visualization utilities.

3D bounding-box projection onto camera images, top-down BEV map rendering,
and composite frame visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .processing import CLASS_COLORS, denormalize_image

RADAR_CHANNELS: tuple[str, ...] = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)
RADAR_SENSOR_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 0, 255),
    (0, 128, 255),
    (0, 255, 255),
    (255, 0, 128),
    (255, 128, 0),
]

_MAP_LAYERS: list[str] = [
    "drivable_area",
    "road_segment",
    "lane",
    "ped_crossing",
    "walkway",
    "lane_divider",
    "road_divider",
]

# BGR palette per layer (matches devkit approximate colors).
_MAP_LAYER_COLORS_BGR: list[tuple[int, int, int]] = [
    (180, 220, 180),  # drivable_area
    (160, 210, 160),  # road_segment
    (140, 200, 140),  # lane
    (210, 210, 240),  # ped_crossing
    (190, 220, 190),  # walkway
    (160, 130, 210),  # lane_divider
    (130, 100, 190),  # road_divider
]

# Module-level NuScenesMap cache (one instance per map name per process).
_MAP_CACHE: dict[str, object] = {}

# ── 3D box helpers ────────────────────────────────────────────────────────────


def _bbox_corners_3d(bboxes: np.ndarray) -> np.ndarray:
    """Compute (N, 8, 3) corner coordinates from (N, 9+) bbox array."""
    if len(bboxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)

    centers = bboxes[:, :3]
    # column order: [x, y, z, w, l, h, yaw, ...]
    # box dims in local frame: x_size=l, y_size=w, z_size=h
    dims = bboxes[:, [4, 3, 5]].astype(np.float32)  # (l, w, h)
    yaws = bboxes[:, 6] - np.pi / 2.0

    # Unit cube corners. x/y centred at 0, z origin at bottom (matching the
    # bottom-centre convention used for boxes[:, 2] after post_process).
    idx = np.array(np.unravel_index(np.arange(8), [2, 2, 2])).T.astype(np.float32)
    idx = idx[[0, 1, 3, 2, 4, 5, 7, 6]] - np.array([0.5, 0.5, 0.0], dtype=np.float32)
    corners = dims[:, np.newaxis, :] * idx[np.newaxis, :, :]  # (N, 8, 3)

    # Rotate around z-axis
    cos_y = np.cos(yaws)
    sin_y = np.sin(yaws)
    N = len(yaws)
    rot = np.zeros((N, 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos_y
    rot[:, 0, 1] = -sin_y
    rot[:, 1, 0] = sin_y
    rot[:, 1, 1] = cos_y
    rot[:, 2, 2] = 1.0
    corners = np.einsum("aij,ajk->aik", corners, rot)

    return corners + centers[:, np.newaxis, :]


def _draw_boxes_on_image(
    img: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    lidar2img: np.ndarray,
    thickness: int = 2,
) -> np.ndarray:
    """Project 3D boxes onto a camera image."""
    if len(bboxes) == 0:
        return img.copy()

    out = img.copy()
    corners = _bbox_corners_3d(bboxes)  # (N, 8, 3)
    N = corners.shape[0]
    flat = corners.reshape(-1, 3)
    pts4d = np.concatenate([flat, np.ones((len(flat), 1), np.float32)], axis=1)
    proj = pts4d @ np.array(lidar2img, dtype=np.float32).reshape(4, 4).T
    proj[:, 2] = np.clip(proj[:, 2], 1e-5, 1e5)
    proj[:, :2] /= proj[:, 2:3]
    pts2d_raw = proj[:, :2].reshape(N, 8, 2)
    # Clamp to int32-safe range after perspective divide; nan/inf (behind-camera
    # or degenerate geometry) become a large off-screen value skipped by the
    # bounds check below.
    _INT32_SAFE = 1 << 20  # 1M px — far off any real image
    pts2d = np.clip(np.where(np.isfinite(pts2d_raw), pts2d_raw, -_INT32_SAFE), -_INT32_SAFE, _INT32_SAFE)

    lines = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = out.shape[:2]
    for i in range(N):
        color = CLASS_COLORS.get(int(labels[i]), (200, 200, 200))
        pts = pts2d[i].astype(np.int32)
        for s, e in lines:
            p1 = (int(pts[s, 0]), int(pts[s, 1]))
            p2 = (int(pts[e, 0]), int(pts[e, 1]))
            if (0 <= p1[0] < w and 0 <= p1[1] < h) or (0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
    return out


# ── BEV map layer builders ────────────────────────────────────────────────────


def _make_layer(size_px: int) -> np.ndarray:
    """Return a fully-transparent BGRA layer."""
    return np.zeros((size_px, size_px, 4), dtype=np.uint8)


def _alpha_composite_bgr(
    layers: list[tuple[str, np.ndarray, float]],
    size_px: int,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Alpha-composite BGRA layers onto an opaque background; return BGR uint8.

    Each entry is (name, bgra_layer, opacity_scale 0..1).  Layers are applied
    bottom-to-top using straight-alpha "over" blending:
        out.rgb = src.rgb * src.a + dst.rgb * (1 - src.a)
    where src.a is the per-pixel alpha scaled by the per-layer opacity.
    """
    canvas = np.full((size_px, size_px, 3), background, dtype=np.float32)
    for _name, layer, opacity in layers:
        if layer is None:
            continue
        opacity = float(np.clip(opacity, 0.0, 1.0))
        if opacity == 0.0:
            continue
        a = layer[:, :, 3:4].astype(np.float32) / 255.0 * opacity
        src = layer[:, :, :3].astype(np.float32)
        canvas = src * a + canvas * (1.0 - a)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _layer_grid(size_px: int) -> np.ndarray:
    layer = _make_layer(size_px)
    color = (210, 210, 210, 255)
    for i in range(11):
        g = int(i / 10 * size_px)
        cv2.line(layer, (g, 0), (g, size_px), color, 1)
        cv2.line(layer, (0, g), (size_px, g), color, 1)
    return layer


def _layer_ego(size_px: int, pc_range: list[float], size_multiplier: float = 2.0) -> np.ndarray:
    layer = _make_layer(size_px)
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xr, yr = x_max - x_min, y_max - y_min
    ecx = int((0 - y_min) / yr * size_px)
    ecy = int((x_max - 0) / xr * size_px)
    s = size_multiplier
    # Cross-hairs
    cv2.line(layer, (ecx, 0), (ecx, size_px), (160, 160, 160, 200), max(1, round(s)))
    cv2.line(layer, (0, ecy), (size_px, ecy), (160, 160, 160, 200), max(1, round(s)))
    # Ego vehicle box — base size at multiplier=1 is 10×26 px
    ew, el = round(10 * s), round(26 * s)
    ego = np.array(
        [
            [-el // 2, -ew // 2],
            [-el // 2, ew // 2],
            [el // 2, ew // 2],
            [el // 2, -ew // 2],
        ],
        dtype=np.int32,
    ) + np.array([ecx, ecy])
    cv2.fillPoly(layer, [ego], (30, 30, 180, 255))
    # Arrow points canvas-right so after the global 90°CCW+flip it points UP in the final image.
    cv2.arrowedLine(layer, (ecx, ecy), (ecx + el // 2, ecy), (0, 0, 0, 255), max(1, round(s)), tipLength=0.35)
    return layer


def _layer_map_polylines(
    size_px: int,
    pc_range: list[float],
    map_polylines: Sequence[np.ndarray],
) -> np.ndarray:
    """Render HD-map polylines as a BGRA layer."""
    layer = _make_layer(size_px)
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xr, yr = x_max - x_min, y_max - y_min
    color = (40, 75, 140, 220)
    for poly in map_polylines:
        if poly is None or len(poly) < 2:
            continue
        pix = []
        for j in range(len(poly)):
            x, y = float(poly[j, 0]), float(poly[j, 1])
            pix.append((int((y - y_min) / yr * size_px), int((x_max - x) / xr * size_px)))
        if len(pix) > 1:
            arr = np.array(pix, dtype=np.int32)
            closed = bool(
                len(pix) > 2
                and abs(int(pix[0][0]) - int(pix[-1][0])) <= 1
                and abs(int(pix[0][1]) - int(pix[-1][1])) <= 1
            )
            cv2.polylines(layer, [arr], closed, color, 2, cv2.LINE_AA)
    return layer


def _layer_map_raster(size_px: int, map_underlay: dict) -> np.ndarray:
    """Render a devkit raster map underlay as a BGRA layer."""
    layer = _make_layer(size_px)
    map_geom = map_underlay["map_geom"]
    pc_range = map_underlay["pc_range"]

    for layer_idx, (_layer_name, geoms) in enumerate(map_geom):
        b, g, r = _MAP_LAYER_COLORS_BGR[layer_idx % len(_MAP_LAYER_COLORS_BGR)]
        color = (b, g, r, 220)
        for geom in geoms:
            try:
                if geom.is_empty:
                    continue
                for simple in _iter_simple_geoms(geom):
                    if simple.is_empty:
                        continue
                    is_polygon = hasattr(simple, "exterior")
                    pts = _shapely_to_bev_pts(simple, pc_range, size_px)
                    if pts is None or len(pts) < 2:
                        continue
                    if is_polygon:
                        cv2.fillPoly(layer, [pts], color)
                    else:
                        cv2.polylines(layer, [pts], is_polygon, color, 1, cv2.LINE_AA)
            except Exception:
                continue
    return layer


def _layer_lidar(
    size_px: int,
    pc_range: list[float],
    lidar_xy: np.ndarray | None,
    lidar_depth: np.ndarray | None,
    size_multiplier: float = 2.0,
) -> np.ndarray:
    """Render LiDAR scatter as a BGRA layer (depth-colored: blue=near, red=far)."""
    layer = _make_layer(size_px)
    if lidar_xy is None or len(lidar_xy) == 0:
        return layer
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xr, yr = x_max - x_min, y_max - y_min
    if lidar_depth is not None and len(lidar_depth) == len(lidar_xy):
        d = lidar_depth.astype(np.float64)
    else:
        d = np.linalg.norm(lidar_xy, axis=1)
    d0, d1 = float(d.min()), float(d.max())
    for i in range(len(lidar_xy)):
        x, y = float(lidar_xy[i, 0]), float(lidar_xy[i, 1])
        ix = int((y - y_min) / yr * size_px)
        iy = int((x_max - x) / xr * size_px)
        if not (0 <= ix < size_px and 0 <= iy < size_px):
            continue
        t = (d[i] - d0) / (d1 - d0 + 1e-6)
        b = int(255 * (1 - t))
        r = int(255 * t)
        cv2.circle(layer, (ix, iy), max(1, round(0.5 * size_multiplier)), (b, 40, r, 255), -1)
    return layer


def _layer_radar(
    size_px: int,
    pc_range: list[float],
    radar_xy: np.ndarray | None,
    radar_vel: np.ndarray | None,
    radar_sensor_ids: np.ndarray | None,
    radar_draw_velocity: bool = True,
    size_multiplier: float = 2.0,
) -> np.ndarray:
    """Render radar returns (+ optional Doppler arrows) as a BGRA layer."""
    layer = _make_layer(size_px)
    if radar_xy is None or len(radar_xy) == 0:
        return layer
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xr, yr = x_max - x_min, y_max - y_min
    pt_r = max(1, round(0.5 * size_multiplier))
    vel_scale = round(7.5 * size_multiplier)
    for i in range(len(radar_xy)):
        x, y = float(radar_xy[i, 0]), float(radar_xy[i, 1])
        ix = int((y - y_min) / yr * size_px)
        iy = int((x_max - x) / xr * size_px)
        if not (0 <= ix < size_px and 0 <= iy < size_px):
            continue
        sid = int(radar_sensor_ids[i]) if radar_sensor_ids is not None and i < len(radar_sensor_ids) else 0
        b, g, r = RADAR_SENSOR_COLORS_BGR[sid % len(RADAR_SENSOR_COLORS_BGR)]
        cv2.rectangle(layer, (ix - pt_r, iy - pt_r), (ix + pt_r, iy + pt_r), (b, g, r, 255), -1)
        if (
            radar_draw_velocity
            and radar_vel is not None
            and i < len(radar_vel)
            and (abs(radar_vel[i, 0]) + abs(radar_vel[i, 1])) > 1e-3
        ):
            vx, vy = radar_vel[i, 0], radar_vel[i, 1]
            ex = int(ix + vel_scale * np.sign(vx) * min(abs(vx), 5.0))
            ey = int(iy - vel_scale * np.sign(vy) * min(abs(vy), 5.0))
            cv2.arrowedLine(
                layer, (ix, iy), (ex, ey), (60, 60, 60, 200), max(1, round(0.5 * size_multiplier)), tipLength=0.25
            )
    return layer


def _layer_boxes(
    size_px: int,
    pc_range: list[float],
    bboxes: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Render BEV detection boxes as a BGRA layer. Box logic unchanged from original."""
    layer = _make_layer(size_px)
    if len(bboxes) == 0:
        return layer
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xr, yr = x_max - x_min, y_max - y_min
    for i, box in enumerate(bboxes):
        x, y = float(box[0]), float(box[1])
        wb, lb = float(box[3]), float(box[4])
        yaw = float(box[6]) - np.pi / 2.0
        b, g, r = CLASS_COLORS.get(int(labels[i]), (140, 140, 140))

        ix = int((y - y_min) / yr * size_px)
        iy = int((x_max - x) / xr * size_px)
        if not (0 <= ix < size_px and 0 <= iy < size_px):
            continue

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        loc = np.array(
            [
                [lb / 2, wb / 2],
                [lb / 2, -wb / 2],
                [-lb / 2, -wb / 2],
                [-lb / 2, wb / 2],
            ]
        )
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        corners = loc @ rot.T
        pts_img = np.array(
            [
                [
                    int((y + c[1] - y_min) / yr * size_px),
                    int((x_max - (x + c[0])) / xr * size_px),
                ]
                for c in corners
            ],
            dtype=np.int32,
        )

        # Fill: 20% black mixed into the class color so it reads as a tinted shade.
        fb = int(b * 0.8)
        fg = int(g * 0.8)
        fr = int(r * 0.8)
        cv2.fillPoly(layer, [pts_img], (fb, fg, fr, 200))
        # Outline and arrow in the full-saturation class color.
        cv2.polylines(layer, [pts_img], True, (b, g, r, 255), 2)
        al = max(int(lb / xr * size_px // 2), 6)
        cv2.arrowedLine(
            layer,
            (ix, iy),
            (int(ix + al * sin_y), int(iy - al * cos_y)),
            (b, g, r, 255),
            1,
            tipLength=0.4,
        )
    return layer


# ── Map underlay helpers (inlined from devkit_render) ─────────────────────────


def _shapely_to_bev_pts(
    geom,
    pc_range: list[float],
    size_px: int,
) -> np.ndarray | None:
    """Convert a simple Shapely geometry to BEV canvas pixel coordinates (col, row)."""
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xr, yr = x_max - x_min, y_max - y_min
    try:
        if hasattr(geom, "exterior"):
            raw = list(geom.exterior.coords)
        elif hasattr(geom, "coords"):
            raw = list(geom.coords)
        else:
            return None
    except Exception:
        return None
    if len(raw) < 2:
        return None
    return np.array(
        [(int((x_max - c[0]) / xr * size_px), int((c[1] - y_min) / yr * size_px)) for c in raw],
        dtype=np.int32,
    )


def _iter_simple_geoms(geom):
    """Yield leaf Shapely geometries, recursing into Multi* and GeometryCollection."""
    if hasattr(geom, "geoms"):
        for sub in geom.geoms:
            yield from _iter_simple_geoms(sub)
    else:
        yield geom


def _import_nuscenes_map():
    """Import NuScenesMap, patching the matplotlib seaborn style that may not exist."""
    import matplotlib.style as _mpl_style

    def _noop(*_a, **_kw) -> None:
        pass

    _orig = _mpl_style.use
    setattr(_mpl_style, "use", _noop)
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap

        return NuScenesMap
    finally:
        setattr(_mpl_style, "use", _orig)


def build_map_underlay_for_bev(
    nusc,
    scene_token: str,
    lidar_sd_token: str,
    pc_range: list[float],
    *,
    size_px: int = 200,
) -> dict | None:
    """Build a BEV raster map underlay using ``NuScenesMapExplorer.get_map_geom``.

    Returns a dict with keys ``map_geom`` and ``pc_range``, or ``None`` on failure.
    The patch is derived from ego pose + pc_range so the visible region matches the
    BEV inset exactly.
    """
    try:
        NuScenesMap = _import_nuscenes_map()
        from pyquaternion import Quaternion

        scene = nusc.get("scene", scene_token)
        log = nusc.get("log", scene["log_token"])
        map_name = log["location"]
        if map_name not in _MAP_CACHE:
            _MAP_CACHE[map_name] = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)
        nusc_map = _MAP_CACHE[map_name]

        lidar_sd = nusc.get("sample_data", lidar_sd_token)
        pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_q = Quaternion(pose["rotation"])
        ego_x = float(pose["translation"][0])
        ego_y = float(pose["translation"][1])

        x_min, y_min, _, x_max, y_max, _ = pc_range
        patch_box = (ego_x, ego_y, y_max - y_min, x_max - x_min)
        patch_angle = float(ego_q.yaw_pitch_roll[0]) * 180.0 / np.pi

        map_geom = nusc_map.explorer.get_map_geom(patch_box, patch_angle, _MAP_LAYERS)
        n_geoms = sum(len(gs) for _, gs in map_geom)
        if n_geoms == 0:
            print(f"[map] WARNING: get_map_geom returned 0 geometries for {map_name} at ({ego_x:.1f}, {ego_y:.1f})")
        return {"map_geom": map_geom, "pc_range": pc_range, "size_px": size_px}
    except Exception as e:
        print(f"[map] build_map_underlay_for_bev failed: {e!r}")
        return None


# ── BEV map composite ─────────────────────────────────────────────────────────


def _draw_bev_map(
    bboxes: np.ndarray,
    labels: np.ndarray,
    pc_range: list[float],
    size_px: int = 200,
    *,
    size_multiplier: float = 2.0,
    lidar_xy: np.ndarray | None = None,
    lidar_depth: np.ndarray | None = None,
    radar_xy: np.ndarray | None = None,
    radar_vel: np.ndarray | None = None,
    radar_sensor_ids: np.ndarray | None = None,
    radar_draw_velocity: bool = True,
    map_polylines: Sequence[np.ndarray] | None = None,
    map_underlay: dict | None = None,
    layer_opacities: dict[str, float] | None = None,
) -> np.ndarray:
    """Render a top-down BEV map with per-layer BGRA compositing.

    Layer order (bottom to top): grid → map → lidar → radar → boxes → ego.
    Each layer is an independent BGRA buffer; all are composited before the
    final 90° CCW rotate + horizontal flip (forward = up).

    ``lidar_xy`` and ``radar_xy`` layers are each pre-rotated 90° CW before
    compositing to correct their orientation in the pre-flip canvas.
    """
    ops = layer_opacities or {}
    grid_l = _layer_grid(size_px)
    map_l = None
    if map_underlay is not None:
        map_l = cv2.flip(_layer_map_raster(size_px, map_underlay), 1)
    elif map_polylines:
        map_l = _layer_map_polylines(size_px, pc_range, map_polylines)

    lidar_l = _layer_lidar(size_px, pc_range, lidar_xy, lidar_depth, size_multiplier)

    radar_l = _layer_radar(
        size_px, pc_range, radar_xy, radar_vel, radar_sensor_ids, radar_draw_velocity, size_multiplier
    )

    boxes_l = _layer_boxes(size_px, pc_range, bboxes, labels)
    ego_l = _layer_ego(size_px, pc_range, size_multiplier)

    bev = _alpha_composite_bgr(
        layers=[
            ("grid", grid_l, ops.get("grid", 1.0)),
            ("map", map_l, ops.get("map", 0.7)),
            ("lidar", lidar_l, ops.get("lidar", 0.35)),
            ("radar", radar_l, ops.get("radar", 0.5)),
            ("boxes", boxes_l, ops.get("boxes", 0.85)),
            ("ego", ego_l, ops.get("ego", 1.0)),
        ],
        size_px=size_px,
    )

    # Final orientation: rotate 90° CCW + horizontal flip → forward = up.
    M = cv2.getRotationMatrix2D((size_px / 2, size_px / 2), 90, 1.0)
    bev = cv2.warpAffine(bev, M, (size_px, size_px), borderValue=(255, 255, 255))
    bev = cv2.flip(bev, 1)
    return bev


# ── Modality overlays (LiDAR / radar / map) ───────────────────────────────────


def sensor_footprint_hw(
    img_meta: dict | None,
    cam_idx: int,
    canvas_h: int,
    canvas_w: int,
) -> tuple[int, int]:
    """Return ``(H, W)`` of camera pixels before ``PadMultiViewImage`` margin, when available.

    ``img_meta['ori_shape']`` from the mmcv pipeline is the size **after** scaling but **before**
    bottom/right pad to a ``size_divisor``. Clipping draws to this footprint avoids projecting
    LiDAR/radar into the padded grey band (intrinsics still target the real sensor extent).
    """
    if img_meta is None:
        return canvas_h, canvas_w
    ori = img_meta.get("ori_shape")
    if ori is None:
        return canvas_h, canvas_w
    try:
        item = ori[cam_idx]
        oh, ow = int(item[0]), int(item[1])
        if oh <= 0 or ow <= 0:
            return canvas_h, canvas_w
        return min(oh, canvas_h), min(ow, canvas_w)
    except (TypeError, IndexError, ValueError):
        return canvas_h, canvas_w


def draw_lidar_on_image(
    img_bgr: np.ndarray,
    points_lidar: np.ndarray,
    lidar2img: np.ndarray,
    *,
    max_points: int = 10_000,
    valid_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Project LiDAR points onto a camera; color by depth (red=near, blue=far).

    Args:
        valid_hw: If set, ``(H, W)`` clip region (e.g. pre-pad ``ori_shape``); only points
            inside ``[0,W) × [0,H)`` are drawn so pad bands stay empty.
    """
    if points_lidar is None or len(points_lidar) == 0:
        return img_bgr
    pts = np.asarray(points_lidar[:, :3], dtype=np.float64)
    n = len(pts)
    if n > max_points:
        sel = np.random.choice(n, max_points, replace=False)
        pts = pts[sel]
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = hom @ np.asarray(lidar2img, dtype=np.float64).reshape(4, 4).T
    z = np.clip(proj[:, 2], 1e-4, None)
    u_f = np.where(np.isfinite(proj[:, 0]), proj[:, 0] / z, -1e6)
    v_f = np.where(np.isfinite(proj[:, 1]), proj[:, 1] / z, -1e6)
    u = u_f.astype(np.int32)
    v = v_f.astype(np.int32)
    depth = z
    d0, d1 = float(depth.min()), float(depth.max())
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if valid_hw is not None:
        vh = min(int(valid_hw[0]), h)
        vw = min(int(valid_hw[1]), w)
    else:
        vh, vw = h, w
    for i in range(len(u)):
        if not (0 <= u[i] < vw and 0 <= v[i] < vh):
            continue
        t = (depth[i] - d0) / (d1 - d0 + 1e-6)
        b = int(255 * (1 - t))
        r = int(255 * t)
        cv2.circle(out, (int(u[i]), int(v[i])), 1, (b, 40, r), -1, cv2.LINE_AA)
    return out


def draw_radar_on_image(
    img_bgr: np.ndarray,
    radar_xyz: np.ndarray,
    lidar2img: np.ndarray,
    sensor_ids: np.ndarray | None,
    *,
    max_points: int = 2_000,
    valid_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Project radar returns as small filled squares (per-sensor color)."""
    if radar_xyz is None or len(radar_xyz) == 0:
        return img_bgr
    pts = np.asarray(radar_xyz[:, :3], dtype=np.float64)
    n = len(pts)
    if n > max_points:
        sel = np.random.choice(n, max_points, replace=False)
        pts = pts[sel]
        sensor_ids = sensor_ids[sel] if sensor_ids is not None else None
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = hom @ np.asarray(lidar2img, dtype=np.float64).reshape(4, 4).T
    z = np.clip(proj[:, 2], 1e-4, None)
    u_f = np.where(np.isfinite(proj[:, 0]), proj[:, 0] / z, -1e6)
    v_f = np.where(np.isfinite(proj[:, 1]), proj[:, 1] / z, -1e6)
    u = u_f.astype(np.int32)
    v = v_f.astype(np.int32)
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if valid_hw is not None:
        vh = min(int(valid_hw[0]), h)
        vw = min(int(valid_hw[1]), w)
    else:
        vh, vw = h, w
    for i in range(len(u)):
        if not (0 <= u[i] < vw and 0 <= v[i] < vh):
            continue
        sid = int(sensor_ids[i]) if sensor_ids is not None and i < len(sensor_ids) else 0
        color = RADAR_SENSOR_COLORS_BGR[sid % len(RADAR_SENSOR_COLORS_BGR)]
        cv2.rectangle(out, (u[i] - 2, v[i] - 2), (u[i] + 2, v[i] + 2), color, -1, cv2.LINE_AA)
    return out


def load_nuscenes_lidar_xyz(lidar_bin_path: str | Path) -> np.ndarray | None:
    """Return (N, 3) LiDAR points in the LiDAR sensor frame, or ``None`` if load fails."""
    try:
        from nuscenes.utils.data_classes import LidarPointCloud
    except Exception:
        return None
    p = Path(lidar_bin_path)
    if not p.is_file():
        return None
    pc = LidarPointCloud.from_file(str(p))
    pts = pc.points[:3, :].T.astype(np.float32)
    return pts


def _sd_to_global_4x4(nusc, sd_token: str) -> np.ndarray:
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    sensor2ego = transform_matrix(cs["translation"], Quaternion(cs["rotation"]), inverse=False)
    ego2global = transform_matrix(pose["translation"], Quaternion(pose["rotation"]), inverse=False)
    return ego2global @ sensor2ego


def merge_radar_points_lidar_frame(
    nusc,
    lidar_sd_token: str,
    radar_sd_by_channel: dict[str, str],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Stack all RADAR returns into LiDAR coordinates at ``lidar_sd_token`` time.

    Returns:
        ``(xyz, sensor_id_per_point, vel_xy)`` or three ``None`` if radar libs/paths fail.
    """
    try:
        from nuscenes.utils.data_classes import RadarPointCloud
    except Exception:
        return None, None, None
    try:
        T_lidar_global = np.linalg.inv(_sd_to_global_4x4(nusc, lidar_sd_token))
    except Exception:
        return None, None, None
    xyz_all, sid_all, vel_all = [], [], []
    for sid, ch in enumerate(RADAR_CHANNELS):
        tok = radar_sd_by_channel.get(ch)
        if not tok:
            continue
        try:
            path = nusc.get_sample_data_path(tok)
            pc = RadarPointCloud.from_file(path)
        except Exception:
            continue
        pts = pc.points
        n = pts.shape[1]
        if n == 0:
            continue
        hom = np.vstack([pts[0:3], np.ones((1, n))])
        T_glob_radar = _sd_to_global_4x4(nusc, tok)
        T_lr = T_lidar_global @ T_glob_radar
        li = (T_lr @ hom).T[:, :3].astype(np.float32)
        xyz_all.append(li)
        sid_all.append(np.full(n, sid, dtype=np.int32))
        vxy = np.stack([pts[6], pts[7]], axis=0).astype(np.float64)  # (2, n)
        R = T_lr[:3, :3]
        v3 = np.vstack([vxy, np.zeros((1, n), dtype=np.float64)])
        v_l = (R @ v3).T[:, :2].astype(np.float32)
        vel_all.append(v_l)
    if not xyz_all:
        return None, None, None
    return (
        np.concatenate(xyz_all, axis=0),
        np.concatenate(sid_all, axis=0),
        np.concatenate(vel_all, axis=0),
    )


def build_hdmap_polylines_lidar_xy(
    nusc,
    scene_token: str,
    lidar_sd_token: str,
    pc_range: list[float],
    *,
    radius: float = 120.0,
) -> list[np.ndarray]:
    """Return polylines in LiDAR x,y (metres) for BEV drawing. Best-effort (no crash).

    Uses ``NuScenesMap.get_records_in_radius`` near the ego pose of ``lidar_sd_token``.
    Geometry comes from ``extract_polygon`` / lane centerlines / line ``node_tokens`` —
    ``discretize_lanes`` applies only to ``lane`` and ``lane_connector`` records (not
    ``road_segment`` or ``lane_divider``).
    """
    polys: list[np.ndarray] = []
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
    except Exception:
        return polys
    try:
        scene = nusc.get("scene", scene_token)
        log = nusc.get("log", scene["log_token"])
        map_name = log["location"]
        dataroot = Path(nusc.dataroot)
        nusc_map = NuScenesMap(dataroot=str(dataroot), map_name=map_name)
    except Exception:
        return polys
    try:
        lidar_sd = nusc.get("sample_data", lidar_sd_token)
        pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        ego2global = transform_matrix(pose["translation"], Quaternion(pose["rotation"]), inverse=False)
        ex, ey = float(ego2global[0, 3]), float(ego2global[1, 3])
        layers = (
            "lane",
            "lane_connector",
            "lane_divider",
            "road_divider",
            "road_segment",
        )
        rec = nusc_map.get_records_in_radius(ex, ey, radius, layers)
        cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar2ego = transform_matrix(cs["translation"], Quaternion(cs["rotation"]), inverse=False)
        global_from_lidar = ego2global @ lidar2ego
        lidar_from_global = np.linalg.inv(global_from_lidar)
        x0, y0, _, x1, y1, _ = pc_range
        xy_margin = float(radius) + max(10.0, 0.15 * max(x1 - x0, y1 - y0))

        def _to_lidar_xy(gxy: np.ndarray) -> np.ndarray:
            hom = np.concatenate([gxy, np.zeros((len(gxy), 1)), np.ones((len(gxy), 1))], axis=1)
            l = (lidar_from_global @ hom.T).T[:, :2]
            return l.astype(np.float32)

        def _clip_and_append(xy: np.ndarray) -> None:
            if xy.shape[0] < 2:
                return
            m = (
                (xy[:, 0] >= x0 - xy_margin)
                & (xy[:, 0] <= x1 + xy_margin)
                & (xy[:, 1] >= y0 - xy_margin)
                & (xy[:, 1] <= y1 + xy_margin)
            )
            xy_f = xy[m]
            if len(xy_f) > 1:
                polys.append(xy_f.astype(np.float32))

        for layer in layers:
            for tok in rec.get(layer, []):
                try:
                    if layer in ("lane", "lane_connector"):
                        disc = nusc_map.discretize_lanes([tok], resolution_meters=0.5)
                        poses = disc.get(tok)
                        if not poses:
                            continue
                        arr = np.asarray(poses, dtype=np.float64)
                        if arr.ndim != 2 or arr.shape[1] < 2:
                            continue
                        xy = _to_lidar_xy(arr[:, :2])
                        _clip_and_append(xy)
                    elif layer == "road_segment":
                        record = nusc_map.get(layer, tok)
                        p_tok = record.get("polygon_token")
                        if not p_tok:
                            continue
                        shp = nusc_map.extract_polygon(p_tok)
                        if shp.is_empty:
                            continue
                        ext = np.asarray(shp.exterior.coords, dtype=np.float64)
                        if ext.shape[0] < 2:
                            continue
                        xy = _to_lidar_xy(ext[:, :2])
                        _clip_and_append(xy)
                    elif layer in ("lane_divider", "road_divider"):
                        record = nusc_map.get(layer, tok)
                        node_tokens = record.get("node_tokens")
                        if not node_tokens:
                            continue
                        gxy = np.array(
                            [
                                (
                                    float(nusc_map.get("node", nt)["x"]),
                                    float(nusc_map.get("node", nt)["y"]),
                                )
                                for nt in node_tokens
                            ],
                            dtype=np.float64,
                        )
                        if len(gxy) < 2:
                            continue
                        xy = _to_lidar_xy(gxy)
                        _clip_and_append(xy)
                except Exception:
                    continue
    except Exception:
        return polys
    return polys


# ── Composite frame visualization ─────────────────────────────────────────────


def visualize_frame(
    img_np: np.ndarray,
    result: dict[str, np.ndarray],
    lidar2img: np.ndarray,
    img_norm: dict,
    pc_range: list[float],
    *,
    lidar_xyz: np.ndarray | None = None,
    overlay_lidar_cam: bool = False,
    overlay_lidar_bev: bool = False,
    radar_xyz: np.ndarray | None = None,
    radar_vel_xy: np.ndarray | None = None,
    radar_sensor_ids: np.ndarray | None = None,
    overlay_radar_cam: bool = False,
    overlay_radar_bev: bool = False,
    radar_draw_velocity: bool = False,
    map_polylines_xy: Sequence[np.ndarray] | None = None,
    map_underlay: dict | None = None,
    overlay_map_bev: bool = False,
    img_metas: dict | None = None,
) -> np.ndarray:
    """Render 6 camera views with projected 3D boxes + BEV map.

    Optional modality overlays (LiDAR / radar / HD map) are applied **after** denormalization
    and respect the same ``lidar2img`` / ``pc_range`` conventions as detections.

    Args:
        img_np: (1, N_cams, C, H, W) float32 normalized images.
        result: dict with 'boxes_3d', 'scores_3d', 'labels_3d'.
        lidar2img: (1, N_cams, 4, 4) projection matrices.
        img_norm: normalization config used for img_np.
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        lidar_xyz: (N, 3+) points in LiDAR frame (typically from ``LidarPointCloud``).
        overlay_lidar_cam / overlay_lidar_bev: draw depth-colored LiDAR projections.
        radar_*: merged radar returns in LiDAR frame (see :func:`merge_radar_points_lidar_frame`).
        map_polylines_xy: list of (M,2) polylines in LiDAR x,y metres (polylines style).
        map_underlay: raster map underlay dict from :func:`build_map_underlay_for_bev`.
        overlay_map_bev: draw map on BEV inset.
        img_metas: optional dataloader ``img_metas[0]`` dict; ``ori_shape`` clips LiDAR and
            radar camera overlays to the real sensor footprint (excludes ÷32 pad).

    Returns:
        (H, W, 3) uint8 BGR composited visualization.
    """
    n_cams = img_np.shape[1]
    raw_imgs = [denormalize_image(img_np[0, c], img_norm) for c in range(n_cams)]

    boxes = result.get("boxes_3d", np.zeros((0, 9), dtype=np.float32))
    labels = result.get("labels_3d", np.zeros((0,), dtype=np.int64))

    cam_imgs = []
    for cam_idx, raw in enumerate(raw_imgs):
        foot_h, foot_w = sensor_footprint_hw(img_metas, cam_idx, raw.shape[0], raw.shape[1])
        if overlay_lidar_cam and lidar_xyz is not None and len(lidar_xyz) > 0:
            raw = draw_lidar_on_image(
                raw,
                lidar_xyz,
                lidar2img[0, cam_idx],
                valid_hw=(foot_h, foot_w),
            )
        if overlay_radar_cam and radar_xyz is not None and len(radar_xyz) > 0:
            raw = draw_radar_on_image(
                raw,
                radar_xyz,
                lidar2img[0, cam_idx],
                radar_sensor_ids,
                valid_hw=(foot_h, foot_w),
            )
        if len(boxes) > 0 and lidar2img.shape[1] > cam_idx:
            raw = _draw_boxes_on_image(raw, boxes, labels, lidar2img[0, cam_idx])
        cam_imgs.append(raw)

    bev_sz = 500
    lidar_xy = lidar_depth = None
    if lidar_xyz is not None and len(lidar_xyz) > 0:
        lidar_xy = np.asarray(lidar_xyz[:, :2], dtype=np.float32)
        lidar_depth = np.asarray(lidar_xyz[:, 2], dtype=np.float32)

    radar_xy_bev = radar_vel_b = r_sensors = None
    if radar_xyz is not None and len(radar_xyz) > 0:
        radar_xy_bev = np.asarray(radar_xyz[:, :3], dtype=np.float32)
        radar_vel_b = radar_vel_xy
        r_sensors = radar_sensor_ids

    effective_map_underlay = map_underlay if (overlay_map_bev and map_underlay is not None) else None
    effective_polys = map_polylines_xy if (overlay_map_bev and map_underlay is None and map_polylines_xy) else None

    size_multiplier = 4
    bev = _draw_bev_map(
        boxes,
        labels,
        pc_range,
        size_px=bev_sz * size_multiplier,
        size_multiplier=size_multiplier,
        lidar_xy=lidar_xy if overlay_lidar_bev else None,
        lidar_depth=lidar_depth if overlay_lidar_bev else None,
        radar_xy=radar_xy_bev if overlay_radar_bev else None,
        radar_vel=radar_vel_b if overlay_radar_bev else None,
        radar_sensor_ids=r_sensors if overlay_radar_bev else None,
        radar_draw_velocity=radar_draw_velocity,
        map_polylines=effective_polys,
        map_underlay=effective_map_underlay,
    )

    # Layout: 6 cameras in 2×3 grid, BEV on the right.
    # Display: top    row L→R: FRONT_LEFT | FRONT | FRONT_RIGHT
    #          bottom row L→R: BACK_LEFT(flip) | BACK(flip) | BACK_RIGHT(flip)
    ordered = [
        cam_imgs[2],
        cam_imgs[0],
        cam_imgs[1],
        cv2.flip(cam_imgs[4], 1),
        cv2.flip(cam_imgs[3], 1),
        cv2.flip(cam_imgs[5], 1),
    ]
    cam_grid = np.vstack([np.hstack(ordered[:3]), np.hstack(ordered[3:])])

    bev_r = cv2.resize(
        bev,
        (int(bev.shape[1] * cam_grid.shape[0] / bev.shape[0]), cam_grid.shape[0]),
    )
    return np.hstack([cam_grid, bev_r])
