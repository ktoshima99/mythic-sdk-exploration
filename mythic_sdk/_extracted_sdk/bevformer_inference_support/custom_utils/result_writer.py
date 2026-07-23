"""BEVFormer result-writing utilities.

Per-scene output management: frame images, MP4 video compilation, and
nuScenes-format JSON export.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, track

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_bgr_uint8(img: np.ndarray) -> np.ndarray | None:
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    return img if img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0 else None


def _make_video(
    images_dir: Path,
    video_path: Path,
    fps: int,
    progress: Progress | None = None,
) -> None:
    imgs = sorted(
        p for p in images_dir.iterdir() if p.suffix in (".jpg", ".jpeg", ".png")
    )
    if not imgs:
        return
    first = cv2.imread(str(imgs[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    max_w, max_h = 3840, 2160
    if w > max_w or h > max_h:
        s = min(max_w / w, max_h / h)
        w, h = int(w * s), int(h * s)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    def _write_frame(p: Path) -> None:
        frame = cv2.imread(str(p))
        if frame is not None:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            vid.write(frame)

    if progress is not None:
        task = progress.add_task(
            f"[green]  └ Video → {video_path.name}", total=len(imgs)
        )
        for p in imgs:
            _write_frame(p)
            progress.advance(task)
        progress.update(task, visible=False)
    else:
        for p in track(
            imgs, description=f"  └ Video → {video_path.name}", console=console
        ):
            _write_frame(p)

    vid.release()
    console.print(f"  video → {video_path}")


def _to_nuscenes_fmt(
    result: dict[str, np.ndarray],
    sample_token: str,
    class_names: list[str],
) -> list[dict]:
    try:
        from pyquaternion import Quaternion as Quat
    except ImportError:
        Quat = None  # noqa: N816

    boxes = result.get("boxes_3d", np.zeros((0, 9)))
    scores = result.get("scores_3d", np.zeros((0,)))
    labels = result.get("labels_3d", np.zeros((0,), dtype=np.int64))
    out = []
    for i in range(len(boxes)):
        box = boxes[i]
        yaw = float(box[6])
        if Quat is not None:
            q = Quat(axis=[0, 0, 1], angle=yaw)
            rot = [float(q.w), float(q.x), float(q.y), float(q.z)]
        else:
            rot = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
        li = int(labels[i])
        if li < 0 or li >= len(class_names):
            raise IndexError(
                f"label index {li} out of range for class_names (len={len(class_names)})"
            )
        name = class_names[li]
        out.append(
            {
                "sample_token": sample_token,
                "translation": [float(box[0]), float(box[1]), float(box[2])],
                "size": [float(box[3]), float(box[4]), float(box[5])],
                "rotation": rot,
                "velocity": [
                    float(box[7]) if len(box) > 7 else 0.0,
                    float(box[8]) if len(box) > 8 else 0.0,
                ],
                "detection_name": name,
                "detection_score": float(scores[i]),
                "attribute_name": "",
            }
        )
    return out


# ── Result writer ─────────────────────────────────────────────────────────────


class ResultWriter:
    """Manages per-scene output: frame images, optional video, optional JSON."""

    def __init__(
        self,
        output_dir: Path,
        save_vis: bool = True,
        save_json: bool = False,
        fps: int = 3,
        output_resolution_scale: float = 0.5,
    ) -> None:
        self.output_dir = output_dir
        self.save_vis = save_vis
        self.save_json = save_json
        self.fps = fps
        self.output_resolution_scale = float(output_resolution_scale)
        self._json_acc: list[dict] = []
        self._scene_dir: Path | None = None
        self._img_dir: Path | None = None
        self._frame_idx: int = 0

    def begin_scene(self, scene_token: str, scene_idx: int) -> None:
        self._frame_idx = 0
        scene_dir = self.output_dir / f"{scene_idx:03d}-{scene_token}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        self._scene_dir = scene_dir
        if self.save_vis:
            self._img_dir = scene_dir / "images"
            self._img_dir.mkdir(exist_ok=True)

    def write_frame(
        self,
        vis_img: np.ndarray | None = None,
        result: dict[str, np.ndarray] | None = None,
        sample_token: str | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        if self.save_vis and vis_img is not None and self._img_dir is not None:
            ready = _to_bgr_uint8(vis_img)
            if ready is not None:
                if self.output_resolution_scale != 1.0:
                    h, w = ready.shape[:2]
                    ready = cv2.resize(
                        ready,
                        (max(1, int(w * self.output_resolution_scale)),
                         max(1, int(h * self.output_resolution_scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imwrite(
                    str(self._img_dir / f"frame_{self._frame_idx:06d}.jpg"), ready,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )

        if self.save_json and result is not None and sample_token is not None:
            if not class_names:
                raise ValueError(
                    "class_names is required when save_json is True (pass cfg.class_names from the model config)"
                )
            entries = _to_nuscenes_fmt(result, sample_token, class_names)
            self._json_acc.extend(entries)

        self._frame_idx += 1

    def end_scene(self, progress: Progress | None = None) -> None:
        if not self.save_vis or self._img_dir is None or self._scene_dir is None:
            return
        _make_video(
            self._img_dir, self._scene_dir / "scene.mp4", self.fps, progress=progress
        )

    def finalize(self) -> None:
        if self.save_json and self._json_acc:
            out: dict = {"meta": {}, "results": {}}
            for item in self._json_acc:
                out["results"].setdefault(item["sample_token"], []).append(item)
            dest = self.output_dir / "results.json"
            with open(dest, "w") as f:
                json.dump(out, f, indent=2)
            console.print(f"Saved results.json → {dest}")
