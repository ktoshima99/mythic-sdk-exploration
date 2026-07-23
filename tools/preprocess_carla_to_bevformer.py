#!/usr/bin/env python3
"""Adapt a CARLA-simulated nuScenes-table dataset for the Mythic SDK BEVFormer-tiny pipeline.

Run with the SDK container's venv python (has PIL/cv2/numpy):
    /root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 preprocess_carla_to_bevformer.py \
        --scratch-dir /workspace/carla_scratch --output-root /workspace/carla_output --force

Source data is only read, never mutated. See doc/reverse-engineering plan
(sdk-wild-snowflake.md) for the full rationale.
"""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# CAM_FRONT_NARROW is renamed to CAM_FRONT; CAM_FRONT_WIDE is dropped entirely.
NARROW_CHANNEL = "CAM_FRONT_NARROW"
WIDE_CHANNEL = "CAM_FRONT_WIDE"
FRONT_CHANNEL = "CAM_FRONT"
KEPT_CAMERA_CHANNELS = {
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
}

SCENE_NAME_MAP_DEFAULT = ["scene-9001", "scene-9002", "scene-9003"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def stage_download_check(scratch_dir: Path):
    required = ["can_bus", "maps", "samples", "sweeps", "v1.0-trainval"]
    missing = [d for d in required if not (scratch_dir / d).is_dir()]
    if missing:
        raise SystemExit(f"scratch-dir missing expected subdirs: {missing}")


def stage_load_tables(scratch_dir: Path):
    table_dir = scratch_dir / "v1.0-trainval"
    tables = {}
    for name in [
        "attribute", "calibrated_sensor", "category", "ego_pose", "instance",
        "log", "map", "sample", "sample_annotation", "sample_data", "scene",
        "sensor", "visibility",
    ]:
        tables[name] = load_json(table_dir / f"{name}.json")
    return tables


def stage_rename_scenes(tables, scene_names):
    scenes = tables["scene"]
    if len(scenes) != len(scene_names):
        raise SystemExit(
            f"Expected {len(scene_names)} scenes to rename, found {len(scenes)}"
        )
    old_to_new = {}
    for scene, new_name in zip(scenes, scene_names):
        old_to_new[scene["name"]] = new_name
        scene["name"] = new_name
    return old_to_new


def stage_filter_sensors(tables):
    sensors = tables["sensor"]
    narrow_token = None
    wide_token = None
    kept = []
    for sensor in sensors:
        if sensor["channel"] == NARROW_CHANNEL:
            narrow_token = sensor["token"]
            sensor["channel"] = FRONT_CHANNEL
            kept.append(sensor)
        elif sensor["channel"] == WIDE_CHANNEL:
            wide_token = sensor["token"]
            # dropped
        else:
            kept.append(sensor)
    if narrow_token is None or wide_token is None:
        raise SystemExit("Could not find CAM_FRONT_NARROW/CAM_FRONT_WIDE in sensor.json")
    tables["sensor"] = kept
    return narrow_token, wide_token


def stage_filter_calibrated_sensors(tables, wide_sensor_token):
    dropped_cs_tokens = set()
    kept = []
    for cs in tables["calibrated_sensor"]:
        if cs["sensor_token"] == wide_sensor_token:
            dropped_cs_tokens.add(cs["token"])
        else:
            kept.append(cs)
    tables["calibrated_sensor"] = kept
    return dropped_cs_tokens


def rescale_intrinsic(intrinsic, width_ratio, height_ratio):
    m = np.array(intrinsic, dtype=float)
    m[0, 0] *= width_ratio   # fx
    m[0, 2] *= width_ratio   # cx
    m[1, 1] *= height_ratio  # fy
    m[1, 2] *= height_ratio  # cy
    return m.tolist()


def stage_rescale_intrinsics(tables, sensor_token_to_channel, cs_token_to_sensor_token,
                              target_w, target_h, old_res_by_channel):
    for cs in tables["calibrated_sensor"]:
        sensor_token = cs["sensor_token"]
        channel = sensor_token_to_channel.get(sensor_token)
        if channel not in KEPT_CAMERA_CHANNELS:
            continue
        old_w, old_h = old_res_by_channel[channel]
        width_ratio = target_w / old_w
        height_ratio = target_h / old_h
        cs["camera_intrinsic"] = rescale_intrinsic(cs["camera_intrinsic"], width_ratio, height_ratio)


def stage_filter_and_rewrite_sample_data(tables, dropped_cs_tokens, sensor_token_to_channel,
                                          cs_token_to_sensor_token, target_w, target_h):
    """Filter out dropped-camera sample_data records; rewrite filename/width/height for kept cameras.

    Returns list of (old_relpath, new_relpath) for images that need resizing (cameras only).
    """
    kept_records = []
    resize_jobs = []
    for rec in tables["sample_data"]:
        cs_token = rec["calibrated_sensor_token"]
        if cs_token in dropped_cs_tokens:
            continue  # drop CAM_FRONT_WIDE sample_data entirely
        sensor_token = cs_token_to_sensor_token[cs_token]
        channel = sensor_token_to_channel.get(sensor_token)
        if channel in KEPT_CAMERA_CHANNELS:
            old_relpath = rec["filename"]
            # old_relpath like "samples/CAM_FRONT_NARROW/<hash>.jpg" -> new channel dir
            parts = old_relpath.split("/")
            assert len(parts) == 3, f"unexpected filename shape: {old_relpath}"
            split_dir, _old_channel_dir, fname = parts
            new_relpath = f"{split_dir}/{channel}/{fname}"
            rec["filename"] = new_relpath
            rec["width"] = target_w
            rec["height"] = target_h
            resize_jobs.append((old_relpath, new_relpath, channel))
        # LIDAR_TOP and anything else: leave untouched
        kept_records.append(rec)
    tables["sample_data"] = kept_records
    return resize_jobs


def resize_one(scratch_dir, output_root, old_relpath, new_relpath, target_w, target_h):
    src = scratch_dir / old_relpath
    dst = output_root / new_relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"failed to read image {src}")
    old_h, old_w = img.shape[:2]
    interp = cv2.INTER_AREA if target_w < old_w else cv2.INTER_LINEAR
    resized = cv2.resize(img, (target_w, target_h), interpolation=interp)
    ok = cv2.imwrite(str(dst), resized)
    if not ok:
        raise RuntimeError(f"failed to write image {dst}")


def stage_resize_images(scratch_dir, output_root, resize_jobs, target_w, target_h, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(resize_one, scratch_dir, output_root, old, new, target_w, target_h)
            for old, new, _channel in resize_jobs
        ]
        for f in futures:
            f.result()


def stage_generate_map_mask(tables, scratch_dir, output_root):
    map_json_path = scratch_dir / "maps" / "expansion" / "carla-town10.json"
    map_data = load_json(map_json_path)
    canvas_edge = map_data.get("canvas_edge")
    resolution = 0.1
    if canvas_edge and len(canvas_edge) == 2:
        w_m, h_m = canvas_edge
        w_px = max(1, int(w_m / resolution))
        h_px = max(1, int(h_m / resolution))
    else:
        w_px, h_px = 4000, 4000  # generous fallback
    mask_relpath = "maps/carla-town10-blank-mask.png"
    mask_path = output_root / mask_relpath
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w_px, h_px), color=0).save(mask_path)

    for m in tables["map"]:
        m["filename"] = mask_relpath
    return (w_px, h_px)


def stage_copy_can_bus(scratch_dir, output_root, old_to_new_scene_name):
    out_can_bus = output_root / "can_bus"
    out_can_bus.mkdir(parents=True, exist_ok=True)
    copied = []
    for old_name, new_name in old_to_new_scene_name.items():
        for message in ["pose", "vehicle_monitor"]:
            src = scratch_dir / "can_bus" / f"{old_name}_{message}.json"
            if not src.exists():
                continue
            dst = out_can_bus / f"{new_name}_{message}.json"
            shutil.copyfile(src, dst)
            copied.append(dst.name)
    return copied


UNCHANGED_TABLES = [
    "attribute", "category", "ego_pose", "instance", "log",
    "sample", "sample_annotation", "visibility",
]


def stage_write_tables(tables, output_root):
    table_dir = output_root / "v1.0-trainval"
    for name in UNCHANGED_TABLES + ["calibrated_sensor", "map", "sample_data", "scene", "sensor"]:
        dump_json(tables[name], table_dir / f"{name}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--target-width", type=int, default=1600)
    ap.add_argument("--target-height", type=int, default=900)
    ap.add_argument(
        "--scene-names", type=str, default=",".join(SCENE_NAME_MAP_DEFAULT),
        help="comma-separated scene-NNNN names, assigned in scene.json order",
    )
    ap.add_argument("--force", action="store_true", help="wipe output-root if it exists")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    scratch_dir = args.scratch_dir.resolve()
    output_root = args.output_root.resolve()
    scene_names = args.scene_names.split(",")

    stage_download_check(scratch_dir)

    if output_root.exists():
        if any(output_root.iterdir()):
            if not args.force:
                raise SystemExit(f"{output_root} is non-empty; pass --force to overwrite")
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/9] Loading tables from {scratch_dir}/v1.0-trainval ...")
    tables = stage_load_tables(scratch_dir)

    print(f"[2/9] Renaming {len(tables['scene'])} scenes to {scene_names} ...")
    old_to_new_scene_name = stage_rename_scenes(tables, scene_names)

    print("[3/9] Filtering sensors (CAM_FRONT_NARROW -> CAM_FRONT, dropping CAM_FRONT_WIDE) ...")
    narrow_token, wide_token = stage_filter_sensors(tables)
    sensor_token_to_channel = {s["token"]: s["channel"] for s in tables["sensor"]}

    print("[4/9] Filtering calibrated_sensor records ...")
    dropped_cs_tokens = stage_filter_calibrated_sensors(tables, wide_token)
    cs_token_to_sensor_token = {
        cs["token"]: cs["sensor_token"] for cs in tables["calibrated_sensor"]
    }
    # also need original cs->sensor mapping including dropped ones, for sample_data filtering
    original_cs = load_json(scratch_dir / "v1.0-trainval" / "calibrated_sensor.json")
    full_cs_token_to_sensor_token = {cs["token"]: cs["sensor_token"] for cs in original_cs}

    print("[5/9] Determining raw per-camera resolution from sample_data.json ...")
    raw_sample_data = load_json(scratch_dir / "v1.0-trainval" / "sample_data.json")
    old_res_by_channel = {}
    for rec in raw_sample_data:
        sensor_token = full_cs_token_to_sensor_token.get(rec["calibrated_sensor_token"])
        channel = sensor_token_to_channel.get(sensor_token)
        if channel in KEPT_CAMERA_CHANNELS and channel not in old_res_by_channel:
            old_res_by_channel[channel] = (rec["width"], rec["height"])
    print(f"       raw resolutions: {old_res_by_channel}")

    print("[6/9] Rescaling camera intrinsics ...")
    stage_rescale_intrinsics(
        tables, sensor_token_to_channel, cs_token_to_sensor_token,
        args.target_width, args.target_height, old_res_by_channel,
    )

    print("[7/9] Filtering + rewriting sample_data (dropping CAM_FRONT_WIDE, updating filenames) ...")
    before_count = len(tables["sample_data"])
    resize_jobs = stage_filter_and_rewrite_sample_data(
        tables, dropped_cs_tokens, sensor_token_to_channel,
        full_cs_token_to_sensor_token, args.target_width, args.target_height,
    )
    after_count = len(tables["sample_data"])
    print(f"       sample_data records: {before_count} -> {after_count} "
          f"({before_count - after_count} dropped), {len(resize_jobs)} images to resize")

    print(f"[8/9] Resizing {len(resize_jobs)} images to {args.target_width}x{args.target_height} "
          f"with {args.workers} workers (this may take a while) ...")
    stage_resize_images(scratch_dir, output_root, resize_jobs, args.target_width, args.target_height,
                        workers=args.workers)

    # copy LIDAR files through unchanged (they aren't in resize_jobs)
    print("       Copying LIDAR_TOP files through unchanged ...")
    lidar_count = 0
    for rec in tables["sample_data"]:
        sensor_token = full_cs_token_to_sensor_token.get(rec["calibrated_sensor_token"])
        channel = sensor_token_to_channel.get(sensor_token)
        if channel not in KEPT_CAMERA_CHANNELS:
            src = scratch_dir / rec["filename"]
            dst = output_root / rec["filename"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)
                lidar_count += 1
    print(f"       copied {lidar_count} non-camera files")

    print("[9/9] Generating map mask, copying can_bus, writing tables ...")
    mask_size = stage_generate_map_mask(tables, scratch_dir, output_root)
    can_bus_files = stage_copy_can_bus(scratch_dir, output_root, old_to_new_scene_name)
    stage_write_tables(tables, output_root)

    print()
    print("=== Summary ===")
    print(f"Scenes: {len(tables['scene'])} -> {[s['name'] for s in tables['scene']]}")
    print(f"Samples: {len(tables['sample'])}")
    print(f"Sample_data records: {after_count}")
    print(f"Camera resolutions (old -> new):")
    for ch, (ow, oh) in old_res_by_channel.items():
        print(f"  {ch}: {ow}x{oh} -> {args.target_width}x{args.target_height}")
    print(f"Map mask: {mask_size[0]}x{mask_size[1]}px -> maps/carla-town10-blank-mask.png")
    print(f"CAN bus files copied: {can_bus_files}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
