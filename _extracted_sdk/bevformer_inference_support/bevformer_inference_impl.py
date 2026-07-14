"""Heavy backend for ``bevformer_inference`` — imported when a subcommand runs, not for top-level ``--help``.

Loads PyTorch, mmcv, ONNX Runtime, nuScenes, and (for TorchNet) munc. The public CLI lives in
``bevformer_inference.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import typer
from mmcv import Config
from mythic.model_zoo.bevformer.bevformer_lib.custom_utils import (
    ResultWriter,
    SceneFilter,
    TemporalState,
    apply_crop_resize_to_batch,
    build_dataloader_from_mmcv_config,
    build_pth_model,
    build_sweeps_dataloader,
    build_torchnet_from_onnx,
    default_out_dir,
    extract_img_scale,
    extract_sample_arrays,
    extract_sample_token,
    get_prev_bev,
    load_onnx_session_or_suggest_torchnet,
    onnx_run_frame,
    post_process,
    precompute_scene_info,
    print_modality_overlays_table,
    print_run_summary,
    pth_run_frame,
    run_inference_loop,
    torchnet_run_frame,
    unwrap_meta,
    visualize_frame,
)
from mythic.model_zoo.bevformer.bevformer_lib.custom_utils.data_loading import (
    accumulate_radar_points_lidar_frame,
    advance_sample_data_to_timestamp,
    earliest_sample_data_in_scene,
    load_nuscenes_cached,
    load_nuscenes_lidar_xyz_multisweep,
    nuscenes_paths_from_test_cfg,
    resolve_lidar_top_sample_data_token,
)
from mythic.model_zoo.bevformer.bevformer_lib.custom_utils.nuscenes_gt import (
    nusc_boxes_to_visualize_result as nusc_interp_boxes,
)
from mythic.model_zoo.bevformer.bevformer_lib.custom_utils.processing import (
    InferenceConfig,
    adjust_lidar2img_for_crop_resize,
    parse_config_py,
    parse_crop,
    parse_resize,
)
from mythic.model_zoo.bevformer.bevformer_lib.custom_utils.visualization import (
    RADAR_CHANNELS,
    build_hdmap_polylines_lidar_xy,
    build_map_underlay_for_bev,
    load_nuscenes_lidar_xyz,
    merge_radar_points_lidar_frame,
)
from rich.console import Console

warnings.filterwarnings(
    "ignore",
    message=".*pretrained is a deprecated key.*",
    category=UserWarning,
)


DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_RESULTS_DIR = "bevformer-inference-results"
DEFAULT_GROUND_TRUTH_CONFIG = (
    Path(__file__).resolve().parent / "bevformer_lib/projects/configs/bevformer/bevformer_tiny.py"
)

console = Console()


class InferenceDataType(str, Enum):
    """Dataloader mode for NuScenes test sequences."""

    samples = "samples"
    sweeps = "sweeps"


def coerce_inference_data_type(value: InferenceDataType | str) -> InferenceDataType:
    """Normalize Typer/Click ``ctx.params`` values (enum or string) to :class:`InferenceDataType`."""
    if isinstance(value, InferenceDataType):
        return value
    return InferenceDataType(str(value).lower().strip())


def coerce_path(value: Path | str) -> Path:
    """Normalize Typer/Click ``ctx.params`` path arguments (often serialized as strings)."""
    return value if isinstance(value, Path) else Path(value)


def coerce_optional_path(value: Path | str | None) -> Path | None:
    """Normalize an optional Typer/Click path argument to ``Path`` or ``None``."""
    if value is None:
        return None
    return coerce_path(value)


@dataclass(frozen=True)
class CommonRunOpts:
    """Resolved CLI options (``resolve_common_opts`` is the only mutator)."""

    config_path: Path
    inference_config: InferenceConfig
    mmcfg: Config
    output_dir: Path
    data_type: InferenceDataType
    start_scene: int
    end_scene: Optional[int]
    score_thr: Optional[float]
    device: str
    crop: Optional[tuple[int, int, int, int]]
    resize: Optional[tuple[int, int]]
    resolution_scale: Optional[float]
    vis: bool
    save_json: bool
    fps: int
    lidar_cameras: bool
    lidar_bev: bool
    radar_cameras: bool
    radar_bev: bool
    map_bev: bool
    map_style_polylines: bool
    lidar_nsweeps: int
    radar_nsweeps: int
    output_resolution_scale: float
    # Whether user explicitly set each modality flag
    lidar_cameras_explicit: bool
    lidar_bev_explicit: bool
    radar_cameras_explicit: bool
    radar_bev_explicit: bool
    map_bev_explicit: bool


def resolve_common_opts(
    ctx: typer.Context,
    *,
    subcommand: str,
    results_dir_stem: str,
) -> CommonRunOpts:
    """Parse crop/resize, resolve FPS + modality defaults from ``cfg.input_modality``, build mmcfg.

    Reads shared CLI options from ``ctx.params``. ``results_dir_stem`` names the auto output folder
    after ``<subcommand>-`` (checkpoint stem, ONNX stem, etc.).
    """
    params = ctx.params
    data_type = coerce_inference_data_type(params["data_type"])

    config_path = coerce_path(params["config"])
    inference_config, mmcfg = parse_config_py(config_path)
    crop = parse_crop(params["crop"])
    resize = parse_resize(params["resize"])

    def _resolve_modality_overlay(cli_flag: Optional[bool], modality_key: str) -> tuple[bool, bool]:
        explicit = cli_flag is not None
        if cli_flag is None:
            return inference_config.modality_use(modality_key), explicit
        return cli_flag, explicit

    lc, lcx = _resolve_modality_overlay(params["lidar_cameras"], "use_lidar")
    lb, lbx = _resolve_modality_overlay(params["lidar_bev"], "use_lidar")
    rc, rcx = _resolve_modality_overlay(params["radar_cameras"], "use_radar")
    rb, rbx = _resolve_modality_overlay(params["radar_bev"], "use_radar")
    mb, mbx = _resolve_modality_overlay(params["map_bev"], "use_map")

    fps = params["fps"]
    resolved_fps = fps if fps is not None else (2 if data_type == InferenceDataType.samples else 12)

    _pipeline = extract_img_scale(mmcfg)
    model_img_size = _pipeline[0] if _pipeline else None

    effective_res = resize if resize else (crop[2], crop[3]) if crop else model_img_size or (0, 0)
    name_stem = results_dir_stem if results_dir_stem else config_path.stem
    out = default_out_dir(
        name_stem or "",
        coerce_optional_path(params["output_dir"]),
        effective_res,
        base_dir=BASE_RESULTS_DIR,
        subcommand_prefix=subcommand,
        data_variant=data_type.value,
        lidar_cameras=lc,
        lidar_bev=lb,
        radar_cameras=rc,
        radar_bev=rb,
        map_bev=mb,
    )

    return CommonRunOpts(
        config_path=config_path,
        inference_config=inference_config,
        mmcfg=mmcfg,
        output_dir=out,
        data_type=data_type,
        start_scene=params["start_scene"],
        end_scene=params["end_scene"],
        score_thr=params.get("score_thr"),
        device=params["device"] or DEFAULT_DEVICE,
        crop=crop,
        resize=resize,
        resolution_scale=params["resolution_scale"],
        vis=params["vis"],
        save_json=params.get("save_json", False),
        fps=resolved_fps,
        lidar_cameras=lc,
        lidar_bev=lb,
        radar_cameras=rc,
        radar_bev=rb,
        map_bev=mb,
        map_style_polylines=params["map_style_polylines"],
        lidar_nsweeps=params["lidar_nsweeps"],
        radar_nsweeps=params["radar_nsweeps"],
        output_resolution_scale=params["output_resolution_scale"],
        lidar_cameras_explicit=lcx,
        lidar_bev_explicit=lbx,
        radar_cameras_explicit=rcx,
        radar_bev_explicit=rbx,
        map_bev_explicit=mbx,
    )


def _modality_cell(enabled: bool, explicit: bool, flag_on: str, flag_off: str) -> str:
    if enabled:
        return f"enabled ({flag_on})" if explicit else "enabled (default from config)"
    return f"disabled ({flag_off})" if explicit else "disabled (default)"


def build_modality_overlay_rows(opts: CommonRunOpts) -> list[tuple[str, str, str, str]]:
    """4-column rows for :func:`print_modality_overlays_table`."""
    cam_use = opts.inference_config.modality_use("use_camera")
    lid_use = opts.inference_config.modality_use("use_lidar")
    rad_use = opts.inference_config.modality_use("use_radar")
    map_use = opts.inference_config.modality_use("use_map")
    ext_use = opts.inference_config.modality_use("use_external")

    rows: list[tuple[str, str, str, str]] = [
        (
            "camera",
            "enabled" if cam_use else "disabled",
            "always (model input)",
            "n/a (BEV is synthetic)",
        ),
        (
            "lidar",
            "enabled" if lid_use else "disabled",
            _modality_cell(
                opts.lidar_cameras,
                opts.lidar_cameras_explicit,
                "--lidar-cameras",
                "--no-lidar-cameras",
            ),
            _modality_cell(
                opts.lidar_bev,
                opts.lidar_bev_explicit,
                "--lidar-bev",
                "--no-lidar-bev",
            ),
        ),
        (
            "radar",
            "enabled" if rad_use else "disabled",
            _modality_cell(
                opts.radar_cameras,
                opts.radar_cameras_explicit,
                "--radar-cameras",
                "--no-radar-cameras",
            ),
            _modality_cell(
                opts.radar_bev,
                opts.radar_bev_explicit,
                "--radar-bev",
                "--no-radar-bev",
            ),
        ),
        (
            "map",
            "enabled" if map_use else "disabled",
            "n/a",
            _modality_cell(opts.map_bev, opts.map_bev_explicit, "--map-bev", "--no-map-bev"),
        ),
        (
            "external / can_bus",
            "enabled" if ext_use else "disabled",
            "n/a (model input only)",
            "n/a (model input only)",
        ),
    ]
    return rows


def _data_root(cfg: Config) -> str:
    t = cfg.data.test
    if isinstance(t, dict):
        return str(t["data_root"])
    return str(t.data_root)


def run_video_pipeline(
    *,
    backend: str,
    subcommand: str,
    opts: CommonRunOpts,
    loader,
    dataset,
    summary_model_path: Path,
    frame_runner: Optional[Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    torchnet=None,
    ground_truth_mode: bool = False,
    interpolate_sweep_gt: bool = False,
    pad_with_shape: bool = True,
    torchnet_checkpoint: Optional[Path] = None,
    nusc=None,
) -> None:
    """Build writer + temporal state, run :func:`run_inference_loop`, finalize.

    ``nusc`` may be passed in by the sweeps dataloader (which already constructs one);
    otherwise we construct it lazily after the summary tables print, and only when
    an enabled feature actually needs it (single-sweep lidar overlays do not).
    """
    extract_gt_result_fn = None
    if ground_truth_mode:
        from mythic.model_zoo.bevformer.bevformer_lib.custom_utils.ground_truth import extract_gt_result

        extract_gt_result_fn = extract_gt_result

    cfg = opts.inference_config
    scene_counts, n_dataset_scenes = precompute_scene_info(dataset)
    _pipeline = extract_img_scale(opts.mmcfg)
    model_img_size = _pipeline[0] if _pipeline else None
    model_img_pipeline = _pipeline[1] if _pipeline else None

    needs_nusc = (
        opts.radar_cameras
        or opts.radar_bev
        or opts.map_bev
        or (opts.lidar_nsweeps > 1 and (opts.lidar_cameras or opts.lidar_bev))
        or (ground_truth_mode and interpolate_sweep_gt)
    )
    data_root = _data_root(opts.mmcfg)

    modality_rows = build_modality_overlay_rows(opts)

    extra_note_sweeps = {}
    if opts.data_type == InferenceDataType.sweeps:
        extra_note_sweeps = {
            "Temporal caveat": ("Model trained on 2Hz keyframes; prev_bev at ~12Hz may differ slightly."),
            "Camera time alignment": ("Master=CAM_FRONT chain; other cams use latest sample_data with ts ≤ master"),
            "Lidar reference": "Closest LIDAR_TOP sample_data with ts ≤ master CAM_FRONT ts.",
            "can_bus": ("closest CAN pose with utime ≤ sample_data ts"),
        }

    score_block = {}
    if opts.score_thr is not None:
        score_block["Score threshold"] = str(opts.score_thr)

    print_run_summary(
        backend=backend,
        device=opts.device,
        model_path=summary_model_path,
        model_name=opts.config_path.stem,
        output_dir=opts.output_dir,
        img_norm=cfg.img_norm,
        pc_range=cfg.pc_range,
        score_thr=opts.score_thr or 0.0,
        crop=opts.crop,
        resize=opts.resize,
        start_scene=opts.start_scene,
        end_scene=opts.end_scene,
        model_img_size=model_img_size,
        model_img_pipeline=model_img_pipeline,
        extra={
            "Backend": f"{subcommand}",
            "Data type": ("samples (2Hz)" if opts.data_type == InferenceDataType.samples else "sweeps (12Hz)"),
            "FPS": (
                f"{opts.fps}  (default for {opts.data_type.value})"
                if opts.fps == (2 if opts.data_type == InferenceDataType.samples else 12)
                else f"{opts.fps}  (user override)"
            ),
            "Resolution scale": (
                f"{opts.resolution_scale}  (overrides RandomScaleImageMultiViewImage)"
                if opts.resolution_scale is not None
                else "none (config pipeline)"
            ),
            "Per-scene reset": "prev_bev=zeros, use_prev_bev=0.0 on first frame (TSA no-history)",
            "Map style": "polylines (--map-style-polylines)" if opts.map_style_polylines else "raster",
            "LiDAR sweeps": str(opts.lidar_nsweeps),
            "Radar sweeps": str(opts.radar_nsweeps),
            **score_block,
            **extra_note_sweeps,
            **(
                {
                    "GT interpolation": (
                        (
                            "enabled — nusc.get_boxes(LIDAR_TOP sd); devkit lerp/slerp between "
                            "2Hz keyframes; transient objects may be dropped"
                        )
                        if interpolate_sweep_gt
                        else ("disabled — sweeps show boxes only with --interpolate-sweep-annotations")
                    )
                }
                if (ground_truth_mode and opts.data_type == InferenceDataType.sweeps)
                else {}
            ),
            **(
                {
                    "Checkpoint overlay": (
                        f"{torchnet_checkpoint}  (strict=False; missing/unexpected keys logged above if any)"
                    )
                }
                if torchnet_checkpoint is not None
                else {}
            ),
        },
    )
    print_modality_overlays_table(modality_rows)

    if nusc is None and needs_nusc:
        version, root, cache_dir = nuscenes_paths_from_test_cfg(opts.mmcfg)
        with console.status("[cyan]Loading NuScenes devkit...[/cyan]"):
            nusc = load_nuscenes_cached(version, root, cache_dir=cache_dir)

    writer = ResultWriter(
        opts.output_dir,
        save_vis=opts.vis,
        save_json=opts.save_json and not ground_truth_mode,
        fps=opts.fps,
        output_resolution_scale=opts.output_resolution_scale,
    )
    if opts.data_type == InferenceDataType.sweeps:
        # build_sweeps_dataloader already pre-slices data_infos to [start_scene, end_scene),
        # so the scene indices yielded by the loader start at 0. Use a pass-through filter
        # and offset the per-scene output dir names by opts.start_scene so they reflect the
        # absolute scene index.
        sf = SceneFilter(0, n_dataset_scenes)
        n_active = n_dataset_scenes
        loop_start_scene = 0
        scene_index_offset = opts.start_scene
    else:
        sf = SceneFilter(opts.start_scene, opts.end_scene)
        n_active = (
            min(opts.end_scene, n_dataset_scenes) if opts.end_scene is not None else n_dataset_scenes
        ) - opts.start_scene
        loop_start_scene = opts.start_scene
        scene_index_offset = 0

    state = TemporalState()
    radar_ptrs: dict[str, str] | None = None

    def on_new_scene(scene_tok: str) -> None:
        nonlocal radar_ptrs
        state.reset()
        state.scene_token = scene_tok
        radar_ptrs = None
        if nusc is not None and (opts.radar_cameras or opts.radar_bev):
            radar_ptrs = {ch: earliest_sample_data_in_scene(nusc, scene_tok, ch) for ch in RADAR_CHANNELS}

    def process_frame(data: dict):
        img_np, l2i, cb_raw, _, _wrapped_norm = extract_sample_arrays(data)
        img_norm = _wrapped_norm or cfg.img_norm
        meta = unwrap_meta(data)

        if opts.crop or opts.resize:
            l2i = adjust_lidar2img_for_crop_resize(
                l2i,
                crop=opts.crop,
                resize=opts.resize,
                tensor_hw=(int(img_np.shape[-2]), int(img_np.shape[-1])),
            )
            img_np = apply_crop_resize_to_batch(img_np, img_norm, opts.crop, opts.resize)

        can_bus_delta = state.update_can_bus_delta(cb_raw)
        use_flag = state.prev_bev is not None
        prev_bev = get_prev_bev(state, (1, cfg.bev_h * cfg.bev_w, cfg.embed_dims))

        if ground_truth_mode:
            if opts.data_type == InferenceDataType.samples:
                assert extract_gt_result_fn is not None
                result = extract_gt_result_fn(data)
            elif interpolate_sweep_gt:
                # Interpolation: sweeps + --interpolate-sweep-annotations → nusc.get_boxes.
                _lt = resolve_lidar_top_sample_data_token(nusc, meta)
                if nusc is None or not _lt:
                    result = dict(
                        boxes_3d=np.zeros((0, 9), dtype=np.float32),
                        scores_3d=np.zeros((0,), dtype=np.float32),
                        labels_3d=np.zeros((0,), dtype=np.int64),
                    )
                else:
                    result = nusc_interp_boxes(
                        nusc,
                        _lt,
                        list(cfg.class_names),
                    )
            else:
                result = dict(
                    boxes_3d=np.zeros((0, 9), dtype=np.float32),
                    scores_3d=np.zeros((0,), dtype=np.float32),
                    labels_3d=np.zeros((0,), dtype=np.int64),
                )
            bev_embed = cls_scores = bbox_preds = None
        else:
            assert frame_runner is not None
            bev_embed, cls_scores, bbox_preds = frame_runner(img_np, l2i, can_bus_delta, prev_bev, use_flag)
            state.prev_bev = bev_embed

            assert opts.score_thr is not None
            result = post_process(
                cls_scores,
                bbox_preds,
                cfg,
                opts.score_thr,
            )

        # ── Modality overlays (lidar / radar / map) ──────────────────────────
        # Resolve the lidar token once — anchors all per-frame overlays.
        lid_tok = resolve_lidar_top_sample_data_token(nusc, meta) if nusc and meta else None

        lidar_xyz = None
        if opts.lidar_cameras or opts.lidar_bev:
            if nusc and lid_tok and opts.lidar_nsweeps > 1:
                xyz_ms, _ = load_nuscenes_lidar_xyz_multisweep(nusc, lid_tok, opts.lidar_nsweeps)
                lidar_xyz = xyz_ms
            if lidar_xyz is None:
                pts_path = meta.get("pts_filename") if meta else None
                if pts_path:
                    p = Path(pts_path)
                    if not p.is_file():
                        p = Path(data_root) / pts_path
                    lidar_xyz = load_nuscenes_lidar_xyz(p)

        radar_xyz = radar_vel = radar_sid = None
        if nusc and lid_tok and (opts.radar_cameras or opts.radar_bev):
            if radar_ptrs:
                # Advance per-channel radar pointers to match the current lidar timestamp.
                master_ts = nusc.get("sample_data", lid_tok)["timestamp"]
                radar_ptrs.update(
                    {ch: advance_sample_data_to_timestamp(nusc, radar_ptrs[ch], master_ts) for ch in RADAR_CHANNELS}
                )
            if opts.radar_nsweeps > 1:
                radar_xyz, radar_sid, radar_vel = accumulate_radar_points_lidar_frame(
                    nusc, lid_tok, radar_ptrs or {}, opts.radar_nsweeps
                )
            else:
                radar_xyz, radar_sid, radar_vel = merge_radar_points_lidar_frame(nusc, lid_tok, radar_ptrs or {})

        map_polys = None
        map_underlay = None
        if opts.map_bev and nusc and lid_tok and meta:
            if opts.map_style_polylines:
                map_polys = build_hdmap_polylines_lidar_xy(
                    nusc,
                    str(meta["scene_token"]),
                    lid_tok,
                    cfg.pc_range,
                )
            else:
                map_underlay = build_map_underlay_for_bev(
                    nusc,
                    str(meta["scene_token"]),
                    lid_tok,
                    cfg.pc_range,
                )

        vis = visualize_frame(
            img_np,
            result,
            l2i,
            img_norm,
            cfg.pc_range,
            lidar_xyz=lidar_xyz,
            overlay_lidar_cam=opts.lidar_cameras,
            overlay_lidar_bev=opts.lidar_bev,
            radar_xyz=radar_xyz,
            radar_vel_xy=radar_vel,
            radar_sensor_ids=radar_sid,
            overlay_radar_cam=opts.radar_cameras,
            overlay_radar_bev=opts.radar_bev,
            radar_draw_velocity=False,
            map_polylines_xy=map_polys,
            map_underlay=map_underlay,
            overlay_map_bev=opts.map_bev,
            img_metas=meta,
        )
        if ground_truth_mode and not pad_with_shape:
            # Camera grid only — rebuild without BEV inset to match legacy GT script.
            from mythic.model_zoo.bevformer.bevformer_lib.custom_utils.ground_truth import (
                build_cam_grid,
            )

            vis = build_cam_grid(img_np, result, l2i, img_norm)

        return vis, result, extract_sample_token(data)

    run_inference_loop(
        loader,
        sf,
        writer,
        scene_counts,
        n_active,
        loop_start_scene,
        cfg.class_names,
        process_frame,
        on_new_scene=on_new_scene,
        scene_index_offset=scene_index_offset,
    )
    writer.finalize()


# ── Subcommand implementations (invoked from thin ``bevformer_inference`` CLI) ─


def run_pytorch_command(opts: CommonRunOpts, *, checkpoint: Path) -> None:
    """Run BEVFormer from a native ``.pth`` checkpoint (``forward_onnx`` export path)."""
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_pth_model(opts.config_path, checkpoint, opts.device)

    def runner(img, l2i, cb, prev, use_pb):
        return pth_run_frame(model, img, l2i, cb, prev, opts.device, use_prev_bev=use_pb)

    shared_nusc = None
    if opts.data_type == InferenceDataType.samples:
        loader, dataset, _, _ = build_dataloader_from_mmcv_config(
            opts.mmcfg,
            resolution_scale_override=opts.resolution_scale,
        )
    else:
        loader, dataset, _, _, shared_nusc = build_sweeps_dataloader(
            opts.mmcfg,
            opts.start_scene,
            opts.end_scene,
            resolution_scale_override=opts.resolution_scale,
        )

    run_video_pipeline(
        backend="PyTorch",
        subcommand="pytorch",
        opts=opts,
        loader=loader,
        dataset=dataset,
        frame_runner=runner,
        summary_model_path=checkpoint,
        nusc=shared_nusc,
    )
    console.print("[green]Done.[/green]")


def run_onnx_runtime_command(opts: CommonRunOpts, *, onnx_path: Path) -> None:
    """Run inference with ONNX Runtime (does not support Mythic ops)."""
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    session = load_onnx_session_or_suggest_torchnet(onnx_path, opts.device)

    def runner(img, l2i, cb, prev, use_pb):
        return onnx_run_frame(session, img, l2i, cb, prev, use_prev_bev=use_pb)

    shared_nusc = None
    if opts.data_type == InferenceDataType.samples:
        loader, dataset, _, _ = build_dataloader_from_mmcv_config(
            opts.mmcfg,
            resolution_scale_override=opts.resolution_scale,
        )
    else:
        loader, dataset, _, _, shared_nusc = build_sweeps_dataloader(
            opts.mmcfg,
            opts.start_scene,
            opts.end_scene,
            resolution_scale_override=opts.resolution_scale,
        )

    run_video_pipeline(
        backend="ONNX Runtime",
        subcommand="onnx",
        opts=opts,
        loader=loader,
        dataset=dataset,
        frame_runner=runner,
        summary_model_path=onnx_path,
        nusc=shared_nusc,
    )
    console.print("[green]Done.[/green]")


def run_torchnet_command(
    opts: CommonRunOpts,
    *,
    onnx_path: Path,
    checkpoint: Optional[Path] = None,
) -> None:
    """Run TorchNet inference from ONNX (+ optional checkpoint overlay); handles Mythic-decorated graphs."""
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    torchnet = build_torchnet_from_onnx(onnx_path, opts.device, checkpoint=checkpoint)

    def runner(img, l2i, cb, prev, use_pb):
        return torchnet_run_frame(
            torchnet,
            img,
            l2i,
            cb,
            prev,
            device=torch.device(opts.device),
            use_prev_bev=use_pb,
        )

    shared_nusc = None
    if opts.data_type == InferenceDataType.samples:
        loader, dataset, _, _ = build_dataloader_from_mmcv_config(
            opts.mmcfg,
            resolution_scale_override=opts.resolution_scale,
        )
    else:
        loader, dataset, _, _, shared_nusc = build_sweeps_dataloader(
            opts.mmcfg,
            opts.start_scene,
            opts.end_scene,
            resolution_scale_override=opts.resolution_scale,
        )

    run_video_pipeline(
        backend="TorchNet",
        subcommand="torchnet",
        opts=opts,
        loader=loader,
        dataset=dataset,
        frame_runner=runner,
        summary_model_path=onnx_path,
        torchnet=torchnet,
        torchnet_checkpoint=checkpoint,
        nusc=shared_nusc,
    )
    console.print("[green]Done.[/green]")


def run_ground_truth_command(
    opts: CommonRunOpts,
    *,
    interpolate_sweep_annotations: bool = False,
    pad_with_shape: bool = True,
) -> None:
    """Render nuScenes ground truth (keyframes or optional sweep interpolation)."""
    from mythic.model_zoo.bevformer.bevformer_lib.custom_utils.ground_truth import (
        inject_ann_info_for_gt,
        patch_pipeline_for_gt,
    )

    if interpolate_sweep_annotations and opts.data_type == InferenceDataType.samples:
        warnings.warn(
            "--interpolate-sweep-annotations has no effect with --data-type samples "
            "(pkl annotations are already keyframe-exact).",
            stacklevel=2,
        )

    opts.output_dir.mkdir(parents=True, exist_ok=True)

    mmcfg = opts.mmcfg
    shared_nusc = None
    if opts.data_type == InferenceDataType.samples:
        patch_pipeline_for_gt(mmcfg)
        loader, dataset, _, _mmc = build_dataloader_from_mmcv_config(
            mmcfg,
            resolution_scale_override=opts.resolution_scale,
        )
        inject_ann_info_for_gt(dataset)
    else:
        loader, dataset, _, _mmc, shared_nusc = build_sweeps_dataloader(
            mmcfg,
            opts.start_scene,
            opts.end_scene,
            resolution_scale_override=opts.resolution_scale,
        )

    run_video_pipeline(
        backend="GroundTruth",
        subcommand="ground-truth",
        opts=opts,
        loader=loader,
        dataset=dataset,
        summary_model_path=opts.config_path,
        ground_truth_mode=True,
        interpolate_sweep_gt=interpolate_sweep_annotations,
        pad_with_shape=pad_with_shape,
        nusc=shared_nusc,
    )
    console.print("[green]Done.[/green]")
