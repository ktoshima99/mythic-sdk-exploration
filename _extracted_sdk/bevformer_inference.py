#!/usr/bin/env -S uv run
"""BEVFormer unified inference / ground-truth CLI.

Four subcommands:
- pytorch
- onnx
- torchnet
- ground-truth

all share a common dataloader / scene loop / visualization pipeline.

Command examples are printed at the bottom of the main ``--help`` output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from mythic.model_zoo.bevformer.bevformer_inference_impl import InferenceDataType

# Defaults shared with ``bevformer_inference_impl`` (impl imports these; keep this module import-light).
DEFAULT_SCORE_THRESHOLD: float = 0.3
RESOLUTION_SCALE_MIN: float = 0.1
RESOLUTION_SCALE_MAX: float = 2.0
DEFAULT_OUTPUT_RESOLUTION_SCALE: float = 0.5


DEFAULT_CONFIG = Path(__file__).resolve().parent / "bevformer_lib/projects/configs/bevformer/bevformer_tiny.py"

_DEFAULT_EXAMPLE_CONFIG = "bevformer_lib/projects/configs/bevformer/bevformer_tiny.py"

# Typer joins epilog "paragraphs" (chunks split on ``\\n\\n``) with newlines; single ``\\n``
# inside a chunk becomes a space—so one line per chunk. Use ``\\n\\n\\n\\n`` for a blank
# line between examples. See ``typer.rich_utils`` (epilogue handling).
_CLI_USAGE_EXAMPLES = (
    "[bold]Examples[/bold]\n\n"
    "[dim]uv run python -m mythic.model_zoo.bevformer.bevformer_inference pytorch "
    "epoch_24.pth[/dim]\n\n"
    f"[dim]    {_DEFAULT_EXAMPLE_CONFIG}[/dim]\n\n\n\n"
    "[dim]uv run python -m mythic.model_zoo.bevformer.bevformer_inference onnx "
    "model.onnx[/dim]\n\n"
    f"[dim]    {_DEFAULT_EXAMPLE_CONFIG}[/dim]\n\n\n\n"
    "[dim]uv run python -m mythic.model_zoo.bevformer.bevformer_inference torchnet "
    "model.onnx[/dim]\n\n"
    f"[dim]    {_DEFAULT_EXAMPLE_CONFIG} --checkpoint epoch_24.pth[/dim]\n\n\n\n"
    "[dim]uv run python -m mythic.model_zoo.bevformer.bevformer_inference ground-truth "
    f"{_DEFAULT_EXAMPLE_CONFIG}[/dim]\n\n"
    "[dim]    --data-type sweeps --interpolate-sweep-annotations[/dim]"
)

app = typer.Typer(
    name="bevformer_inference",
    help="Unified BEVFormer inference (pytorch / ORT / TorchNet) and ground-truth videos.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=_CLI_USAGE_EXAMPLES,
)

# ── Shared Typer option aliases (single source of truth for DRY CLI) ──────────

DataTypeOpt = Annotated[
    InferenceDataType,
    typer.Option(
        "--data-type",
        help="samples (2Hz keyframes from pkl) or sweeps (~12Hz via nuScenes API walk).",
        case_sensitive=False,
        show_choices=True,
    ),
]
StartSceneOpt = Annotated[
    int,
    typer.Option(
        "--start-scene",
        help="Inclusive start scene index (0-based).",
        min=0,
    ),
]
EndSceneOpt = Annotated[
    Optional[int],
    typer.Option(
        "--end-scene",
        help="Exclusive end scene index (omit to run all remaining scenes).",
        min=0,
    ),
]
DeviceOpt = Annotated[
    Optional[str],
    typer.Option(
        "--device",
        help="Torch or ORT device (e.g. cuda:0, cpu). Default: cuda:0 if CUDA is available, else cpu.",
    ),
]
GtDeviceOpt = Annotated[
    str,
    typer.Option(
        "--device",
        help="Device string (ground-truth rendering defaults to cpu).",
    ),
]
CropOpt = Annotated[
    Optional[str],
    typer.Option("--crop", help="Crop before resize: 'x,y,w,h' pixels."),
]
ResizeOpt = Annotated[
    Optional[str],
    typer.Option("--resize", help="Resize after crop: 'w,h' pixels."),
]
ResScaleOpt = Annotated[
    Optional[float],
    typer.Option(
        "--resolution-scale",
        help="Override RandomScaleImageMultiViewImage.scales in the test pipeline.",
        min=RESOLUTION_SCALE_MIN,
        max=RESOLUTION_SCALE_MAX,
    ),
]
VisOpt = Annotated[
    bool,
    typer.Option(
        " /--no-vis",
        help=("Skip those visualization images/videos"),
        show_default=False,
    ),
]
SaveJsonOpt = Annotated[
    bool,
    typer.Option("--save-json", help="Save nuScenes-format predictions JSON."),
]
FpsOpt = Annotated[
    Optional[int],
    typer.Option(
        "--fps",
        help="Output video FPS. Default: 2 for samples, 12 for sweeps (see summary table).",
        min=1,
    ),
]
ScoreThrOpt = Annotated[
    float,
    typer.Option(
        "--score-thr",
        help="Detection score threshold for post_process / NMS.",
        min=0.0,
        max=1.0,
    ),
]
_OUTPUT_DIR_HELP = (
    "Directory for frames, videos, and optional JSON. "
    "If omitted: bevformer-inference-results/"
    "<subcommand>-<stem>-<W>x<H>-(samples|sweeps)-mod-…-<YYYYMMDD-HHMMSS> "
    "<stem> is the model/checkpoint/config filename as relevant"
)
OutputDirOpt = Annotated[
    Optional[Path],
    typer.Option("-o", "--output-dir", help=_OUTPUT_DIR_HELP),
]
LidarCamOpt = Annotated[
    Optional[bool],
    typer.Option(
        "--lidar-cameras/--no-lidar-cameras",
        help="Project LIDAR_TOP onto cameras (default: cfg.input_modality.use_lidar).",
    ),
]
LidarBevOpt = Annotated[
    Optional[bool],
    typer.Option(
        "--lidar-bev/--no-lidar-bev",
        help="Scatter LiDAR on BEV inset (default: cfg.input_modality.use_lidar).",
    ),
]
RadarCamOpt = Annotated[
    Optional[bool],
    typer.Option(
        "--radar-cameras/--no-radar-cameras",
        help="Project RADAR returns onto cameras (default: cfg.input_modality.use_radar).",
    ),
]
RadarBevOpt = Annotated[
    Optional[bool],
    typer.Option(
        "--radar-bev/--no-radar-bev",
        help="Draw radar on BEV (default: cfg.input_modality.use_radar).",
    ),
]
MapBevOpt = Annotated[
    Optional[bool],
    typer.Option(
        "--map-bev/--no-map-bev",
        help="Draw HD-map polylines on BEV only (default: cfg.input_modality.use_map).",
    ),
]
MapStylePolylinesOpt = Annotated[
    bool,
    typer.Option(
        "--map-style-polylines",
        help=(
            "Use simple Shapely+cv2.polylines map rendering instead of the default "
            "devkit raster (get_map_geom filled polygons)."
        ),
    ),
]
LidarNSweepsOpt = Annotated[
    int,
    typer.Option(
        "--lidar-nsweeps",
        help="N past LIDAR_TOP sweeps to aggregate via LidarPointCloud (1 = single frame).",
        min=1,
    ),
]
RadarNSweepsOpt = Annotated[
    int,
    typer.Option(
        "--radar-nsweeps",
        help="N past sweeps per RADAR_* channel to aggregate (default: 1 = single frame).",
        min=1,
    ),
]
OutputResScaleOpt = Annotated[
    float,
    typer.Option(
        "--output-resolution-scale",
        help="Scale saved frames and video by this factor before writing to disk (0.1–2.0). Default: 0.5.",
        min=0.1,
        max=2.0,
    ),
]

# Paths to weights / ONNX / mmcv configs must be files (not directories).
PthCheckpointArg = Annotated[
    Path,
    typer.Argument(
        help=".pth checkpoint",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
]
MmdetConfigArg = Annotated[
    Path,
    typer.Argument(
        help="mmdet3d .py config",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
]
OnnxModelArg = Annotated[
    Path,
    typer.Argument(
        help=".onnx model",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
]
OnnxTorchNetArg = Annotated[
    Path,
    typer.Argument(
        help=".onnx for TorchNet",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
]
OptionalPthOverlayOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--checkpoint",
        help="Optional .pth overlaid with strict=False (missing keys logged).",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
]


@app.command(
    "pytorch",
    no_args_is_help=True,
    help="Run BEVFormer from a native .pth checkpoint (forward_onnx export path).",
)
def cmd_pytorch(
    ctx: typer.Context,
    checkpoint: PthCheckpointArg,
    config: MmdetConfigArg,
    *,
    output_dir: OutputDirOpt = None,
    data_type: DataTypeOpt = InferenceDataType.samples,
    start_scene: StartSceneOpt = 0,
    end_scene: EndSceneOpt = None,
    score_thr: ScoreThrOpt = DEFAULT_SCORE_THRESHOLD,
    device: DeviceOpt = None,
    fps: FpsOpt = None,
    crop: CropOpt = None,
    resize: ResizeOpt = None,
    resolution_scale: ResScaleOpt = None,
    vis: VisOpt = True,
    save_json: SaveJsonOpt = False,
    lidar_cameras: LidarCamOpt = None,
    lidar_bev: LidarBevOpt = None,
    radar_cameras: RadarCamOpt = None,
    radar_bev: RadarBevOpt = None,
    map_bev: MapBevOpt = None,
    map_style_polylines: MapStylePolylinesOpt = False,
    lidar_nsweeps: LidarNSweepsOpt = 1,
    radar_nsweeps: RadarNSweepsOpt = 1,
    output_resolution_scale: OutputResScaleOpt = DEFAULT_OUTPUT_RESOLUTION_SCALE,
) -> None:
    """Run BEVFormer inference from a native PyTorch checkpoint."""
    from mythic.model_zoo.bevformer import bevformer_inference_impl as impl

    opts = impl.resolve_common_opts(ctx, subcommand="pytorch", results_dir_stem=checkpoint.stem)
    impl.run_pytorch_command(opts, checkpoint=checkpoint)


@app.command(
    "onnx",
    no_args_is_help=True,
    help=(
        "Run inference with ONNX Runtime. ONNX models with Mythic ops "
        "should use the `torchnet` subcommand instead."
    ),
)
def cmd_onnx_runtime(
    ctx: typer.Context,
    onnx_path: OnnxModelArg,
    config: MmdetConfigArg,
    *,
    output_dir: OutputDirOpt = None,
    data_type: DataTypeOpt = InferenceDataType.samples,
    start_scene: StartSceneOpt = 0,
    end_scene: EndSceneOpt = None,
    score_thr: ScoreThrOpt = DEFAULT_SCORE_THRESHOLD,
    device: DeviceOpt = None,
    fps: FpsOpt = None,
    crop: CropOpt = None,
    resize: ResizeOpt = None,
    resolution_scale: ResScaleOpt = None,
    vis: VisOpt = True,
    save_json: SaveJsonOpt = False,
    lidar_cameras: LidarCamOpt = None,
    lidar_bev: LidarBevOpt = None,
    radar_cameras: RadarCamOpt = None,
    radar_bev: RadarBevOpt = None,
    map_bev: MapBevOpt = None,
    map_style_polylines: MapStylePolylinesOpt = False,
    lidar_nsweeps: LidarNSweepsOpt = 1,
    radar_nsweeps: RadarNSweepsOpt = 1,
    output_resolution_scale: OutputResScaleOpt = DEFAULT_OUTPUT_RESOLUTION_SCALE,
) -> None:
    """Run BEVFormer inference with ONNX Runtime."""
    from mythic.model_zoo.bevformer import bevformer_inference_impl as impl

    opts = impl.resolve_common_opts(ctx, subcommand="onnx", results_dir_stem=onnx_path.stem)
    impl.run_onnx_runtime_command(opts, onnx_path=onnx_path)


@app.command(
    "torchnet",
    no_args_is_help=True,
    help=("Run TorchNet from ONNX (with optional .pth weights checkpoint overlay)"),
)
def cmd_torchnet(
    ctx: typer.Context,
    onnx_path: OnnxTorchNetArg,
    config: MmdetConfigArg,
    *,
    checkpoint: OptionalPthOverlayOpt = None,
    output_dir: OutputDirOpt = None,
    data_type: DataTypeOpt = InferenceDataType.samples,
    start_scene: StartSceneOpt = 0,
    end_scene: EndSceneOpt = None,
    score_thr: ScoreThrOpt = DEFAULT_SCORE_THRESHOLD,
    device: DeviceOpt = None,
    fps: FpsOpt = None,
    crop: CropOpt = None,
    resize: ResizeOpt = None,
    resolution_scale: ResScaleOpt = None,
    vis: VisOpt = True,
    save_json: SaveJsonOpt = False,
    lidar_cameras: LidarCamOpt = None,
    lidar_bev: LidarBevOpt = None,
    radar_cameras: RadarCamOpt = None,
    radar_bev: RadarBevOpt = None,
    map_bev: MapBevOpt = None,
    map_style_polylines: MapStylePolylinesOpt = False,
    lidar_nsweeps: LidarNSweepsOpt = 1,
    radar_nsweeps: RadarNSweepsOpt = 1,
    output_resolution_scale: OutputResScaleOpt = DEFAULT_OUTPUT_RESOLUTION_SCALE,
) -> None:
    """Run BEVFormer inference via TorchNet (ONNX graph with optional checkpoint overlay)."""
    from mythic.model_zoo.bevformer import bevformer_inference_impl as impl

    results_stem = onnx_path.stem if checkpoint is None else f"{onnx_path.stem}-{checkpoint.stem}"
    opts = impl.resolve_common_opts(ctx, subcommand="torchnet", results_dir_stem=results_stem)
    impl.run_torchnet_command(opts, onnx_path=onnx_path, checkpoint=checkpoint)


@app.command(
    "ground-truth",
    no_args_is_help=True,
    help="Render nuScenes ground-truth videos",
)
def cmd_ground_truth(
    ctx: typer.Context,
    config: MmdetConfigArg = DEFAULT_CONFIG,
    *,
    output_dir: OutputDirOpt = None,
    data_type: DataTypeOpt = InferenceDataType.samples,
    start_scene: StartSceneOpt = 0,
    end_scene: EndSceneOpt = None,
    device: GtDeviceOpt = "cpu",
    interpolate_sweep_annotations: Annotated[
        bool,
        typer.Option(
            "--interpolate-sweep-annotations",
            help=(
                "For --data-type sweeps only: draw devkit-interpolated GT via nusc.get_boxes. "
                "No effect on samples (exact pkl annotations)."
            ),
        ),
    ] = False,
    pad_with_shape: Annotated[
        bool,
        typer.Option(
            " /--no-pad-with-shape",
            help="Include BEV inset of GT boxes (matches inference canvas size).",
        ),
    ] = True,
    fps: FpsOpt = None,
    crop: CropOpt = None,
    resize: ResizeOpt = None,
    resolution_scale: ResScaleOpt = None,
    vis: VisOpt = True,
    lidar_cameras: LidarCamOpt = None,
    lidar_bev: LidarBevOpt = None,
    radar_cameras: RadarCamOpt = None,
    radar_bev: RadarBevOpt = None,
    map_bev: MapBevOpt = None,
    map_style_polylines: MapStylePolylinesOpt = False,
    lidar_nsweeps: LidarNSweepsOpt = 1,
    radar_nsweeps: RadarNSweepsOpt = 1,
    output_resolution_scale: OutputResScaleOpt = DEFAULT_OUTPUT_RESOLUTION_SCALE,
) -> None:
    """Render nuScenes ground-truth visualization videos."""
    from mythic.model_zoo.bevformer import bevformer_inference_impl as impl

    opts = impl.resolve_common_opts(ctx, subcommand="ground-truth", results_dir_stem="")
    impl.run_ground_truth_command(
        opts,
        interpolate_sweep_annotations=interpolate_sweep_annotations,
        pad_with_shape=pad_with_shape,
    )


if __name__ == "__main__":
    app()
