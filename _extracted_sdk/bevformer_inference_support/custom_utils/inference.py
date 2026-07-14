"""BEVFormer inference helpers.

ONNX session loading, prev-BEV state management, ORT frame execution,
scene-range filtering, inference loop orchestration, and run summaries.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .data_loading import TemporalState, extract_scene_token
from .result_writer import ResultWriter

console = Console()


# ── ONNX session ──────────────────────────────────────────────────────────────


def load_onnx_session(onnx_path: Path, device: str = "cuda:0"):
    """Load ONNX model into an OnnxRuntime InferenceSession."""

    available = ort.get_available_providers()
    use_cuda = "cuda" in device.lower() and "CUDAExecutionProvider" in available

    # Parse the device index from strings like "cuda:1"; default to 0.
    device_id = 0
    if use_cuda and ":" in device:
        try:
            device_id = int(device.split(":")[-1])
        except ValueError:
            pass

    providers = (
        [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    # Grow the arena by exactly what is needed rather than
                    # doubling, which can request one giant contiguous block.
                    "arena_extend_strategy": "kSameAsRequested",
                },
            ),
            "CPUExecutionProvider",
        ]
        if use_cuda
        else ["CPUExecutionProvider"]
    )

    console.print(f"ORT version : {ort.__version__}")
    console.print(f"Available   : {available}")
    console.print(f"Using       : {providers}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)


def load_onnx_session_or_suggest_torchnet(onnx_path: Path, device: str = "cuda:0"):
    """Load ONNX via ORT. On failure (unknown ops, invalid graph), suggest ``torchnet`` subcommand.

    Use this from the ``onnx`` CLI only. For eval or tools that must see the raw
    ORT error, call :func:`load_onnx_session` directly.

    ORT is the source of truth for Mythic / custom-op incompatibility — no graph pre-scan.
    """
    try:
        return load_onnx_session(onnx_path, device)
    except Exception as e:
        msg = (
            f"\n[bold red]onnxruntime failed to load[/bold red] [cyan]{onnx_path}[/cyan].\n"
            "The graph likely contains ops that ONNX Runtime does not register "
            "(e.g. Mythic-specific ops). Use the [bold]torchnet[/bold] subcommand instead:\n"
            f"  [dim]uv run python -m mythic.model_zoo.bevformer.bevformer_inference torchnet "
            f"{onnx_path} <config.py>[/dim]\n\n"
            f"[yellow]Original error:[/yellow] {e!r}"
        )
        console.print(msg)
        raise RuntimeError(
            "onnxruntime failed to load model (see message above); use the torchnet subcommand"
        ) from e


def _find_model_zoo_root_with_hydra() -> Path:
    """Locate repo root that contains ``configs/bevformer`` (for Hydra TorchNet session)."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "configs" / "bevformer").is_dir():
            return p
        p = p.parent
    raise RuntimeError(
        "Could not find configs/bevformer — is this running from a model-zoo checkout?"
    )


def build_torchnet_from_onnx(
    onnx_path: Path,
    device: str,
    *,
    checkpoint: Path | None = None,
    hydra_config_name: str = "bevformer_tiny",
    return_session: bool = False,
) -> Any:
    """Build munc ``TorchNet`` from ONNX + optional training snapshot (``strict=False``).

    Lazy-imports munc / Hydra inside this function so ``import custom_utils.inference`` does not
    pull TorchNet on ONNX-only code paths.

    ``strict=False`` is intentional: TorchNet submodule names may differ from the keys in a raw
    ``.pth`` produced outside this wrapper. If ``checkpoint`` is None, weights come from the ONNX
    initializers only.

    Returns:
        ``TorchNet``, or ``(TorchNet, session)`` when ``return_session=True`` (munc session for
        ``save_torch_to_onnx_object`` / ``session.model.save``).
    """
    import os

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from munc.cli.helpers import SessionFromConfig, add_munc_configs_to_hydra_search_path
    from mythic.model_zoo.common.utils import add_model_zoo_common_configs_to_hydra_search_path
    from omegaconf import DictConfig, OmegaConf

    add_munc_configs_to_hydra_search_path()
    add_model_zoo_common_configs_to_hydra_search_path()

    os.environ["MYTHIC_SDK_ROOT"] = ""
    hydra_dir = _find_model_zoo_root_with_hydra() / "configs" / "bevformer"
    torch_device = torch.device(device)


    if GlobalHydra.instance().is_initialized():
        hydra_cfg: DictConfig = compose(config_name=hydra_config_name, overrides=[])
    else:
        with initialize_config_dir(version_base=None, config_dir=str(hydra_dir)):
            hydra_cfg: DictConfig = compose(config_name=hydra_config_name, overrides=[])
    OmegaConf.resolve(hydra_cfg)
    session = SessionFromConfig(
        {
            "src": str(onnx_path),
            "torchnet": OmegaConf.to_container(hydra_cfg.default_torchnet),
        },
        save_model=False,
    )
    torchnet = session.make_torch_net()
    torchnet.to(torch_device)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=torch_device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = torchnet.load_state_dict(state_dict, strict=True)
        if missing:
            console.print(
                f"[yellow]Missing keys: {len(missing)} (first 5: {missing[:5]})[/yellow]"
            )
        if unexpected:
            console.print(
                f"[yellow]Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})[/yellow]"
            )

    if return_session:
        return torchnet, session
    return torchnet


def build_pth_model(config_path: Path, checkpoint_path: Path, device: str):
    """Build mmdet3d BEVFormer from config and load ``.pth`` checkpoint."""
    import torch
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    from .data_loading import load_mmcv_config

    cfg = load_mmcv_config(config_path)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint_path), map_location=device)
    model.eval()
    mod = model.module if hasattr(model, "module") else model
    mod.to(torch.device(device))
    return model


def pth_run_frame(
    model,
    img_np: np.ndarray,
    lidar2img: np.ndarray,
    can_bus: np.ndarray,
    prev_bev: np.ndarray,
    device: str,
    *,
    use_prev_bev: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One forward via ``model.forward_onnx`` (pure PyTorch path).

    I/O contract matches :func:`onnx_run_frame` / ``torchnet_run_frame``: ``bev`` batch-first;
    ``cls`` / ``box`` decoder-first ``[num_dec, B, Q, C]`` as numpy.

    First frame of each scene: pass ``use_prev_bev=False`` with ``prev_bev`` zeros so
    TemporalSelfAttention matches the original ``prev_bev=None`` branch (see ``get_prev_bev``).
    """
    import torch

    dev = torch.device(device)
    export_model = model.module if hasattr(model, "module") else model
    img = torch.from_numpy(np.ascontiguousarray(img_np)).to(dev)
    lidar2img_t = torch.from_numpy(np.ascontiguousarray(lidar2img)).to(dev)
    can_bus_t = torch.from_numpy(np.ascontiguousarray(can_bus)).to(dev)
    prev_bev_t = torch.from_numpy(np.ascontiguousarray(prev_bev)).to(dev)
    use_t = torch.tensor(
        [1.0 if use_prev_bev else 0.0], dtype=torch.float32, device=dev
    )

    export_model.eval()
    with torch.no_grad():
        bev, cls, box = export_model.forward_onnx(
            img, can_bus_t, lidar2img_t, prev_bev_t, use_t
        )

    cls = cls.permute(1, 0, 2, 3)
    box = box.permute(1, 0, 2, 3)
    return (
        bev.detach().cpu().numpy(),
        cls.detach().cpu().numpy(),
        box.detach().cpu().numpy(),
    )


# ── prev_bev helpers ──────────────────────────────────────────────────────────


def get_prev_bev(state: TemporalState, shape: tuple[int, ...]) -> np.ndarray:
    """Return the prev_bev array for the current frame.

    Returns zeros on the first frame of each scene (`state.prev_bev is None`),
    otherwise reshapes `state.prev_bev` to `shape`.

    Shape convention: batch-first `(B, bev_h*bev_w, embed_dims)`.

    Important: always returns a tensor (never None).  Pass `use_prev_bev=False`
    separately to `onnx_run_frame` / `torchnet_run_frame` on the first frame so
    the model can branch on it via `torch.where` in TemporalSelfAttention — matching
    the original PyTorch `prev_bev=None` code path (encoder.py:265-273, TSA:177-180).
    """
    if state.prev_bev is None:
        return np.zeros(shape, dtype=np.float32)
    return state.prev_bev.reshape(shape)


# NOTE: Also used by ``onnx_eval.py``. The unified ``bevformer_inference`` CLI uses
# ``get_prev_bev`` with shape from ``InferenceConfig`` instead. If ``onnx_eval`` is
# refactored later, this helper might become removable.
def onnx_prev_bev(
    state: TemporalState,
    session: Any,
    bev_h: int,
    bev_w: int,
    embed_dims: int,
) -> np.ndarray:
    """Return the prev_bev array shaped to match the ONNX session input.

    Resolves dynamic dims from the session's input spec, then delegates to
    `get_prev_bev`.  Always returns batch-first `(B, bev_h*bev_w, embed_dims)`.

    See `get_prev_bev` for the use_prev_bev / None-vs-zeros note.
    """
    # default = (bev_h * bev_w, 1, embed_dims)  # original: batch was dim 1
    default = (1, bev_h * bev_w, embed_dims)
    inp = next((i for i in session.get_inputs() if i.name == "prev_bev"), None)
    if inp is not None:
        shape = tuple(
            d if (isinstance(d, int) and d > 0) else default[j]
            for j, d in enumerate(inp.shape)
        )
    else:
        shape = default
    return get_prev_bev(state, shape)


# ── ORT frame execution ───────────────────────────────────────────────────────


def onnx_run_frame(
    session: Any,
    img_np: np.ndarray,
    lidar2img: np.ndarray,
    can_bus: np.ndarray,
    prev_bev: np.ndarray,
    use_prev_bev: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one ONNX inference step. Returns (bev_embed, cls_scores, bbox_preds).

    Args:
        session: ORT InferenceSession.
        img_np: (1, N_cams, C, H, W) float32.
        lidar2img: (1, N_cams, 4, 4) float32.
        can_bus: (1, 18) float32 delta (zeros on first frame of scene).
        prev_bev: (1, bev_h*bev_w, embed_dims) float32. Batch-first, zeros on first frame.
        use_prev_bev: False on the first frame of each scene so the model matches the
            original prev_bev=None code path in TemporalSelfAttention.

    Returns:
        bev_embed:  (B, bev_h*bev_w, embed_dims) — batch-first, fed back as next prev_bev.
        cls_scores: (num_decoder_layers, B, num_query, num_classes) — decoder-first for post_process/loss.
        bbox_preds: (num_decoder_layers, B, num_query, code_size)   — decoder-first for post_process/loss.

    Output layout note: the ONNX model emits batch-first (B, num_dec, Q, C) for cls/bbox.
    We transpose to decoder-first (num_dec, B, Q, C) here so callers match the original
    pts_bbox_head.loss() / post_process() expectations. bev_embed stays batch-first for the
    prev_bev round-trip. The '6' in num_dec is decoder layers, NOT batch size.
    """
    names = {i.name for i in session.get_inputs()}
    feeds: dict[str, np.ndarray] = {}
    if "img" in names:
        feeds["img"] = img_np
    if "lidar2img" in names:
        feeds["lidar2img"] = lidar2img
    if "can_bus" in names:
        feeds["can_bus"] = can_bus
    if "prev_bev" in names:
        feeds["prev_bev"] = prev_bev
    if "use_prev_bev" in names:
        feeds["use_prev_bev"] = np.array([1.0 if use_prev_bev else 0.0], dtype=np.float32)
    if "img_shape" in names:
        # [height, width] from img_np (1, N, C, H, W)
        feeds["img_shape"] = np.array(
            [img_np.shape[-2], img_np.shape[-1]], dtype=np.float32
        )
    outs = session.run(None, feeds)
    # return outs[0], outs[1], outs[2]  # original: all in internal layout (batch not first)
    bev_embed = outs[0]                               # [B, bev_h*bev_w, embed]  — batch-first
    cls_scores = np.transpose(outs[1], (1, 0, 2, 3))  # [B, num_decoder_layers, Q, C] -> [num_decoder_layers, B, Q, C]
    bbox_preds = np.transpose(outs[2], (1, 0, 2, 3))  # [B, num_decoder_layers, Q, C] -> [num_decoder_layers, B, Q, C]
    return bev_embed, cls_scores, bbox_preds


def build_torchnet_inputs(
    torchnet,
    curr_img,
    lidar2img,
    can_bus,
    *,
    H: int,
    W: int,
    B: int,
    prev_bev,
    mmcv_config,
    device,
) -> dict:
    """Build the TorchNet input dict for one forward pass.

    ``prev_bev=None`` triggers zeros + ``use_prev_bev=0.0`` (first frame of scene).
    """
    import torch

    ext_inputs = set(torchnet.external_input_names)
    inputs: dict = {}
    if "img" in ext_inputs:
        inputs["img"] = curr_img
    if "lidar2img" in ext_inputs:
        inputs["lidar2img"] = lidar2img
    if "can_bus" in ext_inputs:
        inputs["can_bus"] = can_bus
    if "img_shape" in ext_inputs:
        inputs["img_shape"] = torch.tensor([H, W], dtype=torch.float32, device=device).expand(B, -1)
    if "prev_bev" in ext_inputs:
        if prev_bev is not None:
            inputs["prev_bev"] = prev_bev
        else:
            bev_h = mmcv_config.model.pts_bbox_head.bev_h
            bev_w = mmcv_config.model.pts_bbox_head.bev_w
            embed_dims = mmcv_config.model.pts_bbox_head.transformer.embed_dims
            inputs["prev_bev"] = torch.zeros(
                (B, bev_h * bev_w, embed_dims), dtype=torch.float32, device=device
            )
    if "use_prev_bev" in ext_inputs:
        inputs["use_prev_bev"] = torch.tensor(
            [1.0 if prev_bev is not None else 0.0], dtype=torch.float32, device=device
        )
    return inputs


def torchnet_run_frame(
    torchnet,
    img_np: np.ndarray,
    lidar2img: np.ndarray,
    can_bus: np.ndarray,
    prev_bev: np.ndarray,
    device=None,
    training: bool = False,
    use_prev_bev: bool = True,
) -> tuple:
    """Run one TorchNet forward pass.

    I/O contract matches :func:`onnx_run_frame` / :func:`pth_run_frame`: ``bev`` batch-first;
    ``cls`` / ``box`` decoder-first ``[num_dec, B, Q, C]`` as numpy (or tensors when
    ``training=True``).
    """
    import torch

    resolved_device = device if device is not None else torchnet.device
    names = set(torchnet.external_input_names)

    def _to_t(x: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).to(resolved_device)

    inputs: dict = {}
    if "img" in names:
        inputs["img"] = _to_t(img_np)
    if "lidar2img" in names:
        inputs["lidar2img"] = _to_t(lidar2img)
    if "can_bus" in names:
        inputs["can_bus"] = _to_t(can_bus)
    if "prev_bev" in names:
        inputs["prev_bev"] = _to_t(prev_bev)
    if "use_prev_bev" in names:
        inputs["use_prev_bev"] = torch.tensor(
            [1.0 if use_prev_bev else 0.0], dtype=torch.float32, device=resolved_device
        )
    if "img_shape" in names:
        inputs["img_shape"] = torch.tensor(
            [img_np.shape[-2], img_np.shape[-1]], dtype=torch.float32, device=resolved_device
        )

    if not training:
        torchnet.eval()
        with torch.no_grad():
            outputs = torchnet(inputs)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise RuntimeError(
                f"TorchNet expected ≥3 outputs [bev_embed, cls, box], got {type(outputs)}"
            )
        bev, cls, box = outputs[0], outputs[1], outputs[2]
        cls = cls.permute(1, 0, 2, 3)  # [B, num_dec, Q, C] → [num_dec, B, Q, C]
        box = box.permute(1, 0, 2, 3)
        return (
            bev.detach().cpu().numpy(),
            cls.detach().cpu().numpy(),
            box.detach().cpu().numpy(),
        )
    else:
        outputs = torchnet(inputs)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise RuntimeError(
                f"TorchNet expected ≥3 outputs [bev_embed, cls, box], got {type(outputs)}"
            )
        bev, cls, box = outputs[0], outputs[1], outputs[2]
        return bev, cls.permute(1, 0, 2, 3), box.permute(1, 0, 2, 3)


# ── Scene-range filter ────────────────────────────────────────────────────────


class SceneFilter:
    """Tracks scene transitions and filters by index range."""

    def __init__(self, start: int, end: int | None) -> None:
        self.start = start
        self.end = end
        self._count = -1
        self._current = None

    @property
    def scene_index(self) -> int:
        return self._count

    def is_active(self) -> bool:
        if self._count < self.start:
            return False
        if self.end is not None and self._count >= self.end:
            return False
        return True

    def is_done(self) -> bool:
        return self.end is not None and self._count >= self.end

    def update(self, scene_token: str) -> bool:
        """Returns True if this is a new scene."""
        if scene_token == self._current:
            return False
        self._current = scene_token
        self._count += 1
        return True


# ── Inference loop ────────────────────────────────────────────────────────────


def modality_out_dir_suffix(
    *,
    lidar_cameras: bool,
    lidar_bev: bool,
    radar_cameras: bool,
    radar_bev: bool,
    map_bev: bool,
) -> str:
    """Path token for enabled visualization overlays: lc lb rc rb mb; ``mod-none`` if all off."""
    tags: list[str] = []
    if lidar_cameras:
        tags.append("lc")
    if lidar_bev:
        tags.append("lb")
    if radar_cameras:
        tags.append("rc")
    if radar_bev:
        tags.append("rb")
    if map_bev:
        tags.append("mb")
    tags.sort()
    return "mod-none" if not tags else "mod-" + "-".join(tags)


def default_out_dir(
    stem: str,
    explicit: Path | None,
    effective_resolution: tuple[int, int],
    *,
    base_dir: str = "onnx-inference-results",
    subcommand_prefix: str | None = None,
    data_variant: str = "samples",
    lidar_cameras: bool = False,
    lidar_bev: bool = False,
    radar_cameras: bool = False,
    radar_bev: bool = False,
    map_bev: bool = False,
) -> Path:
    """Auto-pick an output folder under ``base_dir`` when ``explicit`` is None.

    Name pattern:
    ``[{subcommand_prefix}-]{stem}-{W}x{H}-{samples|sweeps}-{mod-…}-{YYYYMMDD-HHMMSS}``

    Overlay token is ``mod-none`` or sorted ``mod-lb-lc-…`` (lc=lidar on cameras, lb=lidar BEV,
    rc/rb=radar, mb=map BEV).
    """
    if explicit is not None:
        return Path(explicit) if isinstance(explicit, str) else explicit
    variant = data_variant.strip().lower()
    mod = modality_out_dir_suffix(
        lidar_cameras=lidar_cameras,
        lidar_bev=lidar_bev,
        radar_cameras=radar_cameras,
        radar_bev=radar_bev,
        map_bev=map_bev,
    )
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    res_suffix = f"{effective_resolution[0]}x{effective_resolution[1]}"
    core = f"{stem}-{res_suffix}" if stem else res_suffix
    name = f"{core}-{variant}-{mod}-{ts}"
    if subcommand_prefix:
        name = f"{subcommand_prefix}-{name}"
    return Path(base_dir) / name


def run_inference_loop(
    loader,
    sf: SceneFilter,
    writer: ResultWriter,
    scene_counts: dict[str, int],
    n_active_scenes: int,
    start_scene: int,
    class_names: list[str],
    process_frame: Callable,
    on_new_scene: Callable | None = None,
    show_progress: bool = True,
    scene_index_offset: int = 0,
) -> None:
    """Shared scene-iteration loop for ONNX and TorchNet inference.

    Args:
        process_frame: called for each active frame.
            Signature: (data) -> (vis_img | None, result_dict, sample_token | None)
        on_new_scene: optional callback called on each new scene (before is_active check).
            Signature: (scene_token) -> None
        show_progress: if False (e.g. distributed non-rank-0), no progress bar is shown.
        scene_index_offset: added to sf.scene_index when naming per-scene output dirs.
            Use when the dataloader has been pre-sliced and sf counts from 0 but the
            user-facing scene indices should reflect the absolute offset.
    """
    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        if show_progress
        else contextlib.nullcontext()
    )

    with progress_cm as progress:
        overall = None
        scene_t = None
        if progress is not None:
            overall = progress.add_task("[cyan]Frames", total=len(loader))
            scene_t = progress.add_task("", total=None, visible=False)

        for data in loader:
            scene_token = extract_scene_token(data)
            new_scene: bool = sf.update(scene_token)

            if new_scene:
                if sf.scene_index > 0 and sf.scene_index - 1 >= start_scene:
                    writer.end_scene(progress=progress)

                if sf.is_done():
                    break

                if on_new_scene:
                    on_new_scene(scene_token)

                if not sf.is_active():
                    if progress is not None:
                        progress.advance(overall)
                    continue

                writer.begin_scene(scene_token, sf.scene_index + scene_index_offset)
                if progress is not None:
                    active_num = sf.scene_index - start_scene + 1
                    progress.update(
                        overall,
                        description=f"[cyan]Scene {active_num}/{n_active_scenes} | Frames",
                    )
                    progress.reset(
                        scene_t,
                        description=f"[green]  └ {scene_token[:16]}",
                        total=scene_counts.get(scene_token),
                        visible=True,
                    )

            if not sf.is_active():
                if progress is not None:
                    progress.advance(overall)
                continue
            if sf.is_done():
                break

            vis, result, sample_token = process_frame(data)
            writer.write_frame(
                vis, result, sample_token=sample_token, class_names=class_names
            )
            if progress is not None:
                progress.advance(overall)
                progress.advance(scene_t)

        if sf.is_active():
            writer.end_scene(progress=progress)


# ── Run summary ───────────────────────────────────────────────────────────────


def print_run_summary(
    backend: str,
    device: str,
    model_path: Path,
    model_name: str | None,
    output_dir: Path,
    img_norm: dict,
    pc_range: list[float],
    score_thr: float,
    crop: tuple | None,
    resize: tuple | None,
    start_scene: int,
    end_scene: int | None,
    src_img_size: tuple[int, int] = (1600, 900),
    model_img_size: tuple[int, int] | None = None,
    model_img_pipeline: str
    | None = None,  # e.g. "1600x900 → x0.5 → 800x450 → pad÷32 → 800x480"
    extra: dict | None = None,
) -> None:
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    norm_str = (
        f"mean={img_norm.get('mean', '?')}  "
        f"std={img_norm.get('std', '?')}  "
        f"to_rgb={img_norm.get('to_rgb', '?')}"
    )

    if crop:
        crop_str = (
            f"x={crop[0]} y={crop[1]} w={crop[2]} h={crop[3]}  → {crop[2]}x{crop[3]}"
        )
        after_crop: tuple[int, int] = (crop[2], crop[3])
    else:
        crop_str = "none"
        after_crop = src_img_size

    if resize:
        resize_str = f"{resize[0]}x{resize[1]}"
        output_res: tuple[int, int] = resize
    else:
        resize_str = model_img_pipeline if model_img_pipeline else "none"
        output_res = model_img_size or after_crop

    rows: list[tuple[str, str]] = []
    if model_name:
        rows.append(("Model name", model_name))
    rows += [
        ("Device", device),
        ("Model file", model_path.name),
        ("Output", str(output_dir)),
        ("Scenes", f"{start_scene} → {'end' if end_scene is None else end_scene - 1}"),
        ("Score threshold", str(score_thr)),
        (
            "Point Cloud Range",
            f"x [{xmin}, {xmax}]  y [{ymin}, {ymax}]  z [{zmin}, {zmax}]",
        ),
        ("Img norm", norm_str),
        (
            "Input resolution",
            f"{src_img_size[0]}x{src_img_size[1]}  (nuScenes cameras)",
        ),
        ("Crop", crop_str),
        ("Resize", resize_str),
        ("Output resolution", f"{output_res[0]}x{output_res[1]}"),
    ]
    if model_img_pipeline:
        rows.append(("  pipeline", model_img_pipeline))
    if extra:
        rows += [(k.strip(), str(v)) for k, v in extra.items()]

    table = Table(
        title=f"BEVFormer {backend} inference",
        show_header=False,
        padding=(0, 2),
    )
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    for label, value in rows:
        table.add_row(label, value)

    console.print()
    console.print(table)
    console.print()


def print_modality_overlays_table(
    rows: list[tuple[str, str, str, str]],
    *,
    title: str = "Modalities (model vs video) overlays",
) -> None:
    """Print a 4-column Rich table: Modalities | Model Config | Camera Overlay | BEV Overlay.

    Each row is ``(modality_label, cfg_cell, camera_cell, bev_cell)``. Cell strings should
    already include the reason (e.g. ``enabled (--lidar-cameras)``, ``n/a (out of scope)``).
    """
    table = Table(title=title, show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Modalities", style="bold", no_wrap=True, overflow="fold")
    table.add_column("Model Config", overflow="fold")
    table.add_column("Camera Overlay", overflow="fold")
    table.add_column("BEV Overlay", overflow="fold")
    for mod_name, cfg_s, cam_s, bev_s in rows:
        table.add_row(mod_name, cfg_s, cam_s, bev_s)
    console.print()
    console.print(table)
    console.print()
