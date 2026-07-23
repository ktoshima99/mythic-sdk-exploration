"""BEVFormer shared utilities.
Organised into sub-modules:
  data_loading  — temporal state, batch extraction, mmdet3d dataloader builders
  processing    — constants, InferenceConfig, config parsing, image pre/post-processing
  visualization — camera-view and BEV-map rendering
  result_writer — per-scene frame/video/JSON output
  inference     — ONNX session helpers, scene loop, run summary
``build_torchnet_from_onnx`` (in ``inference``) lazy-imports munc only when called.
"""
from .data_loading import (
    BEVFormerTorchNetDataLoader,
    TemporalState,
    accumulate_radar_points_lidar_frame,
    advance_sample_data_to_timestamp,
    build_dataloader_from_mmcv_config,
    build_sweeps_dataloader,
    closest_can_bus_pose,
    earliest_sample_data_in_scene,
    extract_img_scale,
    extract_sample_arrays,
    extract_sample_token,
    extract_scene_token,
    load_mmcv_config,
    load_nuscenes_lidar_xyz_multisweep,
    precompute_scene_info,
    resolve_lidar_top_sample_data_token,
    unwrap_dc,
    unwrap_meta,
)
from .ground_truth import (
    inject_ann_info_for_gt,
    patch_pipeline_for_gt,
)
from .inference import (
    SceneFilter,
    build_pth_model,
    build_torchnet_from_onnx,
    build_torchnet_inputs,
    default_out_dir,
    get_prev_bev,
    load_onnx_session,
    load_onnx_session_or_suggest_torchnet,
    onnx_prev_bev,
    onnx_run_frame,
    pth_run_frame,
    print_modality_overlays_table,
    print_run_summary,
    run_inference_loop,
    torchnet_run_frame,
)
from .nuscenes_gt import nusc_boxes_to_visualize_result
from .processing import (
    CLASS_COLORS,
    InferenceConfig,
    apply_crop_resize_to_batch,
    denormalize_image,
    parse_config_py,
    parse_crop,
    parse_resize,
    post_process,
    preprocess_image,
)
from .result_writer import ResultWriter
from .visualization import (
    build_map_underlay_for_bev,
    visualize_frame,
)
__all__ = [
    # data_loading
    "BEVFormerTorchNetDataLoader",
    "TemporalState",
    "accumulate_radar_points_lidar_frame",
    "build_dataloader_from_mmcv_config",
    "build_sweeps_dataloader",
    "closest_can_bus_pose",
    "advance_sample_data_to_timestamp",
    "earliest_sample_data_in_scene",
    "extract_img_scale",
    "extract_sample_arrays",
    "extract_sample_token",
    "extract_scene_token",
    "load_mmcv_config",
    "load_nuscenes_lidar_xyz_multisweep",
    "precompute_scene_info",
    "resolve_lidar_top_sample_data_token",
    "unwrap_dc",
    "unwrap_meta",
    # ground_truth
    "inject_ann_info_for_gt",
    "patch_pipeline_for_gt",
    # inference
    "SceneFilter",
    "build_pth_model",
    "build_torchnet_from_onnx",
    "default_out_dir",
    "get_prev_bev",
    "load_onnx_session",
    "load_onnx_session_or_suggest_torchnet",
    "onnx_prev_bev",
    "onnx_run_frame",
    "pth_run_frame",
    "build_torchnet_inputs",
    "torchnet_run_frame",
    "nusc_boxes_to_visualize_result",
    "print_modality_overlays_table",
    "print_run_summary",
    "run_inference_loop",
    # processing
    "CLASS_COLORS",
    "InferenceConfig",
    "apply_crop_resize_to_batch",
    "denormalize_image",
    "parse_config_py",
    "parse_crop",
    "parse_resize",
    "post_process",
    "preprocess_image",
    # result_writer
    "ResultWriter",
    # visualization
    "build_map_underlay_for_bev",
    "visualize_frame",
]
