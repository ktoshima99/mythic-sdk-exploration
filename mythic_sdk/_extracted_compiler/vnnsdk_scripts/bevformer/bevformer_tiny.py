import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from numpy.typing import NDArray
from onnx.inliner import inline_selected_functions
from onnxscript import values

from vnnort import configure_logging
from vnnort.data.container import (
    InputData,
    MultiViewDetection3DInput,
    MultiViewDetection3DOutput,
)
from vnnort.models.initialization_config import InitializationConfig
from . import bevformer_tiny_config
from .modeling.bevformer import (
    build_bevformer_tiny,
    build_bevformer_tiny_transformer,
)
from .utils import RandomNuscenesBevformerDataset, _post_process
from vnnort.models.vid_model import ModelState, VidModel
from vnnort.optimizer.optimization_config import OptimizationConfig
from vnnort.optimizer.symbolic_shape_inference import symbolic_shape_inference
from vnnort.optimizer.utils import move_constants_to_wgts
from vnnort.quantizer.quantization_config import QuantizationConfig
from vnnort.utils.onnx_utils.optimize_execution_order import optimize_execution_order
from vnnort.utils.onnx_utils.utils import set_input_shape

logger = logging.getLogger(__name__)


class BevformerTiny(VidModel):
    """BEVFormer-tiny multi-camera 3D detection model."""

    config = bevformer_tiny_config
    TRANSFORMER_PART_ONLY: bool = True
    IMAGE_HEIGHT: int = 900
    IMAGE_WIDTH: int = 1600
    N_CAM: int = 6
    N_BEV_QUEUE: int = 2

    # ── Variant constants — BEVFormer-tiny ───────────────────────────────────
    # Pipeline choices (live inside config.test_pipeline as dicts, not exposed at top level)
    IMAGE_SCALE: float = 0.5  # 1600x900 → 800x450 → pad to 800x480
    PAD_DIVISOR: int = 32
    # Derived from `config`. ImageNet RGB stats — PIL.Image.convert("RGB") already
    # gives RGB, so the config's `to_rgb=True` is a no-op for us.
    NORM_MEAN: tuple[float, ...] = tuple(config.img_norm_cfg["mean"])  # type: ignore[arg-type]
    NORM_STD: tuple[float, ...] = tuple(config.img_norm_cfg["std"])  # type: ignore[arg-type]
    PC_RANGE: tuple[float, ...] = tuple(config.point_cloud_range)
    BEV_H: int = config.bev_h_
    BEV_W: int = config.bev_w_
    EMBED_DIMS: int = config._dim_
    POST_CENTER_RANGE: tuple[float, ...] = tuple(config.model["pts_bbox_head"]["bbox_coder"]["post_center_range"])  # type: ignore[index]
    MAX_NUM: int = config.model["pts_bbox_head"]["bbox_coder"]["max_num"]  # type: ignore[index]
    NUM_CLASSES: int = config.model["pts_bbox_head"]["num_classes"]  # type: ignore[index]
    # Not in config — variant-specific eval choice.
    SCORE_THR: float = 0.3

    # (image_height, image_width) -> (h_feat, w_feat) for the encoder feature map.
    _FEATURE_HW: dict[tuple[int, int], tuple[int, int]] = {
        (900, 1600): (29, 50),
        (450, 800): (25, 16),
    }

    @classmethod
    def _feature_map_hw(cls) -> tuple[int, int]:
        """Return encoder feature-map (H, W) for the configured image resolution."""
        try:
            return cls._FEATURE_HW[(cls.IMAGE_HEIGHT, cls.IMAGE_WIDTH)]
        except KeyError:
            raise ValueError(f"Unsupported input resolution {cls.IMAGE_HEIGHT}x{cls.IMAGE_WIDTH}") from None

    @classmethod
    def _input_shapes(cls) -> dict[str, list[int]]:
        """Per-input ONNX shape spec, derived from `cls.config`.

        Single source of truth shared by `initialize_onnx` (graph IO) and
        `_random_inputs` (tracing tensors).
        """
        cfg = cls.config
        hidden_size = cfg._dim_
        n_bev = cfg.bev_h_ * cfg.bev_w_
        num_query = cfg.model["pts_bbox_head"]["num_query"]

        encoder_attn_cfgs = cfg.model["pts_bbox_head"]["transformer"]["encoder"]["transformerlayers"]["attn_cfgs"]
        decoder_attn_cfgs = cfg.model["pts_bbox_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"]
        # num_heads lives on the decoder's first attention (MultiheadAttention); the
        # encoder's first attn (TemporalSelfAttention) has no num_heads key.
        h_dec = decoder_attn_cfgs[0]["num_heads"]
        l_enc = encoder_attn_cfgs[1]["deformable_attention"]["num_levels"]
        p_enc = encoder_attn_cfgs[1]["deformable_attention"]["num_points"]
        ref_pts_ch = h_dec * l_enc * p_enc * 2

        shapes: dict[str, list[int]] = {
            "bev_query": [1, hidden_size, 1, n_bev],
            "bev_value_stacked": [1, hidden_size, cls.N_BEV_QUEUE, n_bev],
            "bev_mask": [1, 1, cls.N_CAM, n_bev],
            "ref_pts_cam": [1, ref_pts_ch, cls.N_CAM, n_bev],
            "count_normalizer": [1, 1, 1, n_bev],
            "obj_queries": [1, hidden_size, 1, num_query],
            "ref_points": [1, 3, 1, num_query],
        }
        if cls.TRANSFORMER_PART_ONLY:
            h_feat, w_feat = cls._feature_map_hw()
            shapes["img_features"] = [1, hidden_size, h_feat * w_feat, cls.N_CAM]
        else:
            shapes["images"] = [1, 3, cls.IMAGE_HEIGHT, cls.N_CAM * cls.IMAGE_WIDTH]
        return shapes

    @classmethod
    def _random_inputs(cls) -> dict[str, NDArray[np.float32]]:
        """Random float32 tensors matching `_input_shapes()`, used to trace the ONNX."""
        return {name: np.random.random(shape).astype(np.float32) for name, shape in cls._input_shapes().items()}

    @classmethod
    def initialize_onnx(
        cls, model_directory: str | Path, initialization_config: InitializationConfig | None = None
    ) -> onnx.ModelProto:
        """Return a runable ONNX ModelProto object.

        Args:
            model_directory (str|Path): Path to model directory
            initialization_config (InitializationConfig | None): The initialization config

        Returns:
            onnx.ModelProto: The rununable ONNX ModelProto object

        """
        logger.info("Build Onnxscript model")

        if cls.TRANSFORMER_PART_ONLY:
            onnxscript_model = build_bevformer_tiny_transformer(cls.config)  # type: ignore[no-untyped-call]
        else:
            onnxscript_model = build_bevformer_tiny(cls.config)  # type: ignore[no-untyped-call]

        onnxscript_model(**cls._random_inputs())  # type: ignore
        onnx_model = onnxscript_model.to_model_proto()  # type: ignore

        model_directory = Path(model_directory)

        for name, shape in cls._input_shapes().items():
            set_input_shape(onnx_model, name, shape)

        logger.info("Inlining ModelProto")
        # Inline model specific onnx functions
        inline_function_names = [
            "Resnet50",
            "Resnet101",
            "ResnetBlock",
            "FPN",
            "TemporalSelfAttention",
            "SpatialCrossAttention",
            "MultiHeadSelfAttention",
            "DecoderDeformableAttention",
            "BevformerTinyEncoder",
            "BevformerTinyDecoder",
            "BevformerTinyTransformer",
            "RegressionBranch",
            "ClassificationBranch",
            "Postprocessing",
        ]
        inline_function_args = []
        cache = values.Opset.cache
        for opset in cache.values():
            if "com.videantis.dynamic_functions" in opset.domain:
                for func in opset.function_defs.values():
                    onnx_model.functions.append(func.to_function_proto())
        for opset in cache.values():
            for func in opset.function_defs.values():
                if func.name in inline_function_names:
                    inline_function_args.append((opset.domain, func.name))
        logger.info("Inlining functions")
        onnx_model = inline_selected_functions(onnx_model, inline_function_args)

        logger.info("Movings constants to weights")
        onnx_model = move_constants_to_wgts(onnx_model)

        logger.info("Performing symbolic shape inference")
        onnx_model = symbolic_shape_inference(onnx_model)

        logger.info("Optimizing graph execution order")
        onnx_model = optimize_execution_order(onnx_model)

        return onnx_model

    def optimize(self, optimization_config: OptimizationConfig = None) -> VidModel:
        """Override optimize method Usually optimization pipeline is triggered, but this model is already optimized."""
        self._state = ModelState.OPTIMIZED
        self._update_onnx_meta()

        return self

    def setup(self) -> None:
        """Initialize per-instance numpy constants and reset the temporal BEV state."""
        self._norm_mean = np.asarray(self.NORM_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self._norm_std = np.asarray(self.NORM_STD, dtype=np.float32).reshape(1, 1, 3)
        self._prev_bev: np.ndarray[Any, Any] | None = None

    def preprocess(self, input_data: MultiViewDetection3DInput) -> dict[str, NDArray[np.float32]]:
        """Resize/normalize/pad 6 camera images, scale lidar2img rows, prepare temporal state."""
        if not isinstance(input_data, MultiViewDetection3DInput):
            raise TypeError(f"Expected MultiViewDetection3DInput, got {type(input_data).__name__}")

        # Reset the temporal state at every scene boundary. Because preprocess is on
        # the input-side of the pipeline, this is the only place we can decide that
        # the next inference call must run with `use_prev_bev=0`.
        if input_data.is_first_in_scene:
            self._prev_bev = None

        imgs: list[NDArray[np.float32]] = []
        for pil_img in input_data.images:
            arr = np.asarray(pil_img, dtype=np.uint8)  # (H, W, 3), RGB (dataset returns convert("RGB"))
            new_w = int(round(arr.shape[1] * self.IMAGE_SCALE))
            new_h = int(round(arr.shape[0] * self.IMAGE_SCALE))
            arr_resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            arr_f = arr_resized.astype(np.float32)
            arr_f = (arr_f - self._norm_mean) / self._norm_std

            pad_h = (-new_h) % self.PAD_DIVISOR
            pad_w = (-new_w) % self.PAD_DIVISOR
            if pad_h or pad_w:
                arr_f = np.pad(arr_f, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            imgs.append(arr_f.transpose(2, 0, 1))  # HWC → CHW

        img_np = np.stack(imgs, axis=0)[None, ...].astype(np.float32)  # noqa: F841  # (1, 6, 3, H', W')

        # Scale the projection's first two rows (u, v) by the image scale; padding
        # adds zeros at the right/bottom so the principal point is unchanged.
        lidar2img = input_data.lidar2img.copy()
        lidar2img[:, :2, :] *= self.IMAGE_SCALE
        lidar2img = lidar2img[None, ...].astype(np.float32)  # noqa: F841

        can_bus = input_data.can_bus.astype(np.float32).reshape(1, 18)  # noqa: F841

        if self._prev_bev is None:
            prev_bev = np.zeros((1, self.BEV_H * self.BEV_W, self.EMBED_DIMS), dtype=np.float32)  # noqa: F841
        else:
            prev_bev = self._prev_bev  # noqa: F841

        # FIXME this will need to be replaced once the real preprocessing is implemented.
        return self._random_inputs()

        # return {
        #     "img": img_np,
        #     "lidar2img": lidar2img,
        #     "can_bus": can_bus,
        #     "prev_bev": prev_bev,
        # }

    def postprocess(self, model_output: Any, input_data: InputData) -> MultiViewDetection3DOutput:
        """Decode the ONNX outputs to a `MultiViewDetection3DOutput` and stash next prev_bev."""
        if not isinstance(input_data, MultiViewDetection3DInput):
            raise TypeError(f"Expected MultiViewDetection3DInput, got {type(input_data).__name__}")

        bev_embed = np.asarray(model_output["bev_embed"])  # (B, bev_h*bev_w, embed_dims) batch-first
        cls_out = np.asarray(model_output["outputs_classes"])  # (B, num_dec, Q, C)
        bbox_out = np.asarray(model_output["outputs_coords"])  # (B, num_dec, Q, 10)

        # Stash for the next frame in the scene. The ONNX expects (B, bev_h*bev_w, embed_dims)
        # batch-first, matching the bev_embed output layout — no transpose needed.
        self._prev_bev = bev_embed.astype(np.float32, copy=False)

        # Transpose cls/bbox to decoder-first (num_dec, B, Q, C) for _post_process.
        all_cls = np.transpose(cls_out, (1, 0, 2, 3))
        all_bbox = np.transpose(bbox_out, (1, 0, 2, 3))

        boxes, scores, labels = _post_process(
            all_cls,
            all_bbox,
            post_center_range=self.POST_CENTER_RANGE,
            max_num=self.MAX_NUM,
            num_classes=self.NUM_CLASSES,
            score_thr=self.SCORE_THR,
        )

        return MultiViewDetection3DOutput(
            boxes=boxes,
            scores=scores,
            labels=labels.astype(np.int32),
        )

    @classmethod
    def load_default_dataset(cls) -> RandomNuscenesBevformerDataset:
        """Ideally this would be replaced with a `NuscenesBevformerDataset` on the default val split.

        But for now return the random dataset so that no files are required to run the model end-to-end.
        """
        return RandomNuscenesBevformerDataset(split="val")


if __name__ == "__main__":
    configure_logging()
    model_directory = "model_flow_results/Bevformer"
    model = BevformerTiny(model_directory)
    model.optimize()
    report, network = model.quantize(QuantizationConfig(calibration_dataset_size=1))
