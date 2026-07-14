import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx.numpy_helper import from_array
from onnxscript.rewriter.pattern import RewriteRuleClassBase

from vnnort import configure_logging
from vnnort.data.container import ImageDetectionInput, ImageDetectionOutput
from vnnort.models.initialization_config import InitializationConfig
from vnnort.models.vid_model import VidModel
from vnnort.optimizer.pattern_detection import vid_match_patterns_onnxscript
from vnnort.optimizer.utils import (
    get_wgt_by_name,
    infer_shapes_runtime,
    move_constants_to_wgts,
    move_static_cons_to_wgts,
    remove_unused_nodes,
    replace_wgt,
)
from vnnort.utils.onnx_utils.graph_helper import ONNXGraphHelper, Tensor, TensorType

# Python sets __package__=None when run as __main__, making relative imports fail.
# This ensures mythic_utils is on sys.path so it can be imported by name.
sys.path.insert(0, str(Path(__file__).parent))
from mythic_utils import DummyDataset  # noqa: E402

# Path to the source ONNX file, set by the CLI before instantiating the VidModel.
# This is very hacky but we do not have a better way right now to pass the CLI argument to the VidModel.
ONNX_PATH: str | Path = ""


class BoundingBoxRewritePattern(RewriteRuleClassBase):
    """Rewrite the detection postprocessing graph to a format we can handle.

    The incoming graph has three feature-level inputs (144ch = 64 DFL + 80 classes each).
    Each input is scaled per-channel (Mul), flattened (Reshape), and concatenated before
    the DFL softmax decode and class sigmoid paths.

    The rewrite:
    - Splits the per-channel scale factors into DFL (64ch) and class (80ch) parts.
    - Moves the class Sigmoid to immediately after the per-level Mul so it can be fused.
    - Replaces Transpose+Softmax+Transpose with vidSoftmax and the DFL conv with four
      separate padded vidConv calls (one per bbox coordinate group of 16 bins).
    - Replaces Sub/Add/Div in the bbox decode with Shortcut operations.
    """

    level = 2

    @classmethod
    def pattern(
        cls,
        op: Any,
        x1: Any,
        x2: Any,
        x3: Any,
        m1: Any,
        m2: Any,
        m3: Any,
        w: Any,
        a1: Any,
        a2: Any,
        a3: Any,
    ) -> Any:
        """Pattern to be matched."""
        x1 = op.Mul(x1, m1)
        x1 = op.Reshape(x1, _allow_other_inputs=True)
        x2 = op.Mul(x2, m2)
        x2 = op.Reshape(x2, _allow_other_inputs=True)
        x3 = op.Mul(x3, m3)
        x3 = op.Reshape(x3, _allow_other_inputs=True)
        x = op.Concat(x1, x2, x3, axis=2)

        # Class scores path (last 80 channels)
        sig = op.Slice(x, _allow_other_inputs=True)
        sig = op.Sigmoid(sig)

        # DFL decode path (first 64 channels)
        x = op.Slice(x, _allow_other_inputs=True)
        x = op.Reshape(x, _allow_other_inputs=True)
        x = op.Transpose(x)
        x = op.Softmax(x)
        x = op.Transpose(x)
        x = op.vidConv(x, w, None, _domain="com.videantis")

        x = op.Reshape(x, _allow_other_inputs=True)
        x1 = op.Slice(x, _allow_other_inputs=True, _allow_other_attributes=True)
        x2 = op.Slice(x, _allow_other_inputs=True)
        y1 = op.Sub(a1, x1)
        y2 = op.Add(a2, x2)
        z1 = op.Add(y1, y2)
        o1 = op.Div(z1, _allow_other_inputs=True)
        o2 = op.Sub(y2, y1)
        x = op.Concat(o1, o2, axis=1)
        x = op.Mul(x, a3)

        return x, sig

    @classmethod
    def rewrite(
        cls,
        op: Any,
        x1: Any,
        x2: Any,
        x3: Any,
        m1: Any,
        m2: Any,
        m3: Any,
        w: Any,
        a1: Any,
        a2: Any,
        a3: Any,
    ) -> Any:
        """Rewrite graph."""
        # Split scale factors [1, 144, 1, 1] into DFL (first 64ch) and class (last 80ch).
        # check() guarantees const_value is populated before we get here.
        m1_np = m1.const_value.numpy()
        m1_dfl = op.Constant(value=from_array(m1_np[:, :64]))
        m1_cls = op.Constant(value=from_array(m1_np[:, 64:]))
        m2_np = m2.const_value.numpy()
        m2_dfl = op.Constant(value=from_array(m2_np[:, :64]))
        m2_cls = op.Constant(value=from_array(m2_np[:, 64:]))
        m3_np = m3.const_value.numpy()
        m3_dfl = op.Constant(value=from_array(m3_np[:, :64]))
        m3_cls = op.Constant(value=from_array(m3_np[:, 64:]))

        # Slice each input into DFL [0:64] and class [64:144] along channel dim
        x1_dfl = op.Slice(x1, op.Constant(value_ints=[0]), op.Constant(value_ints=[64]), op.Constant(value_ints=[1]))
        x1_cls = op.Slice(x1, op.Constant(value_ints=[64]), op.Constant(value_ints=[144]), op.Constant(value_ints=[1]))
        x2_dfl = op.Slice(x2, op.Constant(value_ints=[0]), op.Constant(value_ints=[64]), op.Constant(value_ints=[1]))
        x2_cls = op.Slice(x2, op.Constant(value_ints=[64]), op.Constant(value_ints=[144]), op.Constant(value_ints=[1]))
        x3_dfl = op.Slice(x3, op.Constant(value_ints=[0]), op.Constant(value_ints=[64]), op.Constant(value_ints=[1]))
        x3_cls = op.Slice(x3, op.Constant(value_ints=[64]), op.Constant(value_ints=[144]), op.Constant(value_ints=[1]))

        # Class path: scale then Sigmoid immediately so it can be merged into the Shortcut
        x1_cls = op.Shortcut(x1_cls, m1_cls, mode="multiplication", _domain="com.videantis", _version=1)
        x1_cls = op.Sigmoid(x1_cls)
        x2_cls = op.Shortcut(x2_cls, m2_cls, mode="multiplication", _domain="com.videantis", _version=1)
        x2_cls = op.Sigmoid(x2_cls)
        x3_cls = op.Shortcut(x3_cls, m3_cls, mode="multiplication", _domain="com.videantis", _version=1)
        x3_cls = op.Sigmoid(x3_cls)
        x1_cls = op.Reshape(x1_cls, op.Constant(value_ints=[1, 80, 1, -1]))
        x2_cls = op.Reshape(x2_cls, op.Constant(value_ints=[1, 80, 1, -1]))
        x3_cls = op.Reshape(x3_cls, op.Constant(value_ints=[1, 80, 1, -1]))
        sig_out = op.Concat(x1_cls, x2_cls, x3_cls, axis=-1)  # [1, 80, 1, 42840]

        # DFL path: scale, flatten, concat, vidSoftmax
        x1_dfl = op.Shortcut(x1_dfl, m1_dfl, mode="multiplication", _domain="com.videantis", _version=1)
        x1_dfl = op.Reshape(x1_dfl, op.Constant(value_ints=[1, 64, 1, -1]))
        x2_dfl = op.Shortcut(x2_dfl, m2_dfl, mode="multiplication", _domain="com.videantis", _version=1)
        x2_dfl = op.Reshape(x2_dfl, op.Constant(value_ints=[1, 64, 1, -1]))
        x3_dfl = op.Shortcut(x3_dfl, m3_dfl, mode="multiplication", _domain="com.videantis", _version=1)
        x3_dfl = op.Reshape(x3_dfl, op.Constant(value_ints=[1, 64, 1, -1]))
        x = op.Concat(x1_dfl, x2_dfl, x3_dfl, axis=-1)  # [1, 64, 1, 42840]

        x = op.vidSoftmax(x, group=[4], _domain="com.videantis")

        # Slice into 4 groups of 16 (one per bbox coordinate)
        x1 = op.Slice(x, op.Constant(value_ints=[0]), op.Constant(value_ints=[16]), op.Constant(value_ints=[1]))
        x2 = op.Slice(x, op.Constant(value_ints=[16]), op.Constant(value_ints=[32]), op.Constant(value_ints=[1]))
        x3 = op.Slice(x, op.Constant(value_ints=[32]), op.Constant(value_ints=[48]), op.Constant(value_ints=[1]))
        x4 = op.Slice(x, op.Constant(value_ints=[48]), op.Constant(value_ints=[64]), op.Constant(value_ints=[1]))

        # Pad w from [1, 16, 1, 1] to [8, 16, 1, 1] to meet vidConv channel requirements
        w_np_arr = w.const_value.numpy()
        padded_w = np.pad(w_np_arr, ((0, 7), (0, 0), (0, 0), (0, 0)), mode="constant")
        w = op.Constant(value=from_array(padded_w))
        x1 = op.vidConv(x1, w, None, _domain="com.videantis", _version=1)
        x2 = op.vidConv(x2, w, None, _domain="com.videantis", _version=1)
        x3 = op.vidConv(x3, w, None, _domain="com.videantis", _version=1)
        x4 = op.vidConv(x4, w, None, _domain="com.videantis", _version=1)

        # Split anchor constants to avoid merging channels and spatial dims in the final reshape
        a11 = op.Constant(value=from_array(a1.const_value.numpy()[:, 0]))
        a12 = op.Constant(value=from_array(a1.const_value.numpy()[:, 1]))
        a21 = op.Constant(value=from_array(a2.const_value.numpy()[:, 0]))
        a22 = op.Constant(value=from_array(a2.const_value.numpy()[:, 1]))

        # Simulate Sub(a1, x) as Shortcut(a1, -x)
        x1 = op.Shortcut(x1, op.Constant(value_float=-1.0), mode="multiplication", _domain="com.videantis", _version=1)
        x2 = op.Shortcut(x2, op.Constant(value_float=-1.0), mode="multiplication", _domain="com.videantis", _version=1)
        x1 = op.Shortcut(a11, x1, _domain="com.videantis", _version=1)
        x2 = op.Shortcut(a12, x2, _domain="com.videantis", _version=1)

        x3 = op.Shortcut(a21, x3, _domain="com.videantis", _version=1)
        x4 = op.Shortcut(a22, x4, _domain="com.videantis", _version=1)

        x1 = op.Concat(x1, x2, axis=1)  # top-left part
        x2 = op.Concat(x3, x4, axis=1)  # bottom-right part

        y1 = op.Shortcut(x1, x2, _domain="com.videantis", _version=1)  # Add → center*2
        x1 = op.Shortcut(x1, op.Constant(value_float=-1.0), mode="multiplication", _domain="com.videantis", _version=1)
        y2 = op.Shortcut(x2, x1, _domain="com.videantis", _version=1)  # Sub → size

        # Simulate Div by 2 with multiply by 0.5
        y1 = op.Shortcut(y1, op.Constant(value_float=0.5), mode="multiplication", _domain="com.videantis", _version=1)

        bbox_path = op.Concat(y1, y2, axis=1)
        bbox_path = op.Shortcut(bbox_path, a3, mode="multiplication", _domain="com.videantis", _version=1)

        return bbox_path, sig_out

    @classmethod
    def check(  # type: ignore[override]
        cls, op: Any, x1: Any, x2: Any, x3: Any, m1: Any, m2: Any, m3: Any, w: Any, a1: Any, a2: Any, a3: Any
    ) -> bool:  # noqa: ARG002
        """Only rewrite when all Mul weights are concrete constants (guards against commuted matches)."""
        return m1.const_value is not None and m2.const_value is not None and m3.const_value is not None


def _expand_to_4D_shape(arr: NDArray[Any]) -> NDArray[Any]:
    """Expand a NumPy array to 4 dimensions by left-padding with singleton dimensions."""
    arr = np.array(arr, dtype=np.float32)  # also handles scalars
    current_shape = arr.shape

    if len(current_shape) > 4:
        raise ValueError("Input array has more than 4 dimensions.")

    new_shape = (1,) * (4 - len(current_shape)) + current_shape
    return arr.reshape(new_shape)


def _ensure_tensor_is_4D(model: onnx.ModelProto, input_tensor: Tensor) -> None:
    if input_tensor.tensor_type == TensorType.NODE_OUTPUT or input_tensor.tensor_type == TensorType.GRAPH_INPUT:
        return  # Dynamic tensor
    elif input_tensor.tensor_type == TensorType.INITIALIZER:
        initializer_data = get_wgt_by_name(model, input_tensor.name)
        new_data = _expand_to_4D_shape(initializer_data)
        model = replace_wgt(model, new_data, input_tensor.name)

    else:
        raise RuntimeError("Unknown tensor type.")


def generate_random_inputs(input_value_protos: Any, batch_size: int = 1) -> dict[str, Any]:
    """Generate a dict of random input tensors for an ONNX model.

    Dynamic or symbolic batch dimension is replaced with batch_size.
    """
    input_dict: dict[str, Any] = {}

    for inp in input_value_protos:
        name = inp.name
        tensor_type = inp.type.tensor_type

        # Resolve shape
        shape = []
        for i, dim in enumerate(tensor_type.shape.dim):
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                # treat unknown/symbolic dimension as batch dimension
                if i == 0:
                    shape.append(batch_size)
                else:
                    raise ValueError(f"Cannot resolve dimension {i} for input '{name}'.")

        # Generate random data
        data = np.random.randn(*shape).astype(np.float32) * 20.0
        input_dict[name] = data

    return input_dict


class MythicYoloV8Postprocessing(VidModel):
    """VidModel definition for the Mythic YoloV8 object detection postprocessing part."""

    @classmethod
    def initialize_onnx(
        cls, model_directory: str | Path, config: Optional[InitializationConfig] = None
    ) -> onnx.ModelProto:
        """Return a runable ONNX ModelProto of the  model."""
        return onnx.load(ONNX_PATH)

    def setup(self) -> None:
        """Extract inputs to be used for random input generation."""
        self.input_value_protos = [inp for inp in self._model_repr.graph.input]

    def preprocess(self, input_data: ImageDetectionInput) -> Any:
        """Preprocess an image by resizing and normalizing."""
        example_data = generate_random_inputs(self.input_value_protos, batch_size=1)
        return example_data

    def postprocess(self, model_output: Any, _: ImageDetectionInput) -> ImageDetectionOutput:  # type: ignore
        """Remove channel padding and restore original channel order.

        BoundingBoxRewritePattern pads 4 bbox channels to 32 (valid at [0, 8, 16, 24]).
        The 80 class-score channels are unpadded and sit contiguously at [32:112].
        A singleton spatial dimension is also inserted during optimization.
        """
        assert len(model_output) == 1, f"Expected 1 output tensor, got {len(model_output)}."
        tensor = next(iter(model_output.values()))  # [1, 112, 1, 42840]

        bbox = tensor[:, [0, 8, 16, 24], :, :]  # [1,  4, 1, 42840]
        classes = tensor[:, 32:112, :, :]  # [1, 80, 1, 42840]

        result = np.concatenate([bbox, classes], axis=1)  # [1, 84, 1, 42840]
        return {next(iter(model_output.keys())): result.squeeze(2)}  # type: ignore  # [1, 84, 42840]

    @classmethod
    def load_default_dataset(cls) -> DummyDataset:
        """Return an initialized dataset that can be used to load data samples for this model."""
        return DummyDataset()

    def optimize_hook(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Rewrite most of the graph so it can be processed by us."""
        rule1 = BoundingBoxRewritePattern.rule()  # type: ignore[no-untyped-call]
        model, count1 = vid_match_patterns_onnxscript(model, rule1, verbose=0, commute=True)

        if count1 != 1:
            raise ValueError(f"Expected to match BoundingBoxRewritePattern once, but matched {count1} times.")

        # Most of our toolbox can't handle constants properly so move them to initializers
        model = move_constants_to_wgts(model)

        # Ensure static inputs of Shortcut nodes have correct input rank so that no automatic broadcasting
        # to higher rank is required
        graph_helper = ONNXGraphHelper(model)
        for node in graph_helper.nodes.values():
            if node.op_type == "Shortcut":
                input_tensor1 = node.inputs[0]
                input_tensor2 = node.inputs[1]
                _ensure_tensor_is_4D(model, input_tensor1)
                _ensure_tensor_is_4D(model, input_tensor2)

        # Clean up graph one last time and infer shapes
        ds = self.load_default_dataset()
        data1, data2 = self.preprocess(ds[0]), self.preprocess(ds[50])
        model = move_static_cons_to_wgts(model, [data1, data2])
        model = remove_unused_nodes(model)  # type: ignore[arg-type]
        infer_shapes_runtime(model, inputs=data1)

        # One last sanity check
        onnx.checker.check_model(model, full_check=True, check_custom_domain=True)

        return model


if __name__ == "__main__":
    from mythic_utils import parse_arguments, run_vnn_flow

    args = parse_arguments()

    # Prepare/Check input/output paths
    result_directory: Path = args.result_directory
    result_directory.mkdir(parents=True, exist_ok=True)

    source_onnx: Path = args.source_onnx
    if not source_onnx.is_file():
        print(f"Error: '{source_onnx}' is not a valid file.")
        sys.exit(1)

    # This is very ugly but we do not have a better way right now to access the CLI path within the VidModel
    ONNX_PATH = source_onnx

    configure_logging()

    # Run v-NN ORT pipeline with this model
    model = MythicYoloV8Postprocessing(result_directory)
    run_vnn_flow(model, result_directory, system_config=Path(__file__).parent / "system_configs" / "yolov8.cfg")
