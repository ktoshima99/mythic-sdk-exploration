import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
from onnx.numpy_helper import from_array
from onnxscript.rewriter.pattern import RewriteRuleClassBase

from vnnort import configure_logging
from vnnort.data.container import ImageDetectionInput, ImageDetectionOutput
from vnnort.models.initialization_config import InitializationConfig
from vnnort.models.vid_model import VidModel
from vnnort.optimizer.pattern_detection import vid_match_patterns_onnxscript
from vnnort.optimizer.utils import infer_shapes_runtime, move_static_cons_to_wgts, remove_unused_nodes

# Python sets __package__=None when run as __main__, making relative imports fail.
# This ensures mythic_utils is on sys.path so it can be imported by name.
sys.path.insert(0, str(Path(__file__).parent))
from mythic_utils import DummyDataset  # noqa: E402

# Path to the source ONNX file, set by the CLI before instantiating the VidModel.
# This is very hacky but we do not have a better way right now to pass the CLI argument to the VidModel.
ONNX_PATH: str | Path = ""


def generate_random_inputs(input_value_protos: Any, batch_size: int = 1) -> dict[str, Any]:
    """Generate a dict of random input tensors for an ONNX model.

    Dynamic or symbolic batch dimension is replaced with batch_size.
    """
    input_dict = {}

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


class ConvPadRewritePattern(RewriteRuleClassBase):
    """Pad the output channels of all convolutions to multiple of 8."""

    level = 2

    @classmethod
    def pattern(cls, op: Any, x: Any, w: Any, b: Any) -> Any:
        """Pattern to be matched."""
        x = op.vidConv(x, w, b, _domain="com.videantis")
        return x

    @classmethod
    def rewrite(cls, op: Any, x: Any, w: Any, b: Any) -> Any:
        """Rewrite the matched pattern by padding the output channels to multiple of 8."""
        # Pad w to multiple of 8 in the output channels dimension
        w_arr = w.const_value.numpy()
        out_channels = w_arr.shape[0]
        pad_channels = (8 - (out_channels % 8)) % 8
        padded_w = np.pad(w_arr, ((0, pad_channels), (0, 0), (0, 0), (0, 0)), mode="constant")
        w = from_array(padded_w, name=w.name)

        padded_b = np.pad(b.const_value.numpy(), (0, pad_channels), mode="constant")
        b = from_array(padded_b, name=b.name)

        w = op.Constant(value=w)
        b = op.Constant(value=b)

        x = op.vidConv(x, w, b, _domain="com.videantis", _version=1)
        return x

    @classmethod  # type: ignore[override]
    def check(cls, op: Any, x: Any, w: Any, b: Any) -> Any:
        """Make sure this only pattern is only applied once by checking that weights are not already replaced."""
        w_arr = w.const_value.numpy()
        out_channels = w_arr.shape[0]
        if out_channels % 8 == 0:
            return False
        return True


class MythicYoloPXPostprocessing(VidModel):
    """VidModel implementation for the postprocessing of the YoloPX pose estimation model.

    This is used as an example of how to implement a custom VidModel for the Mythic runtime and how to apply custom
    graph rewrites to make the model compatible with our compiler and runtime.
    """

    @classmethod
    def initialize_onnx(
        cls, model_directory: str | Path, config: Optional[InitializationConfig] = None
    ) -> onnx.ModelProto:
        """Return a runable ONNX ModelProto of the  model."""
        model = onnx.load(ONNX_PATH)

        # Remove redundant cast nodes and their graph input and output tensors from the model
        cls._remove_cast_nodes(model)

        return model

    def setup(self) -> None:
        """Extract inputs to be used for random input generation."""
        self.input_value_protos = [inp for inp in self._model_repr.graph.input]

    def preprocess(self, input_data: ImageDetectionInput) -> Any:
        """Preprocess an image by resizing and normalizing."""
        example_data = generate_random_inputs(self.input_value_protos, batch_size=1)
        return example_data

    def postprocess(self, model_output: Any, _: ImageDetectionInput) -> ImageDetectionOutput:  # type: ignore
        """Return dummy detection output."""
        # Each detection head (det_1/det_2/det_3) is a Concat of 3 vidConv branches, each padded
        # to 8 output channels, giving 24 total. Only the first 10 channels are valid data.
        assert len(model_output) == 3, f"Expected 3 output tensors (det_1/2/3), got {len(model_output)}."
        valid_indices = [0, 1, 2, 3, 8, 16, 17, 18, 19, 20]
        return {name: tensor[:, valid_indices, :, :] for name, tensor in model_output.items()}  # type: ignore

    @classmethod
    def load_default_dataset(cls) -> DummyDataset:
        """Return an initialized dataset that can be used to load data samples for this model."""
        return DummyDataset()

    @classmethod
    def _remove_cast_nodes(cls, model: onnx.ModelProto) -> onnx.ModelProto:
        # Collect all Cast nodes
        graph = model.graph
        cast_nodes = [n for n in graph.node if n.op_type == "Cast"]

        # Collect their inputs/outputs
        remove_inputs = set()
        remove_outputs = set()

        for node in cast_nodes:
            remove_inputs.update(node.input)
            remove_outputs.update(node.output)

        # Remove Cast nodes
        for node in cast_nodes:
            graph.node.remove(node)

        # Remove corresponding graph inputs
        for inp in list(graph.input):
            if inp.name in remove_inputs:
                graph.input.remove(inp)

        # Remove corresponding graph outputs
        for out in list(graph.output):
            if out.name in remove_outputs:
                graph.output.remove(out)

        return model

    def optimize_hook(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Rewrite most of the graph so it can be processed by us."""
        # Apply some manual changes to the graph after the optimizer has run.
        rule1 = ConvPadRewritePattern.rule()  # type: ignore[no-untyped-call]

        model, count1 = vid_match_patterns_onnxscript(model, rule1, verbose=0, commute=True)
        # model, count2 = vid_match_patterns_onnxscript(model, rule2, verbose=0, commute=True)

        # Check that the patterns were matched the expected number of times to ensure they are only applied where intended
        expected_count1 = 9
        if count1 != expected_count1:
            raise ValueError(f"Expected to match pattern {expected_count1} times, but matched {count1} times.")

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
    model = MythicYoloPXPostprocessing(result_directory)
    run_vnn_flow(model, result_directory, system_config=Path(__file__).parent / "system_configs" / "yolopx.cfg")
