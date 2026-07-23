import sys
from pathlib import Path
from typing import Any, Optional, cast


import numpy as np
import onnx

from vnnort import configure_logging
from vnnort.data.container import InputData, OutputData
from vnnort.models.initialization_config import InitializationConfig
from vnnort.models.vid_model import VidModel

# Python sets __package__=None when run as __main__, making relative imports fail.
# This ensures mythic_utils is on sys.path so it can be imported by name.
sys.path.insert(0, str(Path(__file__).parent))

from mythic_utils import DummyDataset  # noqa: E402

# Path to the source ONNX file, set by the CLI before instantiating the VidModel.
# This is very hacky but we do not have a better way right now to pass the CLI argument to the VidModel.
ONNX_PATH: str | Path = ""


class MythicResnet50Postprocessing(VidModel):
    """VidModel implementation for the postprocessing of the Resnet50 image classification model."""

    @classmethod
    def initialize_onnx(
        cls, model_directory: str | Path, config: Optional[InitializationConfig] = None
    ) -> onnx.ModelProto:
        """Return a runable ONNX ModelProto of the  model."""
        # ONNX_PATH must be set by the CLI before the VidModel is instantiated
        if not ONNX_PATH:
            raise ValueError(
                "ONNX_PATH is not set. " "This should have been set by the CLI when running the postprocessing example."
            )

        return onnx.load(ONNX_PATH)

    def setup(self) -> None:
        """Set up everything needed to run pre- and postprocessing."""
        # Extract inputs to be used for random input generation
        self.input_tensor_name = self._model_repr.graph.input[0].name

    def preprocess(self, input_data: InputData) -> Any:
        """Preprocess an image by resizing and normalizing."""
        data = {self.input_tensor_name: np.random.randn(1, 2048).astype(np.float32) * 20.0}
        return data

    def postprocess(self, model_output: Any, _: InputData) -> OutputData:
        """Return dummy detection output."""
        return cast(OutputData, model_output)

    @classmethod
    def load_default_dataset(cls) -> DummyDataset:
        """Return an initialized dataset that can be used to load data samples for this model."""
        return DummyDataset()


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
    model = MythicResnet50Postprocessing(result_directory)
    run_vnn_flow(
        model,
        result_directory,
        system_config=Path(__file__).parent / "system_configs" / "resnet50.cfg",
        run_full_flow=True,
    )
