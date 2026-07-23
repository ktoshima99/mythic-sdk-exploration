from typing import Any

from numpy.typing import NDArray
from tqdm import tqdm

from vnnort.data.dataloader import Dataloader
from vnnort.models.vid_model import VidModel
from vnnort.quantizer.calibrator.histogram_hook import DynamicNDHistogram, NDHistogram
from vnnort.quantizer.quant_utils import TensorQuantInfo
from vnnort.utils.onnx_utils.hooked_inference import HookedOnnxInferenceSession
from vnnort.utils.onnx_utils.onnx_hooks import OnnxOutputHook

DEBUG_DATA_SAMPLE_INDEX = 2  # Sample dataset index to use for debugging data export


class TensorStatisticCollector:
    """Calibrator, which can be used to collect statistics of intermediate tensors in an ONNX model.

    It works by adding a MinMaxHook to each tensor in the model and then running the model with each
    sample provided by the dataloader. All hooks collect intermediate statistics for each sample and
    then return the collected data to the user.
    """

    def __init__(
        self,
        model: VidModel,
        dataloader: Dataloader,
        tensor_quant_infos: dict[str, TensorQuantInfo],
        percentile_histogram_bins: int,
        n_workers: int,
    ):
        """Initialize a new MinMaxCalibrator.

        Args:
            model (VidModel): ONNX model to calibrate.
            dataloader (Dataloader): Dataloader to use for calibration.
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects.
            percentile_histogram_bins(int): How many histogram bins to use.
            n_workers (int): How many workers to use for inference.
        """
        self._model = model
        self._dataloader = dataloader
        self._tensor_quant_infos = tensor_quant_infos
        self._percentile_histogram_bins = percentile_histogram_bins
        self._n_workers = n_workers

    def run(self) -> None:
        """Run the calibration and attach the collected tensor histograms to the corresponding TensorQuantObjects."""
        # Run hooked inference session on dynamic data
        histograms = self._collect_histograms()

        # Also collect one sample of each intermediate tensor and add it to the tensor info
        tensor_samples = self._collect_tensor_samples()

        # Update entries in TensorQuantInfo objects
        for name, histogram in histograms.items():
            self._tensor_quant_infos[name].tensor_histograms = histogram
            self._tensor_quant_infos[name].tensor.data = tensor_samples[name]

    def _collect_histograms(self) -> dict[str, NDHistogram]:
        """Run the model with each sample in the dataloader and collect statistics for each tensor.

        The collected data will be stored in the MinMaxHook objects.
        """
        # Generate histogram hooks

        histogram_hooks = {
            name: self._generate_histogram_hook(quant_info) for name, quant_info in self._tensor_quant_infos.items()
        }

        # Run hooked inference session for multiple data samples and collect statistics
        with HookedOnnxInferenceSession.create(
            self._model._model_repr, histogram_hooks, n_workers=self._n_workers
        ) as session:
            for input_data, model_input in tqdm(self._dataloader, desc="Collecting tensor statistics"):
                session.run(model_input)
            results = session.results()

        # Extract histograms
        histograms = {name: hook.histograms for name, hook in results.items()}  # type: ignore

        if not all(isinstance(histogram, NDHistogram) for histogram in histograms.values()):
            raise RuntimeError("Not all tensors have collected statistics!")

        return histograms  # type: ignore

    def _collect_tensor_samples(self) -> dict[str, NDArray[Any]]:
        output_hooks = {name: OnnxOutputHook() for name in self._tensor_quant_infos.keys()}
        n_workers = 1  # Only one sample is propagated, no need for many workers

        with HookedOnnxInferenceSession.create(self._model._model_repr, output_hooks, n_workers=n_workers) as session:
            input_data, model_input = self._dataloader._load_item(DEBUG_DATA_SAMPLE_INDEX)
            session.run(model_input)

            results = session.results()
        outputs = {name: hook.compute()[0] for name, hook in results.items()}  # type: ignore
        return outputs

    def _generate_histogram_hook(self, quant_info: TensorQuantInfo) -> DynamicNDHistogram:
        return DynamicNDHistogram(
            quant_info.tensor.name, axis=quant_info.axis, n_bins=self._percentile_histogram_bins, abs_values=True
        )
