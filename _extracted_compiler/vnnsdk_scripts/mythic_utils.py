import argparse
import configparser
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import onnx

if TYPE_CHECKING:
    from vnnort.models.vid_model import VidModel

from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import InputData
from vnnort.inference.engine import InferenceEngine

MODEL_PARAMETER_FIELDS = {
    "MACs (bn)": ("MACs (bn)", "macs_bn", "{:,.3f}"),
    "Model Size (MB)": ("Model Size (MB)", "model_size_mb", "{:,.3f}"),
}

PROFILING_FIELDS = {
    "Total Cycles": ("Total Cycles", "total_cycles", "{:,.0f}"),
    "FPS": ("Effective FPS", "fps", "{:6.2f}"),
    "Power@5nm (mW)": ("Power@eff. fps (mW)", "power_mw_5nm_typ", "{:6.2f}"),
    "Power@5nm, 30fps (mW)": ("Power@30fps (mW)", "power_mw_5nm_typ_30fps", "{:6.2f}"),
    "MAC Utilization %": ("Efficiency (%)", "mac_utilization_pct", "{:7.3f}"),
}

METRIC_FIELDS = {**MODEL_PARAMETER_FIELDS, **PROFILING_FIELDS}


def extract_metrics(stats: dict[str, str]) -> dict[str, float]:
    """Extract the subset of metrics we care about as raw floats."""
    return {json_key: float(stats.get(in_key)) for _, (in_key, json_key, _) in METRIC_FIELDS.items()}


def _print_section(fields: dict[str, tuple[str, str, str]], extracted: dict[str, float]) -> None:
    for display_key, (_, json_key, fmt) in fields.items():
        print(f"{display_key:<25} : {fmt.format(extracted[json_key])}")


def pprint_metrics(stats: dict[str, str]) -> dict[str, float]:
    """Print metrics in a readable form and return the raw float values."""
    extracted = extract_metrics(stats)
    print("#######################################")
    print("Model Parameters:")
    _print_section(MODEL_PARAMETER_FIELDS, extracted)
    print("\n#######################################")
    print("Profiling Results:")
    _print_section(PROFILING_FIELDS, extracted)
    return extracted


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the v-NN ORT flow.

    Returns:
        argparse.Namespace: Parsed command line arguments with source ONNX path and result directory.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process a source ONNX file and write results to a directory.")
    parser.add_argument(
        "--source_onnx",
        type=Path,
        required=True,
        help="Path to the source ONNX file",
    )
    parser.add_argument(
        "--result_directory",
        type=Path,
        required=True,
        help="Directory where results will be written",
    )
    parser.add_argument(
        "--nMPs",
        type=int,
        default=None,
        help="Compatibility argument but ignored (NOP).",
    )
    args = parser.parse_args()

    return args


def read_frequency_from_sys_config(path: Union[str, Path]) -> int:
    """Read the ``frequency`` entry from the ``[sys]`` section of a config file.

    The file is expected to look like::

        [sys]
        nMPs=1
        OCRAM1=4194304
        frequency=1000000000
        DDRConfig=100

    Args:
        path (Union[str, Path]): Filesystem path to the configuration file.

    Returns:
        int: The integer value of the ``frequency`` field.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if the section or the key is missing.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found at expected location: {str(path)}")
    parser = configparser.ConfigParser()
    parser.read(path)

    if "sys" not in parser:
        raise KeyError(f"no [sys] section in {path}")

    freq_str = parser["sys"].get("frequency")
    if freq_str is None:
        raise KeyError("frequency key not found in [sys] section")
    return int(freq_str)


def _sample_onnx_inputs(model_proto: onnx.ModelProto, n_samples: int = 3, seed: int = 42) -> list[dict[str, Any]]:
    """Generate reproducible random inputs from the ONNX model's graph input specs.

    Uses a fixed seed so the same inputs can be replayed before and after optimization.
    Unknown/symbolic dimensions (e.g. batch) default to 1.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        inp_dict: dict[str, Any] = {}
        for graph_input in model_proto.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in graph_input.type.tensor_type.shape.dim]
            inp_dict[graph_input.name] = rng.standard_normal(shape).astype(np.float32)
        samples.append(inp_dict)
    return samples


def _check_numerical_equivalence(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    atol: float = 1e-3,
) -> None:
    """Compare raw tensor output dicts and raise if mean L1 distance exceeds atol."""
    for i, (b, a) in enumerate(zip(before, after)):
        if b.keys() != a.keys():
            raise ValueError(f"Sample {i}: output keys differ: {set(b.keys())} vs {set(a.keys())}")
        for name in b:
            l1 = float(np.abs(b[name] - a[name]).mean())
            if l1 > atol:
                raise ValueError(
                    f"Sample {i}, output '{name}': mean L1 distance {l1:.6f} exceeds threshold {atol}. "
                    "Optimization changed model outputs beyond tolerance."
                )


def run_vnn_flow(
    model: "VidModel",
    result_directory: Path,
    system_config: Path,
    run_full_flow: bool = False,
    skip_validation: bool = False,
    advanced: bool = False,
) -> None:
    """Run the v-NN ORT flow on the provided model and store results in the specified directory.

    Args:
        model(VidModel): The model to be processed. Should be an instance of VidModel.
        result_directory (Path): Directory where results will be stored.
        system_config (Path): Path to the system config file passed directly to vnnmap.
        run_full_flow (bool): Whether to run the full flow including codegen and simulation. Default is False,
            which will only run up to exploration.
        skip_validation (bool): When True, skip the pre/post-optimization numerical equivalence check.
            Needed for models whose postprocess() rejects ``input_data=None`` or returns a non-dict
            container (e.g. BevformerTiny returns MultiViewDetection3DOutput and type-checks its input).
        advanced (bool): Whether to enable advanced exploration mode in vnnmap. Default is False.
            Note this flag has no effect in the EVAL package of the v-NN SDK.
    Returns:
        None:
    """
    from vidsim import run_vidsim
    from vnncodegen import run_codegen
    from vnnmap import explore_model, run_compilation
    from vnnort.quantizer.quantization_config import QuantizationConfig

    cfg = configparser.ConfigParser()
    cfg.read(system_config)
    n_mps = int(cfg["sys"]["nMPs"])
    frequency = int(cfg["sys"]["frequency"])

    result_directory = result_directory.resolve()  # Resolve to absolute path

    if skip_validation:
        model.optimize()
    else:
        # Generate fixed random inputs from the ONNX graph before optimization.
        # Passing raw dicts to InferenceEngine bypasses pre/postprocessing.
        sample_inputs = _sample_onnx_inputs(model._model_repr)  # type: ignore[attr-defined]
        init_outputs = [InferenceEngine(model).run(inp) for inp in sample_inputs]

        model.optimize()

        opt_outputs = [model.postprocess(InferenceEngine(model).run(inp), None) for inp in sample_inputs]

        _check_numerical_equivalence(init_outputs, opt_outputs)  # type: ignore[arg-type]

    model.save(result_directory / (model.model_name + ".vido.onnx"))

    # Run dummy quantization with 1 random sample.
    model.quantize(QuantizationConfig(calibration_dataset_size=1))

    metrics, _ = explore_model(model, system_config=system_config, advanced=advanced)

    print("#######################################")
    print("System Config:")
    print(f"{'Clock Frequency (GHz):':<25} : {frequency / 1e9:.2f}")
    print(f"{'Number of MPs:':<25} : {n_mps}")

    print()
    extracted_metrics = pprint_metrics(metrics)

    profilings = {
        "system_config": {section: dict(cfg[section]) for section in cfg.sections()},
        "profiling": extracted_metrics,
    }
    with open(result_directory / f"{model.model_name}_proflings.json", "w") as f:
        json.dump(profilings, f, indent=2)

    if run_full_flow:
        run_compilation(model, system_config=system_config)
        run_codegen(model, n_mps=n_mps)
        run_vidsim(model, n_mps=n_mps)


class DummyDataset(DatasetBase):
    """Dummy dataset that returns random data.

    This is used for the postprocessing example since we do not have the
    actual postprocessing code and thus cannot use the real dataset.
    """

    def __len__(self) -> int:
        """Return the length of the dataset.

        Since we will not actually use the dataset for anything meaningful, we can just return a dummy length here.
        """
        return 10000

    def __getitem__(self, index: int) -> Any:
        """Return a random data sample."""
        # Return dummy data
        return InputData()

    def get_benchmark(self) -> None:
        """Return None since we will not actually use it."""
        return None
