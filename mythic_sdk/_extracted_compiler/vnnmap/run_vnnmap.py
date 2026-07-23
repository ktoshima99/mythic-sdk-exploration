import logging
import os
import re
import shutil
import site
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

logger = logging.getLogger("vNN-Mapper")

DEFAULT_N_MPS = 1
DEFAULT_OCRAM0 = 0  # Not used by default
DEFAULT_OCRAM1 = 2_097_152  # 2 MB
DEFAULT_DDR = 536_870_912  # 512 MB
DEFAULT_FREQUENCY = 625_000_000  # 625 MHz
DEFAULT_DDR_CONFIG = 100


def _find_executable() -> Path:
    """Find and return the path of the vnnmap executable.

    Raises:
        FileNotFoundError: If the vnnmap executable is not found.
        PermissionError: If the vnnmap executable is found but is not executable.

    Returns:
        Path: Path to the vnnmap executable.
    """
    # Search site-packages explicitly so editable installs (where __file__
    # points to the source tree) still find the installed binary.
    candidates = [Path(d) / "vnnmap" / "vnnmap" for d in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / "vnnmap" / "vnnmap")
    # Fallback: same directory as this file (regular install)
    candidates.append(Path(__file__).parent / "vnnmap")

    for candidate in candidates:
        if candidate.exists():
            if not os.access(candidate, os.X_OK):
                raise PermissionError(f"Not executable: {candidate}")
            return candidate

    raise FileNotFoundError(f"vnnmap executable not found in any expected location: {[str(c) for c in candidates]}")


def _build_system_config_string(nMPs: int, OCRAM0: int, OCRAM1: int, DDR: int, frequency: int, DDRConfig: int) -> str:
    """Build and return the system configuration string for vnnmap."""
    return (
        "[sys]\n"
        f"nMPs={nMPs}\n"
        f"OCRAM0={OCRAM0}\n"
        f"OCRAM1={OCRAM1}\n"
        f"DDR={DDR}\n"
        f"frequency={frequency}\n"
        f"DDRConfig={DDRConfig}\n"
    )


def _create_system_config(
    file_path: str,
    nMPs: int,
    OCRAM0: int,
    OCRAM1: int,
    DDR: int,
    frequency: int,
    DDRConfig: int,
) -> None:
    """Create a system configuration file for vnnmap and write it to the specified path."""
    with open(file_path, mode="w") as f:
        f.write(_build_system_config_string(nMPs, OCRAM0, OCRAM1, DDR, frequency, DDRConfig))


def _extract_metrics(text: str) -> Dict[str, Any]:
    """Extract key numeric values from a profiling text report.

    Returns a nested dictionary with parsed numeric metrics.
    """
    # Output text looks like this:
    """
     maxDDR:   11605.23kB (  11605.23kB)     (wgt/bas/sft:   11408.25kB, inp/out:     196.98kB)
    #MACs:   1853411328
    maxOCR:    1001.30kB (   1001.30kB)     (wgt/bas/sft:      21.30kB, inp/out:     980.00kB)

    Cycles per inference:
    MAC:           32177278
    non MAC:         161504
    exposed DMA:    2565801
    Total:         34904583

    Inference performance:
    eff. fps:  17.91 (DMAs exposed)  eff. latency:  55.85 ms         efficiency: 82.97% (batch size: 1)

    MACs:1.853 bn PARAMs:11.688 mn (11.152 MB) DDR Read:( 11.4 MB) DDR Write:(  0.0 MB)
    """

    def find(pattern: str, cast: Callable[[str], Any] = float) -> Any:
        m = re.search(pattern, text, re.MULTILINE)
        return cast(m.group(1)) if m else None

    metrics = {
        # Memory
        "Max DDR (kB)": find(r"maxDDR:\s+([\d.]+)kB"),
        "Max DDR Weights (kB)": find(r"wgt/bas/sft:\s+([\d.]+)kB"),
        "Max DDR IO (kB)": find(r"inp/out:\s+([\d.]+)kB"),
        "Max OCR (kB)": find(r"maxOCR:\s+(\d+\.\d+)kB"),
        "Max OCR WEIGHTS (kB)": find(r"maxOCR:.*?wgt\/bas\/sft:\s+(\d+\.\d+)kB"),
        "Max OCR IO (kB)": find(r"maxOCR:.*?inp\/out:\s+(\d+\.\d+)kB"),
        # Compute
        "MAC Cycles": find(r"MAC:\s+(\d+)", int),
        "Non MAC Cycles": find(r"non MAC:\s+(\d+)", int),
        "Exposed DMA Cycles": find(r"exposed DMA:\s+(\d+)", int),
        "Total Cycles": find(r"Total:\s+(\d+)", int),
        # Performance
        "Effective FPS": find(r"eff\. fps:\s+([\d.]+)", float),
        "Effective Latency (ms)": find(r"eff\. latency:\s+([\d.]+)\s*ms"),
        "Efficiency (%)": find(r"efficiency:\s+([\d.]+)%"),
        "Batch Size": find(r"batch size:\s+(\d+)", int),
        # Power
        "Power@30fps (mW)": find(r"Power@30fps:\s+([\d.]+)\s*mW", float),
        "Power@eff. fps (mW)": find(r"Power@eff\. fps:\s+([\d.]+)\s*mW", float),
        # Model stats
        "MACs (bn)": find(
            r"MACs:\s*([\d.]+)\s*bn",
        ),
        "Parameters (mn)": find(r"PARAMs:\s*([\d.]+)\s*mn"),
        "Model Size (MB)": find(r"\(([\d.]+)\s*MB\)"),
        "DDR Read (MB)": find(r"DDR Read:\(\s*([\d.]+)\s*MB\)", float),
        "DDR Write (MB)": find(r"DDR Write:\(\s*([\d.]+)\s*MB\)", float),
    }
    for k, v in metrics.items():
        if v is None:
            logger.warning("There was no match found for %s", k)

    ddr_read = metrics["DDR Read (MB)"]
    ddr_write = metrics["DDR Write (MB)"]
    eff_fps = metrics["Effective FPS"]
    metrics["Total Bandwidth"] = (
        (ddr_read + ddr_write) * eff_fps / 1024
        if ddr_read is not None and ddr_write is not None and eff_fps is not None
        else None
    )

    mac_cycles = metrics["MAC Cycles"]
    non_mac_cycles = metrics["Non MAC Cycles"]
    metrics["Proc Cycles"] = (
        mac_cycles + non_mac_cycles if mac_cycles is not None and non_mac_cycles is not None else None
    )

    return metrics


def _load_layerwise_stats(csv_path: str | Path) -> pd.DataFrame:
    """Extract layerwise stats for all layers from a generated flow_csv."""
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="skip",
    )

    return df


def run_vnnmap(
    vidir_path: str | Path,
    result_directory: str | Path,
    n_mps: int = DEFAULT_N_MPS,
    ocram1_bytes: int = DEFAULT_OCRAM1,
    generate_vci: bool = True,
    system_config: Path | None = None,
    advanced: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run vnnmap on the provided .vidir file and store results in the specified directory.

    For now the only thing configurable is the number of MP cores to be used.

    Args:
        vidir_path (str | Path): Path to the .vidir file to be processed.
        result_directory (str | Path): Directory where results will be stored.
        n_mps (int): Number of MP cores to be used. Ignored when system_config is provided.
        ocram1_bytes (int): Size of the OCRAM1 in bytes. Ignored when system_config is provided.
        generate_vci (bool): Whether to generate the vci or not.
        system_config (Path | None): Path to a pre-existing system config file. When provided,
            this file is passed directly to vnnmap instead of generating one from n_mps/ocram1_bytes.
        advanced (bool): Whether to enable advanced exploration mode in vnnmap. Default is False.
            Note this flag has no effect in the EVAL package of the v-NN SDK.

    Raises:
        RuntimeError: If vnnmap execution fails.

    Returns:
        tuple[dict[str, Any], pd.DataFrame]: Tuple of metrics dict and layerwise stats DataFrame.
    """
    if generate_vci and advanced:
        raise RuntimeError("Error: generate_vci and advanced flags may not be set together!")

    # Create result directory
    result_directory = Path(result_directory).absolute()
    if result_directory.exists():
        logger.info("Removing existing vnnmap result directory: %s", result_directory)
        shutil.rmtree(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)

    if system_config is not None:
        system_config_path = Path(system_config)
    else:
        system_config_path = result_directory / "system.cfg"
        _create_system_config(
            str(system_config_path),
            nMPs=n_mps,
            OCRAM0=DEFAULT_OCRAM0,
            OCRAM1=ocram1_bytes,
            DDR=DEFAULT_DDR,
            frequency=DEFAULT_FREQUENCY,
            DDRConfig=DEFAULT_DDR_CONFIG,
        )

    # Prepare and run vnnmap command
    executable_path = _find_executable()
    vidir_path = Path(vidir_path).absolute()
    model_name = vidir_path.stem.split(".")[0]
    args = [
        str(executable_path),
        f"--csv_dir={result_directory}/csv",
        f"--output_prefix={model_name}",
        f"--network={vidir_path}",
        f"--system_cfg={system_config_path}",
    ]
    if generate_vci:
        args.append("--codegen")
    if advanced:
        args.append("--explore")
        args.append("--edma")
    # Run from result directory
    vci_output_path = result_directory / (model_name + ".vci")
    csv_flow_output_path = result_directory / f"csv/{model_name}_flow.csv"

    result = subprocess.run(args, capture_output=True, text=True, cwd=result_directory)
    if result.returncode != 0:
        msg = f"vnnmap execution failed with return code {result.returncode}. Error: \n{result.stderr}"
        raise RuntimeError(msg)
    if generate_vci:
        logger.info(f"You can find the result VCI in {vci_output_path}")

    metrics = _extract_metrics(result.stdout)
    layerwise_stats = _load_layerwise_stats(csv_flow_output_path)
    return metrics, layerwise_stats
