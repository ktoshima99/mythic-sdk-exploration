#
# Copyright (C) 2021, Mythic Inc. All rights reserved.
#
"""Compile a Munc artifact into a firmware artifact."""
import json
import logging
import shutil
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from contextlib import suppress

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from mythic.nvm_program.api.generate_weight_binary import generate_weight_binaries

from mythic.model_deployment.rmcr.common import get_temp_dir_root, get_compiler_docker_image, compiler_main
from mythic.model_deployment.rmcr.firmware_artifact import get_firmware_dir, load_munc_artifact, save_artifact
from mythic.model_deployment.rmcr.compiler import CompilerDockerConfig, dnn_fw_compile
from mythic.model_deployment.rmcr.vnn_compile import (
    resolve_off_chip_onnx,
    vnn_compile,
)


logger = logging.getLogger(__name__)


KEY_TO_OPTION = {
    "TARGET_FRAME_RATE_FILE": "--target-frame-rate-filename",
    "MMA_FRAME_RATE_FILE": "--mma-frame-rate-filename",
    "NP_FRAME_RATE_FILE": "--np-frame-rate-filename",
    "PARALLELIZATION_FILE": "--parallelization-filename",
    "ASSIGNMENTS_FILE": "--assignments-filename",
}


def build_file_options_from_cfg(cfg, docker_config, compiler_config_dir=None):
    """Build staged file options from an RMCR config."""
    file_map = {}
    for key in KEY_TO_OPTION:
        val = cfg.get(key)
        if val:
            file_map[key] = val
    return build_file_options_from_map(file_map, docker_config, compiler_config_dir)


def build_file_options_from_map(file_map, docker_config, compiler_config_dir=None):
    """Build staged file options from a dict of config keys to file paths.

    Keys should match known file config keys (e.g., "ASSIGNMENTS_FILE"). Values can be absolute or relative paths.
    """
    compiler_config_dir = Path(compiler_config_dir) if compiler_config_dir else None
    staging_dir = Path(docker_config.local_work_dir) / "compiler_config_files"

    def resolve_source_path(path_value):
        """Resolve a PATH relative to compiler_config_dir when not absolute."""
        path_value = Path(path_value)
        if not path_value.is_absolute() and compiler_config_dir:
            return compiler_config_dir / path_value
        return path_value

    def stage_file(path_value, dest_name):
        """Copy a resolved PATH into the Docker work dir using dest_name."""
        source_path = resolve_source_path(path_value)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Config '{dest_name}' with PATH '{path_value}' resolved to '{source_path}', which does not exist."
            )
        destination = staging_dir / dest_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return destination

    options = []
    for key, path_value in file_map.items():
        option = KEY_TO_OPTION.get(key)
        if not option or not path_value:
            continue
        staged_path = stage_file(path_value, key)
        filename = docker_config.local_path_to_container(staged_path)
        options.append(f"{option} {filename}")

    return options


def get_on_chip_onnx(munc_artifact_dir, munc_artifact_metadata):
    """Get the on-chip ONNX file from the Munc artifact."""
    onnx_files = munc_artifact_metadata['onnx_graphs']
    on_chip_stages = list(filter(lambda x: "on_chip" in x, onnx_files))
    assert len(on_chip_stages) == 1, "Only one on-chip stage is supported"
    onnx_file = onnx_files[on_chip_stages[0]]
    onnx_file_path = munc_artifact_dir / onnx_file
    return onnx_file_path


def _run_vnn(cfg, munc_artifact_dir, munc_artifact_metadata, docker_config, work_dir, compiler_config_path):
    """Run the v-NN script. Returns True if executed, False if skipped.

    Silently skips when the target arch is not m2000, when the artifact has no `off_chip_2`
    ONNX graph, or when the config explicitly sets `VNN: null`. Raises when VNN is enabled
    and an off_chip_2 ONNX exists but no VNN script can be resolved.
    """
    if "m2000" not in get_arch(cfg):
        logger.info("Skipping v-NN step: only supported on m2000.")
        return False

    vnn_cfg = cfg.get("VNN")
    if vnn_cfg is None:
        logger.info("Skipping v-NN step: `VNN: null` in compiler config.")
        return False

    off_chip_onnx = resolve_off_chip_onnx(munc_artifact_dir, munc_artifact_metadata)
    if off_chip_onnx is None:
        logger.info("Skipping v-NN step: no off_chip_2 postprocessing graph in artifact.")
        return False

    local_script = vnn_cfg.get("LOCAL_SCRIPT")
    if local_script:
        local_script = Path(local_script)
        if not local_script.is_absolute() and compiler_config_path:
            local_script = compiler_config_path / local_script

    vnn_compile(docker_config, off_chip_onnx, get_firmware_dir(work_dir) / "vnn", vnn_cfg, local_script)
    return True


def compile_artifact(cfg, input_artifact, output_artifact):
    """Generate a firmware from a training artifact."""
    compiler_config_path = get_compiler_config_dir()
    logger.info(f"Using compiler configuration directory {compiler_config_path}")
    logger.info(f"input_artifact = {input_artifact}")

    with TemporaryDirectory(dir=get_temp_dir_root()) as work_dir:
        work_dir = Path(work_dir)
        amp_arch = get_arch(cfg)

        logger.info(f"Loading Munc artifact from {input_artifact}")
        munc_artifact_dir, munc_artifact_metadata = load_munc_artifact(input_artifact, work_dir)

        onnx_file_path = get_on_chip_onnx(munc_artifact_dir, munc_artifact_metadata)
        docker_config = CompilerDockerConfig(get_compiler_docker_image(cfg), work_dir)
        fw_source_dir = work_dir / "source_code"
        if "m1000" in amp_arch:
            fw_bin_dir = work_dir / "fw_bin"
        elif "m2000" in amp_arch:
            fw_bin_dir = fw_source_dir / "funcsim_build"
        programming_data_dir = work_dir / "programming_data"

        compiler_options = list(cfg.COMPILER_OPTIONS)
        if cfg.INCLUDE_FW_SOURCE and "-X" not in compiler_options:
            compiler_options.append("-X")

        logger.info("Compiling model")
        file_options = build_file_options_from_cfg(
            cfg, docker_config, compiler_config_dir=compiler_config_path.parent if compiler_config_path else None
        )
        dnn_fw_compile(docker_config, onnx_file_path, fw_source_output_dir=fw_source_dir, fw_bin_output_dir=fw_bin_dir,
                       compile_options=compiler_options,
                       fw_compile_options=cfg.FW_OPTIONS,
                       file_options=file_options)

        if "m1000" in amp_arch:
            # Generate weight binaries for this network. manifest.yml in programming_data_dir will contain info about
            # the files generated.
            generate_weight_binaries(runtime_yml=str(fw_bin_dir / "runtime.yml"),
                                     final_l0_pb=str(fw_source_dir / "l0/final.l0.pb"),
                                     weight_path=str(programming_data_dir),
                                     # mma_clk and si_cfg are not actually used and should be removed after refactoring
                                     # of generate_weight_binaries.
                                     mma_clk=35.714, si_cfg="b0v3a")

        _run_vnn(cfg, munc_artifact_dir, munc_artifact_metadata, docker_config, work_dir, compiler_config_path)

        metadata = dict(config=OmegaConf.to_container(cfg, resolve=True), cli_args=sys.argv, amp_arch=amp_arch)
        create_compiler_artifact(work_dir, fw_source_dir, fw_bin_dir, programming_data_dir, munc_artifact_dir,
                                 cfg.INCLUDE_FW_SOURCE, metadata)
        save_artifact(work_dir, output_artifact)


def vnn_compile_artifact(cfg, input_artifact, output_artifact):
    """Run only the v-NN script on a Munc artifact (no DNN/FW compilation)."""
    compiler_config_path = get_compiler_config_dir()
    logger.info(f"Using compiler configuration directory {compiler_config_path}")
    logger.info(f"input_artifact = {input_artifact}")

    with TemporaryDirectory(dir=get_temp_dir_root()) as work_dir:
        work_dir = Path(work_dir)
        logger.info(f"Loading Munc artifact from {input_artifact}")
        munc_artifact_dir, munc_artifact_metadata = load_munc_artifact(input_artifact, work_dir)

        docker_config = CompilerDockerConfig(get_compiler_docker_image(cfg), work_dir)
        if not _run_vnn(cfg, munc_artifact_dir, munc_artifact_metadata,
                        docker_config, work_dir, compiler_config_path):
            raise SystemExit("--vnn-only: v-NN step was skipped (see log for reason).")
        save_artifact(work_dir, output_artifact)


def create_compiler_artifact(artifact_root_directory, fw_source_dir, fw_bin_dir, programming_data_dir,
                             munc_artifact_dir, include_fw_source, metadata):
    """Create a compiler artifact by copying relevant files from the firmware source and binary directories."""
    artifact_dir = get_firmware_dir(artifact_root_directory)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def include(src, dest_name=None, move=True):
        if dest_name is None:
            dest_name = src.name
        if move:
            src.rename(artifact_dir / dest_name)
        else:
            shutil.copy2(src, artifact_dir / dest_name)
    include(fw_source_dir / "l0/final.l0.pb", move=False)
    include(fw_source_dir / "l0_onnx/final.onnx.pb", move=False)
    # DNN Compiler outputs are different depending on the AMP architecture.
    if "m1000" in metadata["amp_arch"]:
        include(fw_bin_dir / "runtime.yml")
        # Why?
        rename_programming_firmware_files(programming_data_dir)
        include(programming_data_dir, "programming_data")
        if include_fw_source:
            include(fw_source_dir, "source_code")
        include(fw_bin_dir / "weights")
    elif "m2000" in metadata["amp_arch"]:
        include(fw_bin_dir / "program_structs_gen.pb")
        include(fw_source_dir / "weights")

        # Copy report files if they exist.
        for report in [
            "weight_utilization.txt",
            "sram_utilization.txt",
            "ace_utilization.txt",
            "assignments.txt",
        ]:
            report_file = fw_source_dir / "reports" / report
            if report_file.exists():
                include(report_file, move=False)
            else:
                logger.warning(f"{report} report not found; skipping.")

        # Copy dnn_compiler's usage.txt if it exists.
        usage_file = fw_source_dir / "usage.txt"
        if usage_file.exists():
            include(usage_file, move=False)
        else:
            logger.warning("dnn_compiler usage.txt not found; skipping.")

    # Why?
    include(munc_artifact_dir / "contents.json", move=False)
    # This renaming is needed for historical reasons - old versions of Munc artifacts didn't have a fixed name for the
    # top level directory. Now it's always `compiler_ready_artifact`.
    include(munc_artifact_dir, "munc_artifact")

    write_metadata(artifact_dir, metadata)


def write_metadata(artifact_dir, metadata):
    """Add metadata to the artifact."""
    metadata_file = artifact_dir / "contents.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = {}
    existing_metadata.update(metadata)
    with open(metadata_file, "w") as f:
        json.dump(existing_metadata, f, indent=4)


def rename_programming_firmware_files(programming_data_dir):
    """Rename programming firmware files."""
    # This is a programming stage to new firmware file name mapping.
    stage_to_firmware_file_name = {"v1p5": "nvm_tcc.yml", "v3p0": "lms_tcc.yml"}
    manifest = OmegaConf.load(programming_data_dir / "manifest.yml")
    for stage, files in manifest.items():
        new_file_name = stage_to_firmware_file_name[stage]
        (programming_data_dir / files["firmware"]).rename(programming_data_dir / new_file_name)
    # Delete now invalid manifest.
    (programming_data_dir / "manifest.yml").unlink()


def get_compiler_config_dir():
    """Get the compiler configuration file directory."""
    hydra_cfg = HydraConfig.get()
    for source in hydra_cfg.runtime.config_sources:
        if source.provider == "main":
            compiler_config_path = Path(source.path).expanduser()
            if not compiler_config_path.is_absolute():
                compiler_config_path = compiler_config_path.resolve()
            return compiler_config_path
    providers = [s.provider for s in hydra_cfg.runtime.config_sources]
    raise RuntimeError(f"Could not find the 'main' config source among Hydra providers: {providers}")


def get_arch(cfg):
    """Fetch a hardware architecture from the config."""
    if any("--amp-arch boreas" in s for s in cfg.COMPILER_OPTIONS):
        arch = "m1000"
    elif any("--amp-arch m2" in s for s in cfg.COMPILER_OPTIONS):
        arch = "m2000"
    else:
        logger.warning("Could not find expected AMP architecture, defaulting to m1000.")
        arch = "m1000"
    return arch


def rewrite_argv(argv, config_file_arg="--compiler-config"):
    """Rewrite the standard compile.py command line parameters as Hydra key overrides.

    This is for backward compatibility only.
    """
    argv = list(argv)

    config_path = None
    if config_file_arg:
        try:
            pos = argv.index(config_file_arg)
            argv.pop(pos)
            try:
                config_path = Path(argv.pop(pos)).absolute()
            except IndexError:
                print(f"Missing argument value for {config_file_arg}")
                sys.exit(1)
        except ValueError:
            pass

    has_hydra_config = any(opt in argv for opt in ("-cp", "--config-path", "-cn", "--config-name"))
    if not config_path and not has_hydra_config:
        raise SystemExit(f"Missing {config_file_arg}")

    extra_keys = []
    with suppress(ValueError):
        pos = argv.index("--test")
        argv.pop(pos)
        extra_keys.append("++WANDB.TEAM=${WANDB.TEAM_TEST}")

    to_conf_keys = {
        "--run-date": "run_date",
        "--build-id": "build_id",
        "--build-user": "build_user",
        "--input-artifact": "src",
        "--output-artifact": "dest"
    }

    for arg, key in to_conf_keys.items():
        try:
            pos = argv.index(arg)
            argv.pop(pos)
            try:
                val = argv.pop(pos)
                extra_keys.append(f'++{key}="{val}"')
            except IndexError:
                print(f"Missing argument value for {arg}")
                sys.exit(1)
        except ValueError:
            pass

    config_options = ["-cn", config_path.name, "-cp", str(config_path.parent)] if config_path else []
    return [argv[0]] + config_options + argv[1:] + extra_keys


def main():  # noqa: D103
    # my_app is local to avoid side-effects of @compiler_main(), when this file is used as a library.

    @compiler_main()
    def my_app(cfg: DictConfig) -> None:  # noqa: D103
        if cfg.VNN_ONLY:
            vnn_compile_artifact(cfg, cfg.src, cfg.dest)
        else:
            compile_artifact(cfg, cfg.src, cfg.dest)

    sys.argv = rewrite_argv(sys.argv)
    my_app()


if __name__ == "__main__":
    main()
