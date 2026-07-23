# Copyright (C) 2021, Mythic Inc. All rights reserved.
#

"""Support for compiling DNNs into firmware binaries."""

import collections
import logging
import sys

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import docker

logger = logging.getLogger(__name__)


@dataclass
class CompilerDockerConfig():
    """Configuration for running the compiler in a Docker container."""

    docker_image: str
    # Local paths passed to "compile" functions are translated to paths in a container using this directory as
    # the local base directory.
    local_work_dir: Path
    # In a Docker-in-Docker setup, our view of the file system may be different from the host one. This directory is
    # the path to the host's view of the local work directory. By default, it is the same as the local work directory.
    host_work_dir: Optional[Path] = None
    # Local paths passed to "compile" functions are translated to paths in a container using this directory as
    # the in-container base directory.
    container_work_dir: Path = Path("/work-dir")

    def __post_init__(self):
        """Set the host work directory to the local work directory if not specified."""
        if not self.host_work_dir:
            self.host_work_dir = self.local_work_dir

    def local_path_to_container(self, local_path: Path) -> str:
        """Translate a local path to a path in the container.

        The local path should be within the local work directory (`self.local_work_dir`).
        """
        return str(self.container_work_dir / local_path.relative_to(self.local_work_dir))


def run_in_docker(docker_config, command, tty_output=False, shm_size=None):
    """Run a command in a Docker container using the provided container configuration."""
    client = docker.from_env()
    container = None
    try:
        volumes = {str(docker_config.host_work_dir): {"bind": str(docker_config.container_work_dir), "mode": "rw"}}
        extra_kwargs = {}
        if shm_size:
            extra_kwargs["shm_size"] = shm_size
        # Why do we need `-l` in bash command?
        # Can we run as the current user instead of root (client.containers.run has parameter `user`)?
        if (not tty_output):
            container = client.containers.run(docker_config.docker_image, f'/bin/bash -lc "{command}"',
                                              remove=False, detach=True,
                                              # network_mode="host" does not work in the Mythic's environment,
                                              # because some real hostnames (e.g. including "aus5") trigger loading
                                              # of non-existing configuration files (e.g.
                                              # /fw_publish/projects/boreas/isobuild/aus5_settings.rb)
                                              volumes=volumes, **extra_kwargs)
            output = container.logs(stream=True, stdout=True, stderr=True, follow=True)
            for line in output:
                decoded_line = line.decode("utf-8", "ignore")
                logger.info(decoded_line.rstrip())
            exit_code = container.wait()["StatusCode"]
            if exit_code != 0:
                raise RuntimeError(f"Command {command} failed with exit code {exit_code}")
        else:
            # Caller has requested tty-like output. This requires some raw processing outside of the
            # Python logging prints. To support error reporting through the logging, a tail is
            # maintained. This is printed in full on error and only the last line on non-error if
            # the current logging instance is using a FileHandler.
            container = client.containers.run(docker_config.docker_image, f'/bin/bash -lc "{command}"',
                                              remove=False, detach=True, tty=True,
                                              stdout=True, stderr=True,
                                              volumes=volumes, **extra_kwargs)
            tail = collections.deque(maxlen=10)
            for chunk in container.attach(stream=True, logs=True):
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                text = chunk.decode("utf-8", "ignore")
                tail.append(text.rstrip("\r"))
            exit_code = container.wait()["StatusCode"]
            if exit_code != 0:
                logger.error(f"Container exited with status {exit_code}. "
                             f"Last {len(tail)} lines:\n{tail}")
                raise RuntimeError(f"Command {command} failed with exit code {exit_code}")
            else:
                # check if logging is using file output and log the last message if so
                has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
                if has_file_handler:
                    logger.info(tail.pop())
    finally:
        if container:
            container.remove()
        client.close()


def _ensure_writable_output_dir(cmd, output_dir):
    """Modify `cmd` to ensure that the output directory is writable for everyone after it exits."""
    return f"{cmd} ; status=$? ; chmod -R 777 '{output_dir}' ; (exit $status)"


def _dnn_compile_common(docker_config, output_dir, options, file_options=None):
    """Compile an input ONNX DNN or pre-compiled network into firmware source.

    A compiler input file should be specified in options.
    """
    output_dir_in_container = docker_config.local_path_to_container(output_dir)
    if file_options:
        options = options + list(file_options)
    # The files are created for by root in a container and need to be made accessible to the current user.
    run_command = f"/mythic/dnn_compiler -o '{output_dir_in_container}' -B {' '.join(options)}"
    run_command = _ensure_writable_output_dir(run_command, output_dir_in_container)
    logger.debug(f"Compiling DNN: {run_command}")
    run_in_docker(docker_config, run_command)


def dnn_compile(docker_config, onnx_file, output_dir, options, file_options=None):
    """Compile an input ONNX DNN into firmware source, write results to the output directory."""
    onnx_file_in_container = docker_config.local_path_to_container(onnx_file)
    options = options + ["--onnx-file", onnx_file_in_container]
    _dnn_compile_common(docker_config, output_dir, options, file_options=file_options)


def dnn_recompile(docker_config, input_l0_pb, output_dir, buffers_to_dump, options, file_options=None):
    """Compile a pre-compiled network into firmware source, write results to the output directory."""
    if not input_l0_pb.exists():
        raise FileNotFoundError(f"Cannot find pre-compiled {input_l0_pb}") from None
    final_l0_pb_in_container = docker_config.local_path_to_container(input_l0_pb)
    options = options + ["--dump-buffers", ' '.join(buffers_to_dump), "--l0-protobuf-in", final_l0_pb_in_container]
    options.remove("--acm")
    if "--instrument-buffers" not in options:
        options.append("--instrument-buffers")
    if "-X" not in options:
        options.append("-X")
    _dnn_compile_common(docker_config, output_dir, options, file_options=file_options)


def fw_compile(docker_config, fw_source_dir, output_dir, options):
    """Run the firmware compiler on input firmware source, write results to the output directory."""
    fw_source_dir_in_container = docker_config.local_path_to_container(fw_source_dir)
    output_dir_in_container = docker_config.local_path_to_container(output_dir)
    fw_do_options = [f"target={fw_source_dir_in_container}", f"dest={output_dir_in_container}"] + options
    fw_do_options_str = ",".join(f"'{option}'" for option in fw_do_options)
    run_command = f"cd /fw_publish/projects/boreas/isobuild/fw && rake fw:do[{fw_do_options_str}]"
    run_command = _ensure_writable_output_dir(run_command, output_dir_in_container)
    logger.debug("Compiling firmware...")
    run_in_docker(docker_config, run_command)
    logger.debug(f"Compilation of firmware from source {fw_source_dir} complete.")


def dnn_fw_compile(docker_config, onnx_file, fw_source_output_dir, fw_bin_output_dir, compile_options,
                   fw_compile_options, file_options=None):
    """Compile an input ONNX model to a model that is runnable on a Mythic Analog Matrix Processor (AMP)."""
    dnn_compile(docker_config, onnx_file, fw_source_output_dir, compile_options, file_options=file_options)
    if any("--amp-arch boreas" in s for s in compile_options):
        fw_compile(docker_config, fw_source_output_dir, fw_bin_output_dir, fw_compile_options)
