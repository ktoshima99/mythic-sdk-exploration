# flake8: noqa: E402

"""
Initialize the vnnort package.

The vnnort package provides a collection of functions and submodules to facilitate optimization,
quantization, and interfacing with various AI model workflows. It includes tools for enhancing
model performance through techniques like optimization and quantization, and also provides an
interface to v-NN Mapper, a program designed for mapping and managing AI tasks.

Modules:
    - optimizer: Functions and classes to apply various optimization techniques on AI models.
    - quantizer: Tools and utilities for model quantization, reducing model size and improving efficiency.
    - interface: Components to interface with the v-NN Mapper program, supporting model integration and task mapping.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("vnnort")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Initialize environment variables before anything else
from dotenv import load_dotenv

current_directory = current_file = Path(__file__).parent
dot_env_path = current_directory / "../../.env"
if dot_env_path.exists():  # This file is optional
    load_dotenv(dotenv_path=dot_env_path, override=False)  # Make sure file does not override existing vars
default_env_path = current_directory / "default_env"
load_dotenv(dotenv_path=default_env_path, override=False)  # Make sure files does not override existing vars


import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from dotenv import load_dotenv

# Ensure randomization is reproducable
torch.manual_seed(42)
np.random.seed(42)

# --- Logging --- #
logger = logging.getLogger(__name__)
envs = os.environ


# Configure the logging settings
def configure_logging(log_level: str = "INFO", log_file_path: str | Path | None = None) -> None:
    """Configure the logging settings.

    Args:
        log_level (str, optional): The logging level. Defaults to "INFO".
            Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        log_file_path (str | Path | None, optional): Path to the log file. If provided,
            logs will be written to this file in addition to stdout. Defaults to None.

    Returns:
        None: This function doesn't return anything.
    """
    logging.basicConfig(
        level=log_level,  # Set the logging level to INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Add file handler
    if log_file_path is not None:
        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)


def set_log_level(log_level_name: str) -> None:
    """Set the logging level for the logger and the VNNORT_LOG_LEVEL environment variable.

    Args:
        log_level_name (str): The name of the logging level to set (e.g., 'INFO', 'DEBUG', etc.).

    Raises:
        ValueError: If the `log_level_name` is not a valid logging level.

    Returns:
        None: This function doesn't return anything.
    """
    if log_level_name not in logging._nameToLevel:
        logger.error(f"Invalid log level '{log_level_name}'. Choose one of: {list(logging._nameToLevel.keys())}")
        raise ValueError(f"Invalid log level: {log_level_name}. Valid options: {list(logging._nameToLevel.keys())}")

    # Set the logger's level
    log_level = logging._nameToLevel[log_level_name]
    logger.setLevel(log_level)

    # Set the VNNORT_LOG_LEVEL environment variable
    os.environ["VNNORT_LOG_LEVEL"] = log_level_name

    logger.debug(f"Logging level set to {log_level_name}")


# Function to set environment variable only if it doesn't exist
def _set_env_variable_if_not_exists(key: str, default_value: str) -> None:
    """Set the environment variable 'key' to 'default_value' only if it doesn't exist.

    Args:
        key (str): The name of the environment variable to set.
        default_value (str): The default value to set if the environment variable is not already set.

    Returns:
        None: This function doesn't return anything.
    """
    if os.getenv(key) is None:
        os.environ[key] = default_value
        logger.debug(f"Setting {key} to {default_value}")
    else:
        logger.debug(f"{key} already exists with value {os.getenv(key)}")


def get_env_variable(key: str) -> str:
    """
    Retrieve the value of the environment variable 'key'. If the environment variable is not set, raises an EnvironmentError.

    Args:
        key (str): The name of the environment variable to retrieve.

    Returns:
        str: The value of the environment variable.

    Raises:
        EnvironmentError: If the environment variable is not set.
    """
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(f"Environment variable '{key}' is not set.")

    return value


# ORT Logging
ORT_LOG_LEVELS = {
    "VERBOSE": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "FATAL": 4,
}
log_level_str = os.environ.get("VNNORT_LOG_LEVEL").upper()
log_level = ORT_LOG_LEVELS.get(log_level_str, ORT_LOG_LEVELS["WARNING"])
ort.set_default_logger_severity(log_level)

# Override the onnx Model Proto __repr__ method
# Debugging python code with ModelProto objects present is extremely slow, because __repr__ is too large.
if os.getenv("VNNORT_ONNX_MODELPROTO_OVERWRITE"):
    import onnx

    setattr(onnx.ModelProto, "__repr__", lambda x: "ModelProto")
    setattr(onnx.NodeProto, "__repr__", lambda x: "NodeProto")
    setattr(onnx.TensorProto, "__repr__", lambda x: "TensorProto")
    setattr(onnx.GraphProto, "__repr__", lambda x: "GraphProto")
    setattr(onnx.AttributeProto, "__repr__", lambda x: "AttributeProto")
    setattr(onnx.TypeProto, "__repr__", lambda x: "TypeProto")

# Remove future and deprecation warnings originating from the internals of onnxscript (we can't do anything about it)
warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="onnxscript")
