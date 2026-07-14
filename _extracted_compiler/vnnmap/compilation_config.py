from dataclasses import dataclass

from vnnort.utils.config.base_config import BaseConfig


@dataclass
class CompilationConfig(BaseConfig):
    """This class contains the configuration for the compilation process."""

    n_mps: int = 1
    """Number of MP cores to be used. Defaults to 1"""
