from dataclasses import dataclass
from dataclasses import field

from vnnort.utils.config.base_config import BaseConfig


@dataclass
class OptimizationConfig(BaseConfig):
    """This class contains the configuration for the optimization process."""

    allowed_unopt_nodes: dict[str, int] = field(default_factory=dict)
