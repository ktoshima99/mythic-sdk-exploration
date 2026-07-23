from dataclasses import dataclass, field

from vnnmap.compilation_config import CompilationConfig
from vnnort.models.initialization_config import InitializationConfig
from vnnort.optimizer.optimization_config import OptimizationConfig
from vnnort.quantizer.quantization_config import QuantizationConfig
from vnnort.utils.config.base_config import BaseConfig


@dataclass
class ModelFlowConfig(BaseConfig):
    """Wrapper configuration object for all configuration options involved in the videantis flow."""

    model_name: str
    initialization_config: InitializationConfig = field(default_factory=InitializationConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
