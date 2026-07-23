"""Debug Containers for bcm models."""

from typing import Any
from dataclasses import dataclass


@dataclass
class SimpleMMADebugContainer:
    """Data class for tracking debug values for visualization."""

    iflash_pfsr_weight: Any = None
    iflash_pfsr_bias: Any = None
    adc_noise: Any = None
    accumulator_out: Any = None


@dataclass
class ACMSignOffMMADebugContainer:
    """Data class for tracking debug values for visualization."""

    noisy_weight: Any = None
    noisy_bias: Any = None
    accumulator_out: Any = None
    adc_input: Any = None


@dataclass
class TrainingACMMMADebugContainer:
    """Data class for tracking debug values for visualization."""

    iflash_weight: Any = None
    iflash_bias: Any = None
    adc_input: Any = None
    mcd_output: Any = None
    sar_output: Any = None
    output: Any = None
