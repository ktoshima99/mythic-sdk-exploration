from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional

from vnnort.utils.config.base_config import BaseConfig


@dataclass
class InitializationConfig(BaseConfig):
    """Configuration for the initialization of a model."""

    _input_shape: Optional[List[int]] = field(default_factory=list, repr=False, init=False)

    def __init__(self, input_shape: Optional[List[int]] = None) -> None:
        """
        Initialize an InitializationConfig instance.

        Args:
            input_shape (Optional[List[int]]): A list of exactly four integers representing the dimensions
                of the input. If None, the model will use default dimensions.
        """
        self.input_shape = input_shape

    @property
    def input_shape(self) -> Optional[List[int]]:
        """
        Get the input shape.

        Returns:
            Optional[List[int]]: The input shape, which is either None or a list of exactly four integers.
        """
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: Optional[List[int]]) -> None:
        """
        Set the input shape, ensuring it is either None or a list of exactly four integers.

        Args:
            value (Optional[List[int]]): The input shape to set.

        Raises:
            TypeError: If `value` is not None and not a list, or if any element in the list is not an integer.
            ValueError: If `value` is a list but does not contain exactly four integers.

        Returns:
            None: setter does not return anything.
        """
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("input_shape must be a list of exactly four integers or None.")
            if len(value) != 4:
                raise ValueError("input_shape must be a list of exactly four integers.")
            for idx, dim in enumerate(value):
                if not isinstance(dim, int):
                    raise TypeError(
                        f"Element at index {idx} in input_shape must be an integer; got {type(dim).__name__}."
                    )
        else:
            value = list()
        self._input_shape = value
