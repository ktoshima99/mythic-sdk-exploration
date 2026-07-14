from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BaseConfig:
    """Base dataclass classes for all configuration classes.

    It provides some convenience functions to load and save the data to and from dictionaries or files.
    One of the main advantages is that its intended to be used with nested configuration files.

    E.g. we have a ModelFlowConfiguration dataclass, which contains configuration dataclasses for optimization,
    quantization and compilation. If all of these configurations derive from `BaseConfig`, they can be used in
    a nested context.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        result = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            if key.startswith("_"):
                continue
            if isinstance(value, BaseConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> "BaseConfig":
        """Initialize the dataclass from values defined in data_dict.

        Args:
            data_dict (dict[str, Any]): data dict from which to initialize config

        Raises:
            ValueError: if dict contains illegal key


        Returns:
            "BaseConfig": the resulting dataclass object
        """
        result_dict = {}
        dataclass_fields = fields(cls)
        dataclass_field_names = [field.name for field in dataclass_fields]
        if any(key not in dataclass_field_names for key in data_dict.keys()):
            raise ValueError(f"Could not merge dict into dataclass: dict contains illegal keys: {data_dict}")

        for field in dataclass_fields:
            key = field.name
            field_type = field.type
            if key.startswith("_"):
                continue
            if key in data_dict:
                if isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                    result_dict[key] = field_type.from_dict(data_dict[key])
                else:
                    result_dict[key] = data_dict[key]
            else:
                raise ValueError(f"Missing key in dict for dataclass merge: {key}")
        result_object = cls(**result_dict)
        return result_object

    def save(self, path: str | Path) -> None:
        """Save the dataclass to a yaml file.

        Args:
            path(str|Path): Path to where the dataclass should be stored.

        Returns:
            None: Config is saved to path and not returned
        """
        with open(path, "w") as file:
            yaml.dump(asdict(self), file, indent=4, default_flow_style=False)

    @classmethod
    def load(cls, path: str | Path) -> "BaseConfig":
        """Load the data class from a yaml file.

        Args:
            path (str | Path): Path to yaml file from where to initialize the dataclass.

        Raises:
            ValueError: if loading failed

        Returns:
            BaseConfig: The loaded BaseConfig
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Failed loading yaml file. {path} does not exist.")

        with open(path, "r") as file:
            data = yaml.safe_load(file)

        if data is None:
            raise ValueError("Error parsing YAML file")
        return cls.from_dict(data)

    def merge_nested_cli_arguments(self, cli_arguments: dict[str, Any]) -> "BaseConfig":
        """
        Merge nested keyword arguments into this dataclass configuration.

        Updates values in a possibly nested dataclass configuration using a dictionary of
        dot-notation keys and values. Useful for command-line parameter updates to nested configs.

        Args:
            cli_arguments (dict[str, Any]): Dictionary mapping dot-notation keys to values for updating the config.
                Keys use dots to indicate nesting, e.g. "parent.child.param". Values must match
                the type of the target parameter.

        Returns:
            BaseConfig: The updated configuration object (self).

        Raises:
            ValueError: If cli_arguments is not a dict, if any key points to an invalid config path,
                or if there is a type mismatch between existing and new values.

        Examples:
            >>> class DataClass1:
            ...     argument: str = "Hello!"
            >>> class DataClass2:
            ...     dataclass1: DataClass1
            >>> dataclass2 = DataClass2()
            >>> dataclass2.merge_nested_cli_arguments({"dataclass1.argument": "Hello World!"})
        """
        if cli_arguments is None:
            return self
        elif not isinstance(cli_arguments, dict):
            raise ValueError("You need to parse a dictionary")

        for nested_key, value in cli_arguments.items():
            keys = nested_key.split(".")
            current = self

            for key in keys[:-1]:
                if not isinstance(current, BaseConfig):
                    raise ValueError(f"{nested_key} is not a valid config key")
                current = getattr(current, key)

            # Check types match
            last_key = keys[-1]
            if not hasattr(current, last_key):
                raise ValueError(f"Key {last_key} not included in {current}")

            current_value = getattr(current, last_key)
            if current_value is not None and not type(current_value) is type(value):
                raise ValueError(f"Type mismatch for key {last_key} with values {current_value} and {value}")

            setattr(current, last_key, value)

        return self

    def __str__(self) -> str:
        """Retur a prettified string representation of this (nested) dataclass."""
        data_dict = self.to_dict()
        data_str = yaml.dump(data_dict, indent=4, default_flow_style=False, sort_keys=False)
        return data_str
