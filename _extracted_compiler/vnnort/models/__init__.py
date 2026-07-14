"""This init file is used to make the model class definitions available.

Once the models have been imported, they are registered automatically
with the VidModel base class and can then be loaded dynamally in the utils.meta_loading.load_model_class function.
"""

import importlib
import pkgutil
import sys
from types import ModuleType

from vnnort import logger

# Set the global opset version
ONNX_OPSET_VERSION = 20


def import_all_submodules(package: ModuleType) -> None:
    """
    Detect and import all submodules in the given package recursively.

    Args:
        package (ModuleType): The package to import submodules from.

    Raises:
        ImportError: If a submodule could not be imported.

    Returns:
            None: This function doesn't return anything.
    """
    package_name = package.__name__

    # Iterate over all submodules within the given package
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        try:
            # Import the module
            module = importlib.import_module(module_name)
            # If the module is a package, recursively import its submodules
            if is_pkg:
                import_all_submodules(module)

        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            raise  # reraise error to prevent silent failure


current_module = sys.modules[__name__]

import_all_submodules(current_module)
