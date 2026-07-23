"""This init file is used to make the dataset class definitions available.

Once the datasets have been imported, they are registered automatically
with the DatasetBase class and can then be loaded dynamally in the load_dataset_class function.
"""

import importlib
import importlib.util
import pkgutil

import vnnort.data.datasets as datasets_package

package_name = datasets_package.__name__
for _, module_name, is_pkg in pkgutil.iter_modules(datasets_package.__path__):
    full_module_name = f"{package_name}.{module_name}"
    importlib.import_module(full_module_name)
