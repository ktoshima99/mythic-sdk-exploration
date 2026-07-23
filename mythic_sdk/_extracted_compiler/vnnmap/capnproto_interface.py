import site
from pathlib import Path

import capnp

# Prevent capnp from automatically looking for capnp files in PYTHONPATH
capnp.remove_import_hook()
CAPNPROTO_SCHEMA_FILE_NAME = "network_schema.capnp"
ONNX_INTERFACE_CAPNP_SCHEMA_FILE_NAME = "onnx_interface.capnp"

_PACKAGE_NAME = "vnnmap"


def _find_in_site_packages(relative_path: str | Path) -> list[Path]:
    """Return candidate paths for `relative_path` across all site-packages directories."""
    candidates = [Path(d) / relative_path for d in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / relative_path)
    return candidates


def _get_schema_dir() -> Path:
    """Return the directory containing the installed capnproto schema files."""
    candidates = _find_in_site_packages(_PACKAGE_NAME)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"vnnmap package directory not found in any expected location: {[str(c) for c in candidates]}"
    )


def _get_capnproto_schema_path() -> str:
    path = _get_schema_dir() / CAPNPROTO_SCHEMA_FILE_NAME
    if not path.exists():
        raise RuntimeError(f"Could not find capnproto network schema at {path}")
    return str(path)


def _get_onnx_interface_schema_path() -> str:
    path = _get_schema_dir() / ONNX_INTERFACE_CAPNP_SCHEMA_FILE_NAME
    if not path.exists():
        raise RuntimeError(f"Could not find onnx interface capnp schema at {path}")
    return str(path)


# Load schema file
capnproto_interface = capnp.load(_get_capnproto_schema_path())

# Import generated types
Network = capnproto_interface.Network
MetaInfo = capnproto_interface.MetaInfo
Layer = capnproto_interface.Layer
Tensor = capnproto_interface.Tensor
TensorType = capnproto_interface.TensorType
TensorDataType = capnproto_interface.TensorDataType
ActivationType = capnproto_interface.ActivationType
LayerType = capnproto_interface.LayerType

# Layer specific attribute classes
ConvAttributes = capnproto_interface.ConvAttributes
MaxPoolAttributes = capnproto_interface.MaxPoolAttributes
ShortcutAttributes = capnproto_interface.ShortcutAttributes
AveragePoolAttributes = capnproto_interface.AveragePoolAttributes
FlattenAttributes = capnproto_interface.FlattenAttributes
ConcatAttributes = capnproto_interface.ConcatAttributes
ResizeAttributes = capnproto_interface.ResizeAttributes
LayerNormAttributes = capnproto_interface.LayerNormAttributes
SplitAttributes = capnproto_interface.SplitAttributes
SoftmaxAttributes = capnproto_interface.SoftmaxAttributes
SqueezeAttributes = capnproto_interface.SqueezeAttributes
ReshapeAttributes = capnproto_interface.ReshapeAttributes
TransposeAttributes = capnproto_interface.TransposeAttributes
GatherAttributes = capnproto_interface.GatherAttributes
SliceAttributes = capnproto_interface.SliceAttributes
RMSNormalizationAttributes = capnproto_interface.RMSNormalizationAttributes
RopeAttributes = capnproto_interface.RopeAttributes
ExpandAttributes = capnproto_interface.ExpandAttributes
RETRTransformationAttributes = capnproto_interface.RETRTransformationAttributes
RTRTransformationAttributes = capnproto_interface.RTRTransformationAttributes
RERTransformationAttributes = capnproto_interface.RERTransformationAttributes
ScatterAttributes = capnproto_interface.ScatterAttributes
ConvTransposeAttributes = capnproto_interface.ConvTransposeAttributes
GridSampleAttributes = capnproto_interface.GridSampleAttributes
#  TODO: Add more once defined in network_schema.capnp
