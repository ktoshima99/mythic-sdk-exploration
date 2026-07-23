import logging
from typing import TYPE_CHECKING, Optional

import onnx
from onnx import ModelProto

from vnnort.optimizer.optimization_config import OptimizationConfig
from vnnort.optimizer.pattern_detection import pattern_match
from vnnort.optimizer.standardize_onnx_model import standardize_naming
from vnnort.optimizer.utils import (
    fuse_muls,
    fuse_reshape_modes,
    infer_shapes_runtime,
    move_static_cons_to_wgts,
    remove_unused_nodes,
    remove_unused_wgts,
    update_onnx_opset_version,
)
from vnnort.utils.onnx_utils.optimize_execution_order import optimize_execution_order

if TYPE_CHECKING:
    from vnnort.models.vid_model import VidModel

logger = logging.getLogger(__name__)


def run_optimization(model: "VidModel", config: Optional[OptimizationConfig] = None) -> ModelProto:  # noqa: F821
    """
    Optimize a VidModel instance based on the specified configuration.

    Args:
        model (VidModel): The model to optimize.
        config (Optional[OptimizationConfig]): Configuration settings defining the optimization techniques and parameters.

    Returns:
        ModelProto: The optimized model after applying the specified optimizations.
    """
    # TODO: Implement actual optimization code based on config settings
    optimized_model = model._model_repr

    return optimized_model


def optimization_pipeline(vid_model: "VidModel") -> onnx.ModelProto:
    """
    Perform a series of optimization steps on an ONNX model.

    This pipeline executes several preprocessing and optimization steps, including shape inference,
    moving constants, removing unused nodes and weights, pattern matching, and fusing activations.
    The optimization can be customized via flags, and the model can be split into sub-models if specified.

    Args:
        vid_model ("VidModel"): The VidModel to optimize.

    Returns:
        onnx.ModelProto: The optimized ONNX model
    """
    model = vid_model._model_repr

    # Extract test data to be used in various places of the pipeline
    # Make sure to fetch data, which is really different (e.g. nearby Squad dataset entries are very similar)
    ds = vid_model.load_default_dataset()
    data1, data2 = vid_model.preprocess(ds[0]), vid_model.preprocess(ds[51])

    # Step through each function with progress updates
    logger.info("Inferring shapes stage 1")
    # infer_shapes_runtime(model, test_inputs1)

    logger.info("Moving static connections")
    model = move_static_cons_to_wgts(model, [data1, data2])  # type: ignore

    logger.info("Removing unused weights")
    model = remove_unused_wgts(model)  # type: ignore

    logger.info("Removing unused nodes")
    model = remove_unused_nodes(model, None, data1, verbose=False)

    logger.info("Converting and optimizing")
    model = update_onnx_opset_version(model)

    logger.info("Inferring shapes stage 2")
    infer_shapes_runtime(model, data1)

    logger.info("Pattern matching")
    model = pattern_match(model, data1, data2)
    del model.graph.value_info[:]

    logger.info("Fusing multiplications")
    model = fuse_muls(model)

    logger.info("Moving static connections after pattern")
    model = move_static_cons_to_wgts(model, [data1, data2])  # type: ignore

    logger.info("Removing unused weights")
    model = remove_unused_wgts(model)  # type: ignore

    logger.info("Removing unused nodes")
    model = remove_unused_nodes(model, None, data1)

    logger.info("Fusing reshape modes")
    model = fuse_reshape_modes(model)

    logger.info("Calling optimize hook")
    model = vid_model.optimize_hook(model)

    logger.info("Optimize node order")
    model = optimize_execution_order(model)

    logger.info("Standardizing node and tensor names.")
    model = standardize_naming(model)

    logger.info("Final shape inference")
    infer_shapes_runtime(model, data1)

    # Remove this for now to not mess with onnx node order
    # from vnnort.optimizer.standardize_onnx_model import standardize_onnx_model
    # logger.info("Standardizing ONNX model")
    # model = standardize_onnx_model(model)
    return model  # type: ignore
