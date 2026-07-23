import copy
import os
import pathlib
import tempfile
from collections import Counter
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnxruntime
import onnxscript
from numpy.typing import NDArray
from onnx import ModelProto, load_external_data_for_model
from onnx.helper import make_attribute, make_tensor_value_info
from onnx.numpy_helper import from_array, to_array
from onnx.utils import Extractor

from vnnort import logger
from vnnort.optimizer.patterns import vidConvShortcutPostFuse  # type: ignore
from vnnort.optimizer.patterns import vidConvShortcutPreFuse  # type: ignore
from vnnort.utils.onnx_utils.hooked_inference import HookedOnnxInferenceSession
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field
from vnnort.utils.onnx_utils.onnx_hooks import ONNXTensorCompareHook
from vnnort.utils.onnx_utils.unique_initializers import make_initializers_unique


def standardize_input_names(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Standardizes the input names of the ONNX model by renaming them to a consistent format ("input:X").

    Args:
        model (onnx.ModelProto): The ONNX model whose input names need to be standardized.

    Returns:
        onnx.ModelProto: The modified ONNX model with standardized input names.
    """
    replace_dict = {}
    for i, inp in enumerate(model.graph.input):
        replace_dict[inp.name] = "input:" + str(i)
        new_input = model.graph.input[i]
        new_input.name = replace_dict[new_input.name]
        del model.graph.input[i]
        model.graph.input.insert(i, new_input)

    for i, node in enumerate(model.graph.node):
        for i_inp, n_inp in enumerate(node.input):
            if n_inp in replace_dict.keys():
                model.graph.node[i].input[i_inp] = replace_dict[n_inp]

    # In case constants are fed into multiple nodes:
    make_initializers_unique(model)

    return model


def move_static_cons_to_wgts(  # noqa: C901 ---ignores "too complex"-error
    model_test: onnx.ModelProto,
    test_inputs: list[dict[str, NDArray[np.float32]]],
    verbose: bool = False,
    return_no_rem_nodes: bool = False,
) -> Union[onnx.ModelProto, Tuple[onnx.ModelProto, int]]:
    """
    Move static constants in the ONNX model to model weights by analyzing the outputs from test inputs.

    This function identifies static outputs (those that remain constant across multiple test inputs) and moves
    them to model weights. It also removes redundant nodes from the model and optionally returns the number of
    nodes removed.

    Args:
        model_test (onnx.ModelProto): The ONNX model to modify.
        test_inputs (list[dict[str, NDArray[np.float32]]]): A list of test inputs to evaluate the model's outputs.
        verbose (bool): If True, prints information about removed nodes. Defaults to False.
        return_no_rem_nodes (bool): If True, returns the number of nodes removed along with the model.

    Returns:
        Union[onnx.ModelProto, Tuple[onnx.ModelProto, int]]: The modified ONNX model with static constants moved to weights and the
        number of nodes removed, if `return_no_rem_nodes` is True.

    Raises:
        RuntimeError: If weight to be removed could not be found in static values dict.
    """
    from vnnort.optimizer.pattern_detection import get_nodes_by_output

    # Register all outputs as hooks
    output_hooks = {}
    for node in model_test.graph.node:
        for output in node.output:
            output_hooks[output] = ONNXTensorCompareHook()

    with HookedOnnxInferenceSession.create(model_test, output_hooks, n_workers=8) as sess:
        for current_input in test_inputs:
            sess.run(current_input)

    static_keys: list[str] = []
    static_values: dict[str, Any] = dict()
    for output_name, hook in output_hooks.items():
        if hook.output_is_static():
            static_keys.append(output_name)
            static_values[output_name] = hook.output

    model = model_test
    indices = []
    for key in static_keys:
        out_indices, out_nodes = get_nodes_by_output(model, key)
        index = out_indices[0]
        node = out_nodes[0]
        indices.append(index)
    removed_nodes = 0
    for i in sorted(list(set(indices)), reverse=True):

        node = model.graph.node[i]
        output_con = node.output[0]
        if node.op_type == "TopK":
            continue
        if len(node.output) != 1:
            continue
        if output_con in [out.name for out in model.graph.output]:
            continue

        wgt = static_values[output_con]
        if wgt is None:
            raise RuntimeError("wgt not found in static values!")
        add_wgt(model, wgt, output_con)
        logger.debug(
            f"Static connection found: removing node: {model.graph.node[i].name}  {model.graph.node[i].op_type}. New Weight: Name: {output_con}, Shape: {wgt.shape}, dtype: {wgt.dtype}"
        )
        removed_nodes += 1
        del model.graph.node[i]
    if return_no_rem_nodes:
        return model, removed_nodes
    else:
        return model


def update_onnx_opset_version(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert the ONNX model to the toolbox's target ONNX opset version.

    Args:
        model (onnx.ModelProto): The ONNX model to be converted and optimized.
    Returns:
        onnx.ModelProto: The converted ONNX model.
    Raises:
        RuntimeError: If the ModelProto is malformed and does not define an OPSet version.
    """
    from vnnort.models import ONNX_OPSET_VERSION

    # Check if it may already be at correct version
    current_opset_version = None
    for opset in model.opset_import:
        if opset.domain in ["", "ai.onnx"]:
            current_opset_version = opset.version
    if current_opset_version is None:
        msg = "Could not find current model opset version"
        raise RuntimeError(msg)
    if current_opset_version == ONNX_OPSET_VERSION:
        logger.debug("Model is already at correct opset version, skipping update.")
        return model

    model_directory = _get_onnx_meta_field(model, "model_directory")
    if model_directory is not None:
        onnx.load_external_data_for_model(model, str(model_directory))
    model_opt = onnx.version_converter.convert_version(model, ONNX_OPSET_VERSION)
    model_opt.functions.extend(model.functions)
    return model_opt


def replace_node_inputs(model: ModelProto, old_input: str, new_input: str) -> ModelProto:
    """
    Replace an input name throughout the model with a new input name.

    Args:
        model (ModelProto): The ONNX model
        old_input (str): Original input name to replace
        new_input (str): New input name to use

    Returns:
        ModelProto: Updated model with replaced input name

    Raises:
        ValueError: If input names not found in model
        TypeError: If model is not ModelProto or valid path
    """
    if not isinstance(model, ModelProto):
        raise TypeError("Model must be ModelProto")

    # Check if old input exists in model inputs
    input_found = False
    for node in model.graph.node:
        for input in node.input:
            if input == old_input:
                input_found = True
                break

    if not input_found:
        raise ValueError(f"Input name '{old_input}' not found in model inputs")

    # Update all nodes that reference this input
    for node in model.graph.node:
        # Replace in input list
        for i, input_name in enumerate(node.input):
            if input_name == old_input:
                node.input[i] = new_input

    return model


def _get_no_of_static_inputs(model: onnx.ModelProto, node: onnx.NodeProto) -> int:
    inits = [init.name for init in model.graph.initializer]

    static_inputs = [inp for inp in node.input if inp in inits]
    return len(static_inputs)


def remove_unused_wgts(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove unused weights from the ONNX model by identifying and deleting initializers that are not used by any node.

    Args:
        model (onnx.ModelProto): The ONNX model from which to remove unused weights.

    Returns:
        onnx.ModelProto: The ONNX model with unused weights removed.
    """
    all_wgts = [(i, init.name) for i, init in enumerate(model.graph.initializer)]
    all_inputs = [input for node in model.graph.node for input in node.input]
    to_remove = []
    for i, wgt in all_wgts:
        if wgt not in all_inputs:
            to_remove.append(i)
    to_remove.sort(reverse=True)
    for rem in to_remove:
        del model.graph.initializer[rem]
    return model


def remove_unused_nodes(  # noqa: C901 ---ignores "too complex"-error
    model: Union[str, onnx.ModelProto],
    output_model_path: Optional[str] = None,
    test_inputs1: Any = None,
    verbose: bool = False,
) -> onnx.ModelProto:
    """Remove unused nodes from an ONNX model, optimizing its structure.

    Args:
        model (Union[str, onnx.ModelProto]): The ONNX model or file path to the model to process.
        output_model_path (Optional[str], optional): File path to save the modified model. Defaults to None.
        test_inputs1 (Any): List of input names for the model. Defaults to None.
        verbose (bool): Whether to enable detailed logging. Defaults to False.

    Returns:
        onnx.ModelProto: The optimized ONNX model with unused nodes removed.
    """
    if isinstance(model, str):
        model = onnx.load(model)

    # Combine reshape removal and general node removal
    def remove_unused_reshapes(model: onnx.ModelProto) -> onnx.ModelProto:
        nodes_to_remove = []
        for i, node in enumerate(model.graph.node):
            if node.op_type == "Reshape":
                val_info = get_value_info_by_name_val_info(model.graph.value_info, node.input[0])
                input_shape = val_info[1]

                val_info = get_value_info_by_name_val_info(model.graph.value_info, node.output[0])
                output_shape = val_info[1]

                if input_shape == output_shape and output_shape is not None:
                    nodes_to_remove.append((i, node))
        for i, node in reversed(nodes_to_remove):
            logger.info(f"Removing unused Reshape node: {node.name} at index {i}")
            model = remove_node(model, node.name)
        return model

    def remove_redundant_ops(model: onnx.ModelProto) -> onnx.ModelProto:
        """Remove redundant operations from ONNX model.

        - Addition by zero
        - Multiplication by one

        Args:
            model (onnx.ModelProto): The ONNX model to process.
        Returns:
            onnx.ModelProto: The optimized model with redundant operations removed.
        """
        # Collect initializer names for faster lookup
        inits = {init.name for init in model.graph.initializer}

        # Nodes to remove
        nodes_to_remove = []
        model_directory = _get_onnx_meta_field(model, "model_directory")
        load_external_data_for_model(model, model_directory)
        # Single pass through all nodes
        for node in model.graph.node:
            if len(node.input) == 2:  # Only check binary operations
                for input_name in node.input:
                    if input_name in inits:
                        # Fetch weight associated with the input
                        wgt = get_wgt_by_name(model, input_name)

                        # For numpy arrays use np.all(), for floats use direct comparison
                        is_zero = np.all(wgt == 0) if isinstance(wgt, np.ndarray) else wgt == 0
                        is_one = np.all(wgt == 1) if isinstance(wgt, np.ndarray) else wgt == 1

                        # Check for addition by zero
                        if node.op_type == "Add" and is_zero:
                            nodes_to_remove.append(node.name)
                            logger.info(f"Found zero-add node: {node.name}")
                            break

                        # Check for multiplication by one
                        if node.op_type == "Mul" and is_one:
                            nodes_to_remove.append(node.name)
                            logger.info(f"Found one-mul node: {node.name}")
                            break

        # Remove all identified nodes in reverse order
        for node in reversed(model.graph.node):
            if node.name in nodes_to_remove:
                logger.debug(f"Removing node: {node.name}")
                model = remove_node(model, node.name)

        return model

    model = remove_unused_reshapes(model)
    model = remove_redundant_ops(model)

    # Main node removal loop
    node_removed = True
    while node_removed:
        node_removed = False

        # Remove nodes with no following nodes
        for _, node in enumerate(model.graph.node):
            if len(node.output) == 1:
                nxt_nodes = _get_nxt_nodes(model, node.name)[0]
                if nxt_nodes is None:
                    continue
                if len(nxt_nodes) == 0:
                    if (len([out for out in node.output if out in [m_out.name for m_out in model.graph.output]])) == 0:
                        try:
                            model = remove_node(model, node.name)
                            logger.debug(f"Removing node (no following nodes): {node.name}, {node.op_type}")
                            node_removed = True
                        except RuntimeError as e:
                            logger.error(f"Node could not be removed: {e}")
            else:
                nxt_node = False
                for out in node.output:
                    from vnnort.optimizer.pattern_detection import get_nodes_by_input

                    ids, nodes = get_nodes_by_input(model, out)
                    if len(ids) > 0 or out in [model_out.name for model_out in model.graph.output]:
                        nxt_node = True
                if not nxt_node:
                    try:
                        model = remove_node(model, node.name)
                        logger.debug(f"Removing node (no following nodes): {node.name}, {node.op_type}")
                        node_removed = True
                    except RuntimeError as e:
                        logger.error(f"Node could not be removed: {e}")

    if output_model_path is not None:
        onnx.save(model, output_model_path)
        logger.info(f"Model saved to {output_model_path}")

    return model


def check_input_node_valid(input_name: str, model: onnx.ModelProto) -> bool:
    """
    Check if the input node is valid in the ONNX model.

    Args:
        input_name (str): The name of the input node to be checked.
        model (onnx.ModelProto): The ONNX model to check in.

    Returns:
        bool: True if the input node is valid, False otherwise.
    """
    for init in model.graph.initializer:
        if input_name in (init.name):
            return True
    for node in model.graph.node:
        if input_name in (node.input):
            return True
    return False


def onnx_layer_output(  # noqa: C901 ---ignores "too complex"-error
    model: onnx.ModelProto, inputs: list[dict[str, NDArray[Any]]]
) -> list[dict[str, NDArray[Any]]]:
    """
    Execute an ONNX model with all intermediate outputs included, returning the outputs for each layer.

    This function modifies the given ONNX model to include all intermediate outputs (i.e., outputs from each
    node in the graph) and then runs the model using ONNX Runtime with the provided inputs. The function returns
    the output values for all layers in the model.

    Args:
        model (onnx.ModelProto): The ONNX model to modify and execute.
        inputs (list[dict[str, NDArray[Any]]]): A list of dummy inputs to feed into the ONNX model.

    Returns:
        list[dict[str, NDArray[Any]]]: A list of ordered dictionary mapping each layer's output name to its
            corresponding value.

    Raises:
        ValueError: If input data format is not correct
    """
    # Check that types are correct
    if not isinstance(model, onnx.ModelProto):
        msg = "model needs to be of type onnx.ModelProto"
        raise ValueError(msg)

    if not isinstance(inputs, list) or not all(isinstance(entry, dict) for entry in inputs):
        msg = "Input data must be list of input dicts"
        raise ValueError(msg)

    # Create copy of model, since we modify the meta data
    model = copy.deepcopy(model)
    with tempfile.TemporaryDirectory() as temp_dir:
        # For VidModels, we need to save the tmp model into the model_directory, where the corresponding weights are
        model_directory = _get_onnx_meta_field(model, "model_directory")
        if model_directory is None:
            model_directory = temp_dir

        model_path = pathlib.Path(model_directory) / "tmp_model.onnx"
        original_outputs = [o.name for o in model.graph.output]

        for node in model.graph.node:
            for output in node.output:
                if output not in original_outputs:
                    model.graph.output.extend([onnx.ValueInfoProto(name=output)])

        # Save the modified model in the temp_dir or model_directory
        if model_directory == temp_dir:
            onnx.save(model, model_path, save_as_external_data=True, location="weights.dat")
        else:
            # If we already have vid model, the weights are already there, no need to save them again
            onnx.save(model, model_path)

        # Create a new inference session using the modified model file
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        ort_session = onnxruntime.InferenceSession(model_path, so)

        # Delete tmp model. The weights do not need to be deleted, because they are cleaned up by tempfile
        os.remove(model_path)

        # Prepare inputs and execute the model
        results = []
        for model_inputs in inputs:

            # Run model and collect outputs
            all_outputs = [o.name for o in model.graph.output]
            ort_outs = ort_session.run(None, model_inputs)
            ort_outs = dict(zip(all_outputs, ort_outs))
            results.append(ort_outs)

        return results


def add_wgt(model: onnx.ModelProto, wgt: NDArray[Any], name: str) -> onnx.ModelProto:
    """
    Add a weight to the model's graph initializer list.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.
        wgt (NDArray[Any]): The weight data as a NumPy array.
        name (str): The name for the new weight. Raises an error if a weight with this name already exists and differs.

    Returns:
        onnx.ModelProto: The updated ONNX model with the new weight added.

    Raises:
        ValueError: If a weight with the specified name already exists and the data differs.
    """
    from onnx.numpy_helper import from_array, to_array

    existing_names = {init.name: init for init in model.graph.initializer}

    # Check if weight with the same name exists
    if name in existing_names:
        # Convert existing and new weight data to arrays for comparison
        existing_wgt_array = to_array(existing_names[name])
        new_wgt_array = wgt

        # Compare arrays; raise error if they differ
        if not np.array_equal(existing_wgt_array, new_wgt_array):
            raise ValueError(f"A weight with the name '{name}' already exists and differs from the provided weight.")
        else:
            return model

    # Convert weight data to ONNX initializer and append
    new_wgt_initializer = from_array(wgt, name)
    model.graph.initializer.append(new_wgt_initializer)
    return model


def get_wgt_by_name(model: onnx.ModelProto, name: str) -> NDArray[Any]:
    """
    Retrieve a weight tensor from an ONNX model by its name.

    Args:
        model (onnx.ModelProto): The ONNX model from which to retrieve the weight.
        name (str): The name of the weight tensor to retrieve.

    Returns:
        NDArray[Any]: The weight tensor as a NumPy array if found, otherwise None.
    """
    inits = [init.name for init in model.graph.initializer]
    idx = inits.index(name)

    # Model is in external data mode, so the weights need to be loaded first
    # We want to keep it light, so load the weight into temporary object
    onnx_wgt = deepcopy(model.graph.initializer[idx])

    # load_external_data_for_tensor(onnx_wgt, base_dir=model_dir)
    ret_val = to_array(onnx_wgt)
    # Check if ret_val is a 0-dimensional numpy array
    if isinstance(ret_val, np.ndarray) and ret_val.ndim == 0:
        ret_val = ret_val.item()  # Get the scalar value
    return ret_val


def replace_wgt(model: onnx.ModelProto, wgt: NDArray[Any], name: str) -> onnx.ModelProto:
    """
    Replace a specific weight in an ONNX model's initializer list by name.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.
        wgt (NDArray[Any]): The new weight data as a NumPy array.
        name (str): The name of the weight to replace.

    Returns:
        onnx.ModelProto: The updated ONNX model with the new weight.

    Raises:
        ValueError: If no weight with the specified name exists in the model.
    """
    inits = [init.name for init in model.graph.initializer]
    try:
        idx = inits.index(name)
        del model.graph.initializer[idx]
        model.graph.initializer.insert(idx, from_array(wgt, name))
    except ValueError:
        raise ValueError(f"No weight named '{name}' found in the model's initializers.")

    return model


def get_value_info_by_name_val_info(
    val_info_container: List[onnx.ValueInfoProto], name: str
) -> Tuple[Optional[np.dtype[np.generic]], Optional[Tuple[int, ...]]]:
    """
    Retrieve the data type and shape information for a given value name from a container of ONNX ValueInfoProto.

    This function searches through a list of ONNX ValueInfoProto objects to find the value with the specified name.
    If found, it extracts the associated tensor's data type and shape information.

    Args:
        val_info_container (List[onnx.ValueInfoProto]): A list of ValueInfoProto objects that contain metadata about the model's values.
        name (str): The name of the value for which to retrieve the type and shape information.

    Returns:
        Tuple[Optional[np.dtype[np.generic]], Optional[Tuple[int, ...]]]:
            - If the value is found, a tuple containing the tensor data type (as an ONNX TypeProto) and its shape (as a tuple of integers).
            - If the value is not found, returns (None, None).
    """
    from onnx.helper import tensor_dtype_to_np_dtype

    val_infos = [val_info.name for val_info in val_info_container]

    try:
        idx = val_infos.index(name)
    except ValueError:
        return None, None

    val_info = val_info_container[idx]
    shape = []

    for dim in val_info.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)

    shape = tuple(shape)
    tensor_type = tensor_dtype_to_np_dtype(val_info.type.tensor_type.elem_type)

    return tensor_type, shape


def extract_model_vid(
    input_model: onnx.ModelProto,
    input_names: List[str],
    output_names: List[str],
    check_model: bool = True,
) -> onnx.ModelProto:
    """Extract sub-model from an ONNX model.

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the boundary of sub-model,
    which is defined by the input and output tensors, should not cut through the
    subgraph that is connected to the main graph as attributes of these operators.

    Args:
        input_model (onnx.ModelProto): The original ONNX model.
        input_names (List[str]): the input names to start extraction
        output_names (List[str]): the output names to stop extraction
        check_model (bool): whether to check the model after extraction

    Returns:
        onnx.ModelProto: the extracted model
    """
    model = input_model
    if check_model:
        onnx.checker.check_model(model)

    model = input_model

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)
    # if check_model:
    # onnx.checker.check_model(extracted)
    return extracted


def split_model(
    input_model: onnx.ModelProto,
    splits: List[List[str]],
    dummy_inputs: dict[str, NDArray[Any]],
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    check_model: bool = False,
    remove_unused_wgts_: bool = True,
) -> List[onnx.ModelProto]:
    """
    Split the given model into multiple sub-models based on provided input and output names.

    Args:
        input_model (onnx.ModelProto): The input model to be split.
        splits (List[List[str]]): List of splits, where each split is a list of intermediate output names.
        dummy_inputs (dict[str, NDArray[Any]]): Dummy inputs required for model inference.
        input_names (Optional[List[str]], optional): List of input names for the model. If None, defaults to all inputs of the model. Defaults to None.
        output_names (Optional[List[str]], optional): List of output names for the final model. If None, defaults to all outputs of the model. Defaults to None.
        check_model (bool, optional): Flag to check the model validity after splitting. Defaults to False.
        remove_unused_wgts_ (bool, optional): Whether to remove unused weights from the model. Defaults to True.

    Returns:
        List[onnx.ModelProto]: A list of split models, including a post-processing model at the end.

    """
    if input_names is None:
        input_names = [input.name for input in input_model.graph.input]

    if output_names is None:
        output_names = [output.name for output in input_model.graph.output]

    split_models = []
    for split in splits:
        cur_split = deepcopy(split)
        for i, con in enumerate(reversed(split)):
            if con in input_names:
                cur_split.remove(con)
                in_nodes_input = False
                for node in input_model.graph.node:
                    if node.op_type not in ["preprocessing", "postprocessing"]:
                        if con in node.input:
                            in_nodes_input = True
                if not in_nodes_input:
                    input_names.remove(con)
        split_model = extract_from_input_output_names_vid(
            input_model,
            input_names,
            cur_split,
            dummy_inputs,
            remove_unused_wgts_=remove_unused_wgts_,
            check_model=check_model,
        )
        input_names = split
        split_models.append(split_model)

    input_names_post = split
    postprocessing = extract_from_input_output_names_vid(
        input_model,
        input_names_post,
        output_names,
        dummy_inputs,
        remove_unused_wgts_=remove_unused_wgts_,
        check_model=check_model,
    )
    return split_models + [postprocessing]


def remove_unused_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove unused inputs from an ONNX model by directly modifying the input model.

    An input is considered unused if it's not referenced by any node in the graph.

    Args:
        model (onnx.ModelProto): Input ONNX model

    Returns:
        onnx.ModelProto: Modified ONNX model with unused inputs removed
    """
    input_names_to_idx = {input.name: idx for idx, input in enumerate(model.graph.input)}

    used_inputs = set()
    for node in model.graph.node:
        for input_name in node.input:
            if input_name in input_names_to_idx:
                used_inputs.add(input_name)

    unused_indices = sorted([idx for name, idx in input_names_to_idx.items() if name not in used_inputs], reverse=True)

    for idx in unused_indices:
        del model.graph.input[idx]

    onnx.checker.check_model(model)

    return model


def extract_from_input_output_names_vid(
    input_model: onnx.ModelProto,
    input_names: List[str],
    output_names: List[str],
    model_inputs: dict[str, NDArray[Any]],
    check_model: bool = True,
    remove_unused_wgts_: bool = True,
) -> onnx.ModelProto:
    """
    Extract a sub-model from the input ONNX model based on specified input and output names.

    Args:
        input_model (onnx.ModelProto): The original ONNX model from which to extract the sub-model.
        input_names (List[str]): List of input names for the extracted model.
        output_names (List[str]): List of output names for the extracted model.
        model_inputs (dict[str, NDArray[Any]]): List of numpy model inputs.
        check_model (bool, optional): Whether to check the extracted model for validity. Defaults to True.
        remove_unused_wgts_ (bool, optional): Whether to remove unused weights from the model. Defaults to True.

    Returns:
        onnx.ModelProto: The extracted ONNX sub-model.
    """
    model = deepcopy(input_model)

    # Obtain the outputs of the layers using the provided model inputs
    outputs = onnx_layer_output(model, [model_inputs])[0]
    names = output_names

    # Create tensors for the output names
    tensors = [onnx.numpy_helper.from_array(outputs[name], name=name) for name in names]

    try:
        # Create tensors for the input names
        tensors2 = [onnx.numpy_helper.from_array(outputs[name], name=name) for name in input_names]
        model.graph.value_info.extend(
            [
                make_tensor_value_info(name=tensor.name, elem_type=tensor.data_type, shape=tensor.dims)
                for tensor in tensors2
            ]
        )
    except Exception:
        pass

    # Extend the graph with the output and value info
    model.graph.output.extend(
        [make_tensor_value_info(name=tensor.name, elem_type=tensor.data_type, shape=tensor.dims) for tensor in tensors]
    )
    model.graph.value_info.extend(
        [make_tensor_value_info(name=tensor.name, elem_type=tensor.data_type, shape=tensor.dims) for tensor in tensors]
    )

    # Save old initializers and extract the model
    old_initializers = model.graph.initializer
    infer_shapes_runtime(model, model_inputs)
    output_model = extract_model_vid(model, input_names, names, check_model=False)
    model = output_model

    # Restore the initializers and remove unused weights if needed
    del model.graph.initializer[:]
    model.graph.initializer.extend(old_initializers)

    if remove_unused_wgts_:
        model = remove_unused_wgts(model)

    # Optionally check the model for validity
    if check_model:
        onnx.checker.check_model(model)

    return model


def infer_shapes_runtime(model: onnx.ModelProto, inputs: dict[str, NDArray[Any]], only_types: bool = False) -> None:
    """
    Infer the shapes or types of model outputs at runtime based on the given inputs and update the model's \
    graph value information accordingly.

    This function is useful for dynamically updating a model with output shapes or types that are not \
    statically specified.

    Args:
        model (onnx.ModelProto): The ONNX model to infer shapes or types for.
        inputs (dict[str, NDArray[Any]]): The inputs to the model, used to perform runtime inference.
        only_types (bool): A boolean flag indicating whether to only update the types without shapes.

    Returns:
        None: The function updates the model shapes in place.
    """
    del model.graph.value_info[:]
    outputs = onnx_layer_output(model, [inputs])[0]
    # Filter output names that are not already in value_info
    names = [name for name in outputs.keys()]

    # Convert outputs to tensor value info objects
    tensors = [
        onnx.numpy_helper.from_array(outputs[name], name=name)
        for name in names
        if isinstance(outputs[name], np.ndarray)
    ]
    # Extend the model's graph value_info with new tensor value info, optionally only including types
    if only_types:
        model.graph.value_info.extend(
            [make_tensor_value_info(name=tensor.name, elem_type=tensor.data_type, shape=None) for tensor in tensors]
        )
    else:
        model.graph.value_info.extend(
            [
                make_tensor_value_info(name=tensor.name, elem_type=tensor.data_type, shape=tensor.dims)
                for tensor in tensors
            ]
        )
    # For some reason onnx keeps track of graph output shapes somewhere else....
    for output in model.graph.output:
        output_tensor = outputs[output.name]
        output.type.tensor_type.shape.ClearField("dim")
        # Set the new shape
        for dim_size in output_tensor.shape:
            dim = output.type.tensor_type.shape.dim.add()
            dim.dim_value = dim_size


def replace_duplicate_node_names(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Ensure that all nodes in the given ONNX model have unique names. If a node name is empty \
    or duplicated, it assigns a new name based on the node's operation type followed by an \
    incrementing counter to ensure uniqueness.

    Args:
        model (onnx.ModelProto): The ONNX model whose node names are to be checked and updated.

    Returns:
        onnx.ModelProto: The model with all node names made unique.
    """
    model_graph = model.graph
    node_names = [node.name for node in model_graph.node]
    name_counter = Counter(node_names)
    used_names = set(node_names)

    for node in model_graph.node:
        if node.name == "" or name_counter[node.name] > 1:
            counter = name_counter[node.op_type]  # Initialize counter based on op_type occurrences
            new_name = f"{node.op_type}_{counter}"
            while new_name in used_names:
                counter += 1
                new_name = f"{node.op_type}_{counter}"
            node.name = new_name
            used_names.add(new_name)

    return model


def get_attribute_as_numpy_array(node: onnx.NodeProto, attribute_name: str) -> Optional[NDArray[Any]]:
    """
    Extract the specified attribute from an ONNX node and converts it to a NumPy array.

    Args:
        node (onnx.NodeProto): The ONNX node from which to extract the attribute.
        attribute_name (str): The name of the attribute to extract.

    Returns:
        Optional[NDArray[Any]]: The attribute value as a NumPy array if found, otherwise None.

    Raises:
        ValueError: If the attribute type is not supported.
    """
    for attr in node.attribute:
        if attr.name == attribute_name:
            if attr.type == onnx.AttributeProto.FLOAT:
                return np.array([attr.f], dtype=np.float32)
            elif attr.type == onnx.AttributeProto.INT:
                return np.array([attr.i])
            elif attr.type == onnx.AttributeProto.STRING:
                return np.array([attr.s.decode("utf-8")])
            elif attr.type == onnx.AttributeProto.TENSOR:
                return onnx.numpy_helper.to_array(attr.t)
            elif attr.type == onnx.AttributeProto.FLOATS:
                return np.array(attr.floats, dtype=np.float32)
            elif attr.type == onnx.AttributeProto.INTS:
                return np.array(attr.ints)
            elif attr.type == onnx.AttributeProto.STRINGS:
                return np.array([s.decode("utf-8") for s in attr.strings])
            else:
                raise ValueError(f"Unsupported attribute type: {attr.type}")
    return None


def move_constants_to_wgts(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Move Constant nodes in the ONNX model graph to the model's weights and deletes the original Constant nodes.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.

    Returns:
        onnx.ModelProto: The modified ONNX model with Constant nodes moved to weights.

    Raises:
        ValueError: If Constant operator has its Attribute set to None.
    """
    for i, node in reversed(list(enumerate(model.graph.node))):
        if node.op_type == "Constant":
            attr = node.attribute[0]
            constant_value = get_attribute_as_numpy_array(node, attr.name)

            if constant_value is None:
                raise ValueError("Attribute should not be None of operator Constant.")
            add_wgt(model, constant_value, node.output[0])
            del model.graph.node[i]
    return model


def add_function_to_model(
    model: onnx.ModelProto, function: onnx.FunctionProto, domain: str = "com.videantis"
) -> onnx.ModelProto:
    """
    Add a function to the ONNX model's function list if it's not already present.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.
        function (onnx.FunctionProto): The function to add to the model.
        domain (str, optional): The domain for the function. Defaults to "com.videantis".

    Returns:
        onnx.ModelProto: The modified ONNX model with the function added.
    """
    for func in model.functions:
        if func.name == function.name:
            return model
    model.functions.extend([function])
    model.functions[-1].domain = domain
    return model


def _get_nxt_nodes(
    model: onnx.ModelProto, node_name: str
) -> Tuple[Optional[Sequence[int]], Optional[Sequence[onnx.NodeProto]]]:
    """
    Retrieve the nodes immediately following the specified node in the ONNX model graph.

    Args:
        model (onnx.ModelProto): The ONNX model to search.
        node_name (str): The name of the node to find the following nodes for.

    Returns:
        Tuple[Optional[Sequence[int]], Optional[Sequence[onnx.NodeProto]]]: A tuple containing a list of indices of the next nodes and the next nodes themselves. Returns (None, None) if no next nodes are found.
    """
    from vnnort.optimizer.pattern_detection import find_node_by_name, get_nodes_by_input

    i, node = find_node_by_name(model.graph, node_name)
    if node is None or len(node.output) != 1:
        return None, None
    nxt_nodes_idx, nxt_nodes = get_nodes_by_input(model, node.output[0])
    return nxt_nodes_idx, nxt_nodes


def _get_prev_node(model: onnx.ModelProto, node_name: str) -> Tuple[Optional[int], Optional[onnx.NodeProto]]:
    """
    Retrieve the node immediately preceding the specified node in the ONNX model graph.

    Args:
        model (onnx.ModelProto): The ONNX model to search.
        node_name (str): The name of the node to find the preceding node for.

    Returns:
        Tuple[Optional[int], Optional[onnx.NodeProto]]: A tuple containing the index of the previous node and the previous node itself. Returns (None, None) if no previous node is found.
    """
    from vnnort.optimizer.pattern_detection import find_node_by_name, get_nodes_by_output

    i, node = find_node_by_name(model.graph, node_name)
    if node is None or len(node.output) != 1:
        return None, None
    prev_nodes_idx, prev_nodes = get_nodes_by_output(model, node.input[0])
    if len(prev_nodes_idx) == 1:
        prev_node = prev_nodes[0]
        prev_node_idx = prev_nodes_idx[0]
        return prev_node_idx, prev_node
    else:
        return None, None


def fuse_muls(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply rewrite rules for "vidConv" + "mult. Shortcut" fusion.

    Args:
        model (onnx.ModelProto): The ONNX model to optimize.

    Returns:
        onnx.ModelProto: The optimized ONNX model with fused multiplication patterns.
    """
    from vnnort.optimizer.pattern_detection import match_patterns_onnxscript_rule

    ir_model = onnxscript.ir.serde.deserialize_model(model)
    rules = [vidConvShortcutPostFuse.rule(), vidConvShortcutPreFuse.rule()]
    for rule in rules:
        ir_model, c_cnt = match_patterns_onnxscript_rule(ir_model, rule, verbose=rule._verbose, commute=True)

    rewritten_model = onnxscript.ir.serde.serialize_model(ir_model)
    return rewritten_model


def fuse_reshape_modes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Fuses reshape mode operations into the preceding convolutional nodes in the ONNX model to optimize the graph.

    This function identifies specific reshape mode nodes (e.g., reshapeToWgts_MUL_EXPAND) that directly follow
    convolutional operations and fuses them by adding a reshape mode attribute to the convolution node, removing
    the separate reshape mode node.

    Args:
        model (onnx.ModelProto): The ONNX model to optimize by fusing reshape modes.

    Returns:
        onnx.ModelProto: The modified ONNX model with fused reshape modes.
    """
    from vnnort.optimizer.pattern_detection import get_nodes_by_input
    from vnnort.optimizer.utils import remove_node

    reshape_modes = ["reshapeToWgts_MUL_EXPAND"]
    for i, node in reversed(list(enumerate(model.graph.node))):
        if node.op_type == "vidConv":
            _, next_nodes = get_nodes_by_input(model, node.output[0])
            if len(next_nodes) == 1:
                if next_nodes[0].op_type in reshape_modes:
                    model.graph.node[i].attribute.extend(
                        [make_attribute("reshape_mode", next_nodes[0].op_type.replace("reshapeToWgts_", "").upper())]
                    )
                    model = remove_node(model, next_nodes[0].name)
    return model


def remove_node(model: onnx.ModelProto, node_name: str) -> onnx.ModelProto:
    """
    Remove a specified node from the ONNX model, rewiring its inputs and outputs to maintain graph structure.

    If the node has only one dynamic input and one output, it will be removed, and its output will be
    connected to the input of the following node(s). If the node is an output node or cannot be safely removed,
    an error is raised.

    Args:
        model (onnx.ModelProto): The ONNX model from which the node should be removed.
        node_name (str): The name of the node to be removed.

    Returns:
        onnx.ModelProto: The modified ONNX model with the node removed if successful.

    Raises:
        RuntimeError: If the node cannot be removed due to multiple dynamic inputs or outputs.
    """
    from vnnort.optimizer.pattern_detection import find_node_by_name, get_nodes_by_input

    node_idx, node = find_node_by_name(model.graph, node_name)
    if node_idx is None:
        logger.warning(f"Node not in model: {node_name}")
        return model
    actual_inputs = [inp for inp in node.input if inp != ""]  # Remove empty default inputs
    dyn_inputs = len(actual_inputs) - _get_no_of_static_inputs(model, node)
    nxt_node_idxs, nxt_nodes = get_nodes_by_input(model.graph, node.output[0])

    if dyn_inputs == 1:
        if len(node.output) == 1 or node.op_type == "vidConv":
            model_outs = [out.name for out in model.graph.output]
            if node.output[0] not in model_outs:
                for nxt_node_idx, nxt_node in zip(nxt_node_idxs, nxt_nodes):
                    for i, inp in enumerate(nxt_node.input):
                        if nxt_node.input[i] == node.output[0]:
                            if node.input[0] not in [init.name for init in model.graph.initializer]:
                                model.graph.node[nxt_node_idx].input[i] = node.input[0]
                            else:
                                model.graph.node[nxt_node_idx].input[i] = node.input[1]
                del model.graph.node[node_idx]
                return model
            else:
                prev_idx, prev_node = _get_prev_node(model, node_name)
                if prev_idx is not None:
                    model.graph.node[prev_idx].output[0] = node.output[0]
                    del model.graph.node[node_idx]
                    return model
                else:
                    raise RuntimeError(f"Could not remove node: {node_idx}, {node_name}")
    raise RuntimeError(
        f"Dynamic inputs != 1 or len(node.output) != 1. Could not remove node: {node_idx}, {node_name}, {node.op_type}. "
        f"Dynamic inputs: {dyn_inputs}, Output length: {len(node.output)}, "
        f"Static inputs: {_get_no_of_static_inputs(model, node)}"
    )


def remove_nodes(model: onnx.ModelProto, op_types: List[str], excludes: Optional[List[str]] = None) -> onnx.ModelProto:
    """
    Remove all nodes of specified types from the model, except those in the excludes list.

    Args:
        model (onnx.ModelProto): The input ONNX model
        op_types (List[str]): List of operator types to remove (e.g. ["Relu", "Identity"])
        excludes (Optional[List[str]]): List of node names to preserve even if their type matches.
                                      Defaults to None.

    Returns:
        onnx.ModelProto: Modified model with specified nodes removed

    Raises:
        ValueError: If model is invalid or if critical nodes cannot be removed
    """
    if not isinstance(model, onnx.ModelProto):
        raise ValueError("Input must be an ONNX ModelProto")

    if not op_types:
        return model

    # Make a copy to avoid modifying original
    model = copy.deepcopy(model)
    excludes = excludes or []

    # Track nodes to remove
    nodes_to_remove = []

    # First pass: identify nodes to remove
    for node in model.graph.node:
        if node.op_type in op_types and node.name not in excludes:
            nodes_to_remove.append(node.name)

    logger.info(f"Found {len(nodes_to_remove)} nodes to remove of types {op_types}")

    # Second pass: remove nodes one by one
    removed_count = 0
    failed_removals = []

    for node_name in nodes_to_remove:
        try:
            model = remove_node(model, node_name)
            removed_count += 1
        except RuntimeError as e:
            logger.warning(f"Could not remove node {node_name}: {str(e)}")
            failed_removals.append((node_name, str(e)))
            continue

    if failed_removals:
        logger.warning(f"Failed to remove {len(failed_removals)} nodes:")
        for name, error in failed_removals:
            logger.warning(f"  {name}: {error}")

    logger.info(f"Successfully removed {removed_count} nodes")

    return model


def insert_onnx_node(
    model: onnx.ModelProto, new_node_op: str, prev_node_name: str, next_node_name: str, new_node_name: str = ""
) -> onnx.ModelProto:
    """
    Insert a new ONNX node of type `new_node_op` between `prev_node_name` and `next_node_name`.

    Args:
        model (onnx.ModelProto): The ONNX model.
        new_node_op (str): The operation type of the new node to insert (e.g., "Sigmoid").
        prev_node_name (str): The name of the node after which the new node should be inserted.
        next_node_name (str): The name of the node before which the new node should be inserted.
        new_node_name (str): optional new name of node

    Raises:
        ValueError: if prev_node_name or next_node_name was not found in model.

    Returns:
        onnx.ModelProto: The modified ONNX model.
    """
    from vnnort.optimizer.pattern_detection import find_node_by_name

    # Find next and previous nodes in graph
    prev_index, prev_node = find_node_by_name(model.graph, prev_node_name)
    if prev_index is None:
        raise ValueError(f"Could not find {prev_node_name}")

    next_index, next_node = find_node_by_name(model.graph, next_node_name)
    if next_index is None:
        raise ValueError(f"Could not find {next_node_name}")

    # Create new node
    node_name = f"{new_node_op}_{prev_index}_node"
    if new_node_name != "":
        node_name = new_node_name

    # Create a new intermediate tensor name
    new_node_output_name = f"{new_node_op.lower()}_{prev_index}_output"
    if new_node_name != "":
        new_node_output_name = f"{new_node_name}_output"

    # Create the new node
    new_node = onnx.helper.make_node(
        new_node_op,  # Type of operation (e.g., "Sigmoid")
        inputs=[prev_node.output[0]],  # Use the output of prev_node as input to new_node
        outputs=[new_node_output_name],  # Output of the new node
        name=node_name,
    )

    # Insert the new node into the graph
    model.graph.node.insert(next_index, new_node)

    # Redirect the input of the next node
    for i, input_name in enumerate(next_node.input):
        if input_name == prev_node.output[0]:
            next_node.input[i] = new_node_output_name
    return model
