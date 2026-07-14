# type: ignore
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx.helper import printable_attribute
from onnx.inliner import inline_local_functions
from tabulate import tabulate

from vnnort import logger
from vnnort.optimizer.utils import get_value_info_by_name_val_info
from vnnort.optimizer.utils import get_wgt_by_name
from vnnort.optimizer.utils import infer_shapes_runtime
from vnnort.optimizer.utils import move_constants_to_wgts


def profile(  # noqa
    model: onnx.ModelProto,
    dummy_inputs: List[np.ndarray],
    print_output: bool = True,
    verbose: bool = False,
    inline: bool = False,
    output_file: Optional[str] = None,
    table_fmt: str = "fancy_grid",
) -> Tuple[int, int, List[Dict[Any]]]:
    """Profile the multiply-accumulate operations (MACs) and parameters of an ONNX model.

    This function calculates the total number of parameters and MACs for the given ONNX model.
    It can optionally print the results, provide verbose output for unsupported node types,
    inline local functions, and export the profiling table to a file.

    Args:
        model (onnx.ModelProto): The ONNX model to profile.
        dummy_inputs (list[np.ndarray]): Dummy input arrays used for runtime shape inference.
        print_output (bool, optional): If True, print MACs and parameter counts. Defaults to True.
        verbose (bool, optional): If True, print warnings for unsupported node types. Defaults to False.
        inline (bool, optional): If True, inline local functions within the model. Defaults to False.
        output_file (str | None, optional): Path to save the profiling table. If None, do not save.
            Defaults to None.
        table_fmt (str, optional): Output table format supported by ``tabulate``
            (e.g. ``github``, ``fancy_grid``, ``simple``). Defaults to ``fancy_grid``.

    Returns:
        tuple[int, int, list[dict[str, object]]]: ``(params, macs, nodes)`` where
        ``params`` is the total parameter count, ``macs`` is the total MAC count,
        and ``nodes`` is a list of per-node profiling information.

    Note:
        Clears the graph ``value_info`` field and optionally inlines local functions
        if ``inline`` is True. Shapes are inferred at runtime using ``dummy_inputs``.
        Parameters are counted for nodes with weights in initializers. MACs are computed
        for selected ops (Conv, MatMul, Gemm). Unsupported node types yield 0 MACs;
        if ``verbose`` is True, a warning is printed. If ``output_file`` is set, the
        profiling table is written using ``table_fmt``.

    Example::

        import onnx
        import numpy as np

        model = onnx.load("model.onnx")
        dummy_inputs = [np.random.randn(1, 3, 224, 224).astype(np.float32)]
        params, macs, nodes = profile(
            model,
            dummy_inputs,
            print_output=True,
            verbose=True,
            inline=True,
            output_file="profiling_report.txt",
            table_fmt="github",
        )
        print(f"Total Parameters: {params}")
        print(f"Total MACs: {macs}")
    """
    nodes = []

    inits = [init.name for init in model.graph.initializer]
    unsupported_types = ["preprocessing", "postprocessing"]

    del model.graph.value_info[:]
    if inline:
        model = inline_local_functions(model)
    infer_shapes_runtime(model, dummy_inputs)
    model = move_constants_to_wgts(model)

    params = 0
    macs = 0
    for i, node in enumerate(model.graph.node):
        if node.op_type in unsupported_types:
            continue
        node_macs = 0
        node_params = 0
        wgt_shape = None

        # params
        for inp in node.input:
            if inp in inits:
                arr = get_wgt_by_name(model, inp)
                try:
                    node_params += np.prod(arr.shape)
                except Exception:
                    logger.warning(
                        "arr is float/int not array",
                    )

        # macs
        if node.op_type in ["Conv", "FusedConv", "vidConv"]:

            if node.input[1] in inits:
                wgt_shape = get_wgt_by_name(model, node.input[1]).shape
            else:
                wgt_shape = get_value_info_by_name_val_info(model.graph.value_info, node.input[1])[1]

            output_shape = get_value_info_by_name_val_info(model.graph.value_info, node.output[0])[1]
            if node.op_type == "vidConv":
                output_shape = get_value_info_by_name_val_info(model.graph.value_info, node.output[2])[1]
            if len(output_shape) == 2 and len(wgt_shape) == 2:
                output_shape = output_shape + (1, 1)
                wgt_shape = wgt_shape + (1, 1)
            try:
                node_macs += (
                    wgt_shape[2] * wgt_shape[3] * wgt_shape[1] * wgt_shape[0] * output_shape[2] * output_shape[3]
                )
            except Exception:
                logger.warning("mac calc failed for ", node.name, node.op_type)
            # elif node.input[1] in [val_info.name for val_info in model.graph.value_info]:
            #    wgt is dynamic
        elif node.op_type == "vidLayerNormalization":
            node_macs += 0
        elif node.op_type in ["MatMul", "Gemm"]:
            if node.input[1] in inits:
                wgt_shape = get_wgt_by_name(model, node.input[1]).shape
            else:
                wgt_shape = get_value_info_by_name_val_info(model.graph.value_info, node.input[1])[1]

            output_shape = get_value_info_by_name_val_info(model.graph.value_info, node.output[0])[1]
            node_macs += wgt_shape[1] * wgt_shape[0] * np.array(output_shape[:-1]).prod()
        else:
            if verbose:
                if node.op_type not in unsupported_types:
                    unsupported_types.append(node.op_type)
                    logger.warning("Unsupported op_type: ", node.op_type)
        params += node_params
        macs += node_macs
        # node_data
        inp_shape = None
        # print(node.name,node.op_type)
        if node.input[0] in inits:
            try:
                inp_shape = get_wgt_by_name(model, node.input[0]).shape
            except Exception:
                logger.warning("wgt has no shape")
                inp_shape = None
        else:
            inp_shape = get_value_info_by_name_val_info(model.graph.value_info, node.input[0])[1]
        try:
            if inp_shape is None:
                input_idx = [input.name for input in model.graph.input].index(node.input[0])
                inp_shape = dummy_inputs[input_idx].shape
        except ValueError:
            logger.warning("Error: The input item was not found in the list.")
            inp_shape = None  # or set a default value, e.g., inp_shape = (0,)

        output_shape = get_value_info_by_name_val_info(model.graph.value_info, node.output[0])[1]
        attributes = ["-" + printable_attribute(attribute) for attribute in node.attribute]
        node_data = {
            "Name": node.name,
            "OP type": node.op_type,
            "Attributes": "\n".join(attributes),
            "Input Shape": inp_shape,
            "Weight Shape": wgt_shape,
            "Output Shape": output_shape,
            "MACs": node_macs,
            "Params": node_params,
        }

        nodes.append(node_data)
        # print(node_data)

    macs = int(macs)
    params = int(params)
    nodes.append(
        {
            "Name": "Summary",
            "OP type": "Layer Count: " + str(len(nodes)),
            "MACs": str(round(macs / 1e9, 3)) + "G MACs",
            "Params": str(round(params / 1e6, 3)) + "M Params",
        }
    )
    if print_output:
        if output_file is not None:
            with open(output_file, "w") as file:
                file.write(tabulate(nodes, headers="keys", tablefmt=table_fmt))
        else:
            print(tabulate(nodes, headers="keys", tablefmt=table_fmt))
    return params, macs, nodes
