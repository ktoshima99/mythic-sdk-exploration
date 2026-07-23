import logging
from inspect import getmembers
from typing import Any, Sequence, Tuple

import numpy as np
import onnx
import onnxscript
from numpy.typing import NDArray
from onnxscript.rewriter.pattern import RewriteRule

import vnnort.optimizer.functions as functions
from vnnort.optimizer.patterns import ShortcutPatternLinear  # type: ignore[attr-defined]
from vnnort.optimizer.patterns import fetch_pattern_rules  # type: ignore[attr-defined]
from vnnort.optimizer.utils import (
    add_function_to_model,
    infer_shapes_runtime,
    move_static_cons_to_wgts,
    remove_unused_nodes,
    remove_unused_wgts,
)
from vnnort.quantizer.qdq_layer import QDQLayer  # type: ignore
from vnnort.utils.onnx_utils import VIDEANTIS_ONNX_DOMAIN, VIDEANTIS_ONNX_VERSION
from vnnort.utils.onnx_utils.utils import register_op_schema

logger = logging.getLogger(__name__)


def is_onnx_function(obj: Any) -> bool:
    """Check if an object is an ONNX function.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: True if the object is an ONNX function, otherwise False.
    """
    return isinstance(obj, onnxscript.values.OnnxFunction)


def register_videantis_function_opschemas() -> None:
    """Register custom Videantis function OpSchemas with ONNX.

    This function retrieves all ONNX function definitions from the `functions` module
    and registers their corresponding OpSchema with the ONNX framework.
    It collects all function members from the `functions` module that match the ONNX function type.
    It registers each function's OpSchema using `onnx.defs.register_schema`.
    """
    vid_functions = [(member[0], getattr(functions, member[0])) for member in (getmembers(functions, is_onnx_function))]

    for vid_func in vid_functions:
        register_op_schema(vid_func[1].op_schema)

    # Also add QDQLayer from quantization
    onnx.defs.register_schema(QDQLayer.op_schema)


def add_all_videantis_functions(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add all Videantis custom functions to the ONNX model if not already present.

    This function iterates through Videantis-specific ONNX functions and adds them to the model's functions if they
    are not already included. The function also ensures that the required Videantis domain is in the opset imports.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.

    Returns:
        onnx.ModelProto: The modified ONNX model with Videantis functions added.
    """
    vid_functions = [(member[0], getattr(functions, member[0])) for member in (getmembers(functions, is_onnx_function))]
    for vid_func in vid_functions:

        if vid_func[0] not in [func.name for func in model.functions]:
            model = add_function_to_model(model, vid_func[1].to_function_proto())
    for i_func, func in enumerate(model.functions):
        vid_domain_missing = True
        for imp in func.opset_import:
            if imp.domain == "com.videantis":
                vid_domain_missing = False
        if vid_domain_missing:
            model.functions[i_func].opset_import.extend([onnx.helper.make_operatorsetid("com.videantis", 1)])

        # Register function schema with onnx

    model.opset_import.extend([onnx.helper.make_operatorsetid("com.videantis", 1)])
    return model


def add_missing_videantis_functions(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add missing Videantis functions to an ONNX model based on nodes in the graph.

    This function checks nodes in the model's graph for references to Videantis-specific functions and adds any
    missing functions to the model. The function also ensures that the Videantis domain is present in the opset imports.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.

    Returns:
        onnx.ModelProto: The modified ONNX model with missing Videantis functions added.
    """
    vidFunctions = [(member[0], getattr(functions, member[0])) for member in (getmembers(functions, is_onnx_function))]
    for node in model.graph.node:
        if node.domain == "com.videantis":
            if node.op_type not in [func.name for func in model.functions]:
                for vidFunc in vidFunctions:
                    if vidFunc[0] == node.op_type:
                        model = add_function_to_model(model, vidFunc[1].to_function_proto())
    for i_func, func in enumerate(model.functions):
        vid_domain_missing = True
        for imp in func.opset_import:
            if imp.domain == "com.videantis":
                vid_domain_missing = False
        if vid_domain_missing:
            model.functions[i_func].opset_import.extend([onnx.helper.make_operatorsetid("com.videantis", 1)])

    return model


def match_patterns_onnxscript_rule(
    ir_model: onnxscript.ir.Model, rule: RewriteRule, verbose: int = 0, commute: bool = False
) -> Tuple[onnxscript.ir.Model, int]:
    """Apply a rewrite rule to an onnxscript model using ONNX Script.

    Args:
        ir_model (onnxscript.ir.Model): The ONNX model to apply the rule to.
        rule (RewriteRule): The pattern-matching rule to apply.
        verbose (int, optional): Verbosity level for debugging output. Defaults to 0.
        commute (bool, optional): Flag to indicate if commutative transformations should be applied. Defaults to False.

    Returns:
        Tuple[onnxscript.ir.Model, int]: The modified model and the count of pattern matches found.

    Raises:
        RuntimeError: If deserialization did not return a IR Model.
    """
    # ir_model = onnxscript.ir.serde.deserialize_model(model)
    rule._verbose = verbose
    if not isinstance(ir_model, onnxscript.ir.Model):
        raise RuntimeError("Deserilization should return a onnxscript.ir.Model.")
    count = onnxscript.rewriter.pattern.RewriteRuleSet([rule], commute=commute)._apply_to_graph_or_function(
        ir_model, ir_model.graph, verbose=verbose
    )
    return ir_model, count


# TODO: Fix this mess
# vid_pattern_match_rules_for_onnx_model, vid_match_patterns_onnxscript, _match_patterns_single all to the same...
def vid_pattern_match_rules_for_onnx_model(  # noqa: C901 ---ignores "too complex"-error
    onnx_model: onnx.ModelProto, rules: list[RewriteRule]
) -> onnx.ModelProto:
    """Apply a rewrite rule to an onnxscript model using ONNX Script.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model to apply the rule to.
        rules (list[RewriteRule]): A list of rules to apply

    Returns:
        onnx.ModelProto: The modified model
    """
    ir_model = onnxscript.ir.serde.deserialize_model(onnx_model)
    for rule in rules:
        ir_model, c_cnt = match_patterns_onnxscript_rule(ir_model, rule, verbose=rule._verbose, commute=True)
    rewritten_model = onnxscript.ir.serde.serialize_model(ir_model)

    init_dict = dict()
    inits = [init.name for init in rewritten_model.graph.initializer]
    for i, init in enumerate(inits):
        new_name = "init" + str(np.random.randint(0, int(2e8)))
        while new_name in inits:
            new_name = "init" + str(np.random.randint(0, int(2e8)))
        if "val_" in init:
            init_dict.update({rewritten_model.graph.initializer[i].name: new_name})
            rewritten_model.graph.initializer[i].name = new_name

    conn_dict = init_dict
    for i, node in enumerate(rewritten_model.graph.node):
        for inp_i, inp in enumerate(node.input):
            if "val_" not in inp:
                continue
            if inp in conn_dict:
                rewritten_model.graph.node[i].input[inp_i] = conn_dict[inp]
        for out_i, out in enumerate(node.output):
            if "val_" not in out:
                continue
            new_name = out.replace("val", "connection" + str(node.name) + str(np.random.randint(0, int(2e8))))
            if out not in conn_dict:
                conn_dict.update({out: new_name})
            rewritten_model.graph.node[i].output[out_i] = conn_dict[out]
    return rewritten_model


def vid_match_patterns_onnxscript(  # noqa: C901 ---ignores "too complex"-error
    model: onnx.ModelProto, rule: RewriteRule, verbose: int = 0, commute: bool = False
) -> Tuple[onnx.ModelProto, int]:
    """Apply Videantis-specific rewrite patterns to an ONNX model and rename conflicting initializers.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.
        rule (RewriteRule): The pattern-matching rule to apply.
        verbose (int, optional): Verbosity level for debugging output. Defaults to 0.
        commute (bool, optional): Flag to indicate if commutative transformations should be applied. Defaults to False.

    Returns:
        Tuple[onnx.ModelProto, int]: The modified model and the count of patterns matched and rewritten.
    """
    ir_model = onnxscript.ir.serde.deserialize_model(model)
    rewritten_model, count = match_patterns_onnxscript_rule(ir_model, rule, verbose=verbose, commute=commute)
    rewritten_model = onnxscript.ir.serde.serialize_model(rewritten_model)
    if count:
        logger.info(f"Found: {rule.name} {count} times")
    init_dict = dict()
    inits = [init.name for init in rewritten_model.graph.initializer]
    for i, init in enumerate(inits):
        new_name = "init" + str(np.random.randint(0, int(2e8)))
        while new_name in inits:
            new_name = "init" + str(np.random.randint(0, int(2e8)))
        if "val_" in init:
            init_dict.update({rewritten_model.graph.initializer[i].name: new_name})
            rewritten_model.graph.initializer[i].name = new_name

    conn_dict = init_dict
    for i, node in enumerate(rewritten_model.graph.node):
        for inp_i, inp in enumerate(node.input):
            if "val_" not in inp:
                continue
            if inp in conn_dict:
                rewritten_model.graph.node[i].input[inp_i] = conn_dict[inp]
        for out_i, out in enumerate(node.output):
            if "val_" not in out:
                continue
            new_name = out.replace("val", "connection" + str(node.name) + str(np.random.randint(0, int(2e8))))
            if out not in conn_dict:
                conn_dict.update({out: new_name})
            rewritten_model.graph.node[i].output[out_i] = conn_dict[out]

    return rewritten_model, count


def _match_patterns_single(  # noqa: C901 ---ignores "too complex"-error
    model: onnx.ModelProto, commute: bool = False
) -> onnx.ModelProto:
    """Apply all pattern rules iteratively until no further matches are found in an ONNX model.

    This function fetches all available pattern-matching rules and applies them to the model.
    The process continues until no further patterns can be matched.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.
        commute (bool, optional): Flag to allow commutative pattern matching. Defaults to False.

    Returns:
        onnx.ModelProto: The modified model with patterns matched and rewritten.
    """
    count = 1
    while count:
        count = 0
        ir_model = onnxscript.ir.serde.deserialize_model(model)
        for i, rule in enumerate(fetch_pattern_rules()):
            # model, c_cnt = vid_match_patterns_onnxscript(model, rule, verbose=rule._verbose, commute=commute)
            ir_model, c_cnt = match_patterns_onnxscript_rule(ir_model, rule, verbose=rule._verbose, commute=commute)
            if c_cnt:
                logger.info(f"Found: {rule.name} {c_cnt} times")

            count = c_cnt if c_cnt > count else count

        rewritten_model = onnxscript.ir.serde.serialize_model(ir_model)
        # Make names unique
        init_dict = dict()
        inits = [init.name for init in rewritten_model.graph.initializer]
        for i, init in enumerate(inits):
            new_name = "init" + str(np.random.randint(0, int(2e8)))
            while new_name in inits:
                new_name = "init" + str(np.random.randint(0, int(2e8)))
            if "val_" in init:
                init_dict.update({rewritten_model.graph.initializer[i].name: new_name})
                rewritten_model.graph.initializer[i].name = new_name

        conn_dict = init_dict
        for i, node in enumerate(rewritten_model.graph.node):
            for inp_i, inp in enumerate(node.input):
                if "val_" not in inp:
                    continue
                if inp in conn_dict:
                    rewritten_model.graph.node[i].input[inp_i] = conn_dict[inp]
            for out_i, out in enumerate(node.output):
                if "val_" not in out:
                    continue
                new_name = out.replace("val", "connection" + str(node.name) + str(np.random.randint(0, int(2e8))))
                if out not in conn_dict:
                    conn_dict.update({out: new_name})
                rewritten_model.graph.node[i].output[out_i] = conn_dict[out]

        model = rewritten_model
    model.opset_import.extend([onnx.helper.make_opsetid(domain=VIDEANTIS_ONNX_DOMAIN, version=VIDEANTIS_ONNX_VERSION)])
    model.opset_import.extend([onnx.helper.make_opsetid(domain="com.microsoft", version=1)])
    return model


def pattern_match(
    model: onnx.ModelProto,
    test_inputs1: dict[str, NDArray[Any]],
    test_inputs2: dict[str, NDArray[Any]],
    commute: bool = True,
) -> onnx.ModelProto:
    """
    Match and rewrite patterns in an ONNX model to fit our custom architecture.

    Args:
        model (onnx.ModelProto): The ONNX model to optimize and transform.
        test_inputs1 (dict[str, NDArray[Any]]): Valid data used to profile static connections.
        test_inputs2 (dict[str, NDArray[Any]]): Valid data used to profile static connections.
        commute (bool): Flag to control whether to also use commutated inputs for pattern matching.

    Returns:
        onnx.ModelProto: The transformed ONNX model after applying all pattern matches and optimizations.

    Notes:
        - The function uses a two-level loop: an outer loop ensures the process continues until no more high-level rewrites are needed, while an inner loop applies individual rules repeatedly until no further matches are found.
        - Final steps include moving static connections to weights, removing unused weights, and cleaning up any unused nodes in the model.
    """
    model = add_all_videantis_functions(model)
    cnt = 1
    while cnt:
        model = _match_patterns_single(model, commute=commute)
        del model.graph.value_info[:]
        model, cnt = move_static_cons_to_wgts(model, [test_inputs1, test_inputs2], return_no_rem_nodes=True)
        model = remove_unused_wgts(model)
        model = remove_unused_nodes(model, None, test_inputs1)
        del model.graph.value_info[:]
        infer_shapes_runtime(model, test_inputs1)

    rule = ShortcutPatternLinear.rule()
    ir_model = onnxscript.ir.serde.deserialize_model(model)
    ir_model, c_cnt = match_patterns_onnxscript_rule(ir_model, rule, verbose=rule._verbose, commute=commute)
    model = onnxscript.ir.serde.serialize_model(ir_model)
    return model


def find_node_by_name(graph: onnx.GraphProto, node_name: str) -> Tuple[int, onnx.NodeProto]:
    """
    Search for a node by its name in an ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph to search within.
        node_name (str): The name of the node to search for.

    Raises:
        ValueError: if no node with specified name can be found in graph

    Returns:
        Tuple[int, onnx.NodeProto]: A tuple containing the node's index in the graph and the node object if found, or None if not found.
    """
    for index, node in enumerate(graph.node):
        if node_name == node.name:
            return index, node
    # Raise error if the node is not found
    raise ValueError(f"Node with name '{node_name}' not found in the graph.")


def get_nodes_by_output(model: onnx.ModelProto, output_name: str) -> Tuple[Sequence[int], Sequence[onnx.NodeProto]]:
    """
    Retrieve nodes and their indices from an ONNX model or graph with a specific output name.

    Args:
        model (onnx.ModelProto): The ONNX model or graph to search. If a model is provided, its graph is used.
        output_name (str): The name of the output to search for in the nodes.

    Returns:
        Tuple[Sequence[int], Sequence[onnx.NodeProto]]: A tuple containing:
            - A list of nodes that have `output_name` as one of their outputs.
            - A list of indices corresponding to each node found.
    """
    # Use model's graph if model is an instance of ModelProto, else use model directly.
    graph = model.graph if isinstance(model, onnx.ModelProto) else model

    nodes, indices = [], []
    for index, model_node in enumerate(graph.node):
        if output_name in model_node.output:
            nodes.append(model_node)
            indices.append(index)

    return indices, nodes


def get_nodes_by_input(model: onnx.ModelProto, input_name: str) -> Tuple[Sequence[int], Sequence[onnx.NodeProto]]:
    """
    Retrieve nodes and their indices from an ONNX model or graph with a specific input name.

    Args:
        model (onnx.ModelProto): The ONNX model or graph to search. If a model is provided, its graph is used.
        input_name (str): The name of the input to search for in the nodes.

    Returns:
        Tuple[Sequence[int], Sequence[onnx.NodeProto]]: A tuple containing:
            - A list of indices corresponding to each node found.
            - A list of nodes that have `input_name` as one of their inputs.
    """
    # Use model's graph if model is an instance of ModelProto, else use model directly.
    graph = model
    if isinstance(model, onnx.ModelProto):
        graph = model.graph

    nodes, indices = [], []
    for index, model_node in enumerate(graph.node):
        if input_name in model_node.input:
            nodes.append(model_node)
            indices.append(index)
    return indices, nodes
