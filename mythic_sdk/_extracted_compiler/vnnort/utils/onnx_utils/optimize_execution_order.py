import sys

import onnx

from vnnort.utils.onnx_utils.graph_helper import Node, ONNXGraphHelper, TensorType


def optimize_execution_order(model: onnx.ModelProto) -> onnx.ModelProto:  # noqa
    """Optimize the execution order of the model by reordering its nodes.

    The main goal is to always finish subbranches, after they are started and start with those nodes
    furthest away from the output.

    Args:
        model (onnx.ModelProto): Model to be optimized.

    Returns:
        onnx.ModelProto: Optimized model (inplace)

    Raises:
        RuntimeError: If not all nodes could be reached during ordering.
    """
    # For large models a higher recursion limit is required
    sys.setrecursionlimit(5000)

    graph_helper = ONNXGraphHelper(model, load_data=False)

    # Add a custom field to each node, denoting its distance to the output
    for node in graph_helper.nodes.values():
        setattr(node, "_distance", None)

    # Helper funciton to compute path length to output
    def _compute_length_to_output_path(node: Node, current_length: int) -> None:
        """Go over all nodes recursively from end to start and calculate the longest path to any output."""
        if node._distance is None or node._distance < current_length:  # type: ignore
            node._distance = current_length  # type: ignore
            for input_tensor in node.inputs:
                if input_tensor is None or input_tensor.tensor_type in [TensorType.GRAPH_INPUT, TensorType.INITIALIZER]:
                    continue
                producer = input_tensor.producer
                _compute_length_to_output_path(producer, current_length + 1)

    # Compute the _distance field for all nodes by calling the recursive _compute_length_to_output_path from outputs
    output_tensors = graph_helper.get_output_tensors()
    for output_tensor in output_tensors:
        producer = output_tensor.producer
        start_distance = 0
        _compute_length_to_output_path(producer, start_distance)

    current_nodes = sorted([n for n in graph_helper.nodes.values()], key=lambda n: -n._distance)  # type: ignore
    result_nodes: list[Node] = []

    def _can_be_computed(node: Node) -> bool:
        # All tensors are either are initializers or are already produced by nodes in result_nodes
        for tensor in node.inputs:
            if (
                tensor is None
                or tensor.tensor_type is TensorType.INITIALIZER
                or tensor.tensor_type is TensorType.GRAPH_INPUT
            ):
                continue
            elif tensor.producer not in result_nodes:
                return False
        return True

    def _add_nodes_from_branch(node: Node) -> None:
        """Recursively follow a branch started by node and add its following nodes."""
        # Node was already added
        if node in result_nodes:
            return
        if not _can_be_computed(node):
            return

        result_nodes.append(node)
        # current_nodes.remove(node)

        # Sort consumers by distance and go from there. Priority is to follow branches as long as possible
        consumers = node.outputs[0].consumers.copy()
        consumers.sort(key=lambda n: -n._distance)  # type: ignore

        for consumer in consumers:
            if _can_be_computed(consumer):
                _add_nodes_from_branch(consumer)

    for input_tensor in graph_helper.get_input_tensors():
        for node in input_tensor.consumers:
            _add_nodes_from_branch(node)

    if not len(current_nodes) == len(result_nodes):
        msg = f"Did not reach all graph nodes. Missing {len(result_nodes) - len(current_nodes)} nodes."
        raise RuntimeError(msg)

    # Reorder nodes in onnx
    onnx_node_map = {onnx_node.name: onnx_node for onnx_node in model.graph.node}
    model.graph.node.clear()
    for node in result_nodes:
        model.graph.node.append(onnx_node_map[node.name])

    return model
