import onnx
from onnx import defs

from vnnort.utils.onnx_utils.graph_helper import ONNXGraphHelper, TensorType
from vnnort.utils.onnx_utils.unique_initializers import make_initializers_unique

OUTPUT_TENSOR_SUFFIX = "_output"
INPUT_TENSOR_SUFFIX = "_input"


def _optype_input_index_to_name(op_type: str, domain: str, index: int) -> str:
    """Return the name of the parameter of a specific onnx optype given its position. in the arguments."""
    schema = defs.get_schema(op_type, domain=domain)
    return schema.inputs[index].name


def standardize_naming(model: onnx.ModelProto) -> onnx.ModelProto:  # noqa
    # This only works if all initializers are only used once
    make_initializers_unique(model)

    graph_helper = ONNXGraphHelper(model, load_data=False)
    nodes = list(graph_helper.nodes.values())
    graph = model.graph

    # Maximum digits to use for naming scheme of nodes
    max_digits_for_naming = len(str(len(nodes)))

    # Helper structures to quickly access onnx data structures
    onnx_node_map = {node.name: node for node in graph.node}
    onnx_initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    onnx_value_info_map = {value_info.name: value_info for value_info in graph.value_info}

    # Keep track of old to new names
    old_to_new_node_names: dict[str, str | None] = {node.name: None for node in nodes}

    sorted_nodes = nodes

    def _change_output_tensor_name(output_name: str, new_output_name: str) -> None:
        """Change name of tensor output_name to new_output_name."""
        tensor = graph_helper.tensors[output_name]
        if tensor.tensor_type is TensorType.GRAPH_OUTPUT:
            return  # Graph output names are not changed
        producer_name = tensor.producer.name
        onnx_node = onnx_node_map[producer_name]
        index = [output for output in onnx_node.output].index(output_name)
        if len(onnx_node.output) > 1:
            new_output_name += str(index)

        old_name = onnx_node.output[index]
        onnx_node.output[index] = new_output_name

        # Update value info object if applicable
        if old_name in onnx_value_info_map:
            onnx_value_info_map[old_name].name = new_output_name

        #  Go to consumer nodes and change input names
        for consumer in tensor.consumers:
            consumer_onnx = onnx_node_map[consumer.name]
            for index, input_name in enumerate(consumer_onnx.input):
                if input_name == old_name:
                    consumer_onnx.input[index] = new_output_name

    # Update node names
    for current_node_index, node in enumerate(sorted_nodes):
        op_type = node.op_type

        # Update onnx node name
        new_node_name = op_type + "_" + str(current_node_index).zfill(max_digits_for_naming)
        onnx_node = onnx_node_map[node.name]
        old_to_new_node_names[node.name] = new_node_name
        onnx_node.name = new_node_name

        # Output tensors are always named after their producer
        for index, output_name in enumerate(onnx_node.output):
            new_output_name = new_node_name + OUTPUT_TENSOR_SUFFIX
            _change_output_tensor_name(output_name, new_output_name)

        # Input tensors are only named if they are initializers
        for index, input_tensor in enumerate(node.inputs):
            if input_tensor is None or input_tensor.tensor_type is not TensorType.INITIALIZER:
                continue  # Only initializers are changed

            parameter_name = _optype_input_index_to_name(onnx_node.op_type, onnx_node.domain, index)

            new_input_name = onnx_node.name + "_" + parameter_name
            old_input_name = onnx_node.input[index]
            onnx_node.input[index] = new_input_name
            initializer = onnx_initializer_map[old_input_name]
            initializer.name = new_input_name

    return model
