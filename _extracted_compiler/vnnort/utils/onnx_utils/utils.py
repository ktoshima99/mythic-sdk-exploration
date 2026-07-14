import onnx
from onnx.defs import OpSchema


def set_input_shape(model: onnx.ModelProto, input_name: str, new_shape: list[int]) -> onnx.ModelProto:
    """
    Update the shape of a given graph input name in an ONNX ModelProto.

    Args:
        model (onnx.ModelProto): The ModelProto object.
        input_name (str): The name of the input to modify.
        new_shape (list[int]): A list of dimensions (use None or -1 for dynamic dimensions).

    Returns:
        onnx.ModelProto: The modified ModelProto.

    Raises:
        ValueError: If ``input_name`` is not found in the model's graph inputs.
    """
    for inp in model.graph.input:
        if inp.name == input_name:
            tensor_type = inp.type.tensor_type
            tensor_type.shape.ClearField("dim")  # clear existing dims
            for dim in new_shape:
                dim_proto = tensor_type.shape.dim.add()
                if dim is None or dim < 0:  # dynamic dimension
                    dim_proto.dim_param = "None"
                else:
                    dim_proto.dim_value = dim
            return model

    msg = f"Input name '{input_name}' not found in the model graph inputs."
    raise ValueError(msg)


def set_output_shape(model: onnx.ModelProto, output_name: str, new_shape: list[int]) -> onnx.ModelProto:
    """
    Update the output shape of a given graph output name in an ONNX ModelProto.

    Args:
        model (onnx.ModelProto): The ModelProto object.
        output_name (str): The name of the output to modify.
        new_shape (list[int]): A list of dimensions (use None or -1 for dynamic dimensions).

    Returns:
        onnx.ModelProto: The modified ModelProto.
    """
    for graph_output in model.graph.output:
        if graph_output.name == output_name:
            graph_output.type.tensor_type.shape.ClearField("dim")
            for dim in new_shape:
                dim_proto = graph_output.type.tensor_type.shape.dim.add()
                if dim is None or dim < 0:  # dynamic dimension
                    dim_proto.dim_param = "None"
                else:
                    dim_proto.dim_value = dim
            break
    return model


def register_op_schema(op_schema: OpSchema) -> None:
    """Register op_schema with ONNX.

    This is a small wrapper around onnx.defs.register_schema, which checks if the schema is already registered.
    Otherwise that function would raise an exception

    Args:
        op_schema (OpSchema): The schema to register

    Returns:
        None: None
    """
    op_type = op_schema.name
    domain = op_schema.domain
    if not onnx.defs.has(op_type, domain):
        onnx.defs.register_schema(op_schema)
