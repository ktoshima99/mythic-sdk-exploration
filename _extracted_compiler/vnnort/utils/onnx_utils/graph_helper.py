"""Provides OnnxGraphHelper class with a more graph oriented representation of an ONNX ModelProto object."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import defs, load_external_data_for_model
from onnx.helper import get_attribute_value

from vnnort.optimizer.pattern_detection import register_videantis_function_opschemas
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field

logger = logging.getLogger(__name__)
register_videantis_function_opschemas()


class TensorType(Enum):
    """All possible types a tensor can be."""

    GRAPH_INPUT = 0
    GRAPH_OUTPUT = 1  # All graph outputs are also considered node outputs
    NODE_OUTPUT = 2
    INITIALIZER = 3


@dataclass
class Tensor:
    """Wrapper for a tensor in an ONNX graph.

    This class represents a single tensor in the ONNX graph. All tensors can be considered edges of a graph.
    Tensors may have a single producer, when they are node output or none, if they are graph input or initializers.
    Tensors can be consumed by none or multiple consumer Nodes.
    """

    name: str
    tensor_type: TensorType | None = None
    dtype: np.dtype[Any] | None = None
    shape: list[int] | None = None
    producer: Optional["Node"] = None
    consumers: list["Node"] = field(default_factory=list)

    data: Optional[NDArray[Union[np.float32, np.int8]]] = None

    def __post_init__(self) -> None:
        """Make sure consumers is a list."""
        if self.consumers is None:
            self.consumers = []

    def __eq__(self, other: object) -> bool:
        """Comare two tensor objects for equality.

        Args:
            other (object): Other Tensor object

        Returns:
            bool: Whether two tensors are the same.

        Raises:
            ValueError: When other object is not of type Tensor
        """
        if not isinstance(other, Tensor):
            raise ValueError("Can only compare two Tensor objects")

        result = (
            self.name == other.name
            and self.tensor_type == other.tensor_type
            and self.dtype == other.dtype
            and self.shape == other.shape
            and ((self.producer is None and other.producer is None) or self.producer.name == other.producer.name)  # type: ignore
            and [c.name for c in self.consumers] == [c.name for c in other.consumers]
        )
        return result

    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return f"{self.name} {self.tensor_type} {self.dtype} {self.shape}"


@dataclass
class Node:
    """Wrapper for a node("operator") in an ONNX graph.

    This class represents a single node in the onnx graph. A node may have zero or more input and output tensors.
    """

    name: str
    op_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"{self.name}({self.op_type})"


class ONNXGraphHelper:
    """Wrapper class for an ONNX model providing a more convenient graph oriented view of the model.

    After initialization you may access the model nodes via graph_helper.nodes and the tensors via graph_helper.tensor.
    """

    def __init__(self, model: onnx.ModelProto, load_data: bool = True) -> None:
        """Initialize the ONNXGraphHelper.

        Args:
            model (onnx.ModelProto): model to be represented.
            load_data (bool, optional): whether to load data tensors from the model. Defaults to True.
                If this True, all tensor data (weights and biases) are loaded by default.
        """
        self._load_data = load_data

        # Load the tensor data into the model (we use onnx external_data_mode)
        if load_data:
            model_directory = _get_onnx_meta_field(model, "model_directory")
            load_external_data_for_model(model, model_directory)

        self.nodes: dict[str, Node] = {}
        self.tensors: dict[str, Tensor] = {}

        # Nodes
        self._add_model_nodes(model)

        # Update tensor information
        self._update_model_input_tensors(model)
        self._update_intermediate_tensors(model)
        self._update_model_output_tensors(model)
        self._update_initializer_tensors(model)

    def get_input_tensors(self) -> list[Tensor]:
        """Return all tensors marked as graph input."""
        input_tensors = [tensor for tensor in self.tensors.values() if tensor.tensor_type is TensorType.GRAPH_INPUT]
        return input_tensors

    def get_output_tensors(self) -> list[Tensor]:
        """Return all tensors marked as graph output."""
        output_tensors = [tensor for tensor in self.tensors.values() if tensor.tensor_type is TensorType.GRAPH_OUTPUT]
        return output_tensors

    def _update_intermediate_tensors(self, model: onnx.ModelProto) -> None:
        for value_info in model.graph.value_info:
            tensor_name = value_info.name
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            elem_type = value_info.type.tensor_type.elem_type
            dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)

            self._update_tensor(tensor_name, TensorType.NODE_OUTPUT, dtype, shape)

    def _update_model_input_tensors(self, model: onnx.ModelProto) -> None:
        for value_info in model.graph.input:
            tensor_name = value_info.name
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            elem_type = value_info.type.tensor_type.elem_type
            dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
            self._update_tensor(tensor_name, TensorType.GRAPH_INPUT, dtype, shape)

    def _update_model_output_tensors(self, model: onnx.ModelProto) -> None:
        for value_info in model.graph.output:
            tensor_name = value_info.name
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            elem_type = value_info.type.tensor_type.elem_type
            dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)

            self._update_tensor(tensor_name, TensorType.GRAPH_OUTPUT, dtype, shape)

    def _update_initializer_tensors(self, model: onnx.ModelProto) -> None:
        for initializer in model.graph.initializer:
            tensor_name = initializer.name
            shape = [dim for dim in initializer.dims]
            data_type_enum = initializer.data_type
            dtype = onnx.helper.tensor_dtype_to_np_dtype(data_type_enum)
            data = None
            if self._load_data:
                data = onnx.numpy_helper.to_array(initializer).copy()
            self._update_tensor(tensor_name, TensorType.INITIALIZER, dtype, shape, data=data)

    def _update_tensor(  # noqa: C901
        self,
        name: str,
        tensor_type: TensorType | None = None,
        dtype: np.dtype[Any] | None = None,
        shape: list[int] | None = None,
        producer: Node | None = None,
        consumers: list[Node] | None = None,
        data: NDArray[Union[np.float32, np.int8]] | None = None,
    ) -> Tensor:
        # Update if it already exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if name is not None:
                tensor.name = name
            if tensor_type is not None:
                tensor.tensor_type = tensor_type
            if dtype is not None:
                tensor.dtype = dtype
            if shape is not None:
                tensor.shape = shape
            if producer is not None:
                tensor.producer = producer
            if consumers is not None:
                tensor.consumers = consumers
            if data is not None:
                tensor.data = data
            if consumers is None:
                consumers = []
        else:
            # Create a new tensor if it does not yet exist
            if consumers is None:
                consumers = []
            tensor = Tensor(name, tensor_type, dtype, shape, producer, consumers, data)
            self.tensors[name] = tensor

        return tensor

    def _add_model_nodes(self, model: onnx.ModelProto) -> None:
        for onnx_node in model.graph.node:
            node_name = onnx_node.name
            if node_name in self.nodes:
                raise RuntimeError(f"ONNX graph must not contain duplicate node names: {node_name}")

            op_type = onnx_node.op_type
            attributes = self._get_node_attributes(model, onnx_node)
            node = Node(node_name, op_type, attributes=attributes)
            self.nodes[node_name] = node

            # Assign this node as consumer to input tensors
            for input_name in onnx_node.input:
                if input_name == "":  # An empty string indicates an unset optional input in ONNX
                    node.inputs.append(None)  # type: ignore
                else:
                    tensor = self._update_tensor(input_name)
                    tensor.consumers.append(node)
                    node.inputs.append(tensor)

            # Assign this node as producer to output tensors
            for output_name in onnx_node.output:
                tensor = self._update_tensor(output_name, TensorType.NODE_OUTPUT)
                tensor.producer = node
                node.outputs.append(tensor)

    def _get_node_attributes(self, model: onnx.ModelProto, onnx_node: onnx.NodeProto) -> dict[str, Any]:
        """Return all attributes of given ONNX node (explicitly set and default values)."""
        # Get explicitly set node attributes
        attributes = {}
        for attr in onnx_node.attribute:
            name = attr.name
            value = get_attribute_value(attr)
            attributes[name] = value

        # Retrieve default values
        # Pain in the ass.. we need to iterate over all opsets, domains and schemas, to find them reliably
        schema = None
        for opset in model.opset_import:
            domain = opset.domain
            version = opset.version
            try:
                # Try to get the schema for this op_type in the given domain
                schema = defs.get_schema(onnx_node.op_type, version, domain)
            except Exception:
                continue  # If schema lookup fails, continue searching
        if schema is None:
            logger.warning(f"Could not find ONNX schema for {onnx_node.op_type}. Attributes may not be complete.")
            # return {}
            # raise ValueError(f"Could not find schema for {onnx_node.op_type}")
        else:
            for attr in schema.attributes.values():
                if attr.default_value is not None and attr.name not in attributes:
                    attribute_value = get_attribute_value(attr.default_value)
                    attributes[attr.name] = attribute_value

        # Convert bytes to string
        for name, attr in attributes.items():
            if isinstance(attr, bytes):
                attributes[name] = attr.decode()
        return attributes

    def __repr__(self) -> str:
        """Return a string representation of this graph."""
        return "GraphHelper"

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the model.

        This function creates a custom state which is serializable by pickle. While pickle is usually quite capable
        of serialiazing custom objects like this, it sometimes has problems with self nested self-referencing objects.
        In this case, tensor and node objects referencing themselfes, sometimes cause recursion depth errors.

        This function circumvents this, by removing direct references and replacing them with string references.
        """
        state = self.__dict__.copy()
        nodes = state["nodes"]
        tensors = state["tensors"]

        # Replace actual objects with strings
        state["node_dicts"] = {}
        state["tensor_dicts"] = {}

        # Nodes: Track inputs and outputs via string references
        for name, node in nodes.items():
            node = node.__dict__.copy()
            node["inputs"] = [entry.name if entry is not None else None for entry in node["inputs"]]
            node["outputs"] = [entry.name for entry in node["outputs"]]
            state["node_dicts"][name] = node

        # Tensors: Track producers, consumers via string references
        for name, tensor in tensors.items():
            tensor = tensor.__dict__.copy()
            tensor["consumers"] = [entry.name for entry in tensor["consumers"]]
            tensor["producer"] = tensor["producer"].name if tensor["producer"] is not None else None
            state["tensor_dicts"][name] = tensor

        del state["nodes"]
        del state["tensors"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Undo the getstate function."""
        nodes = state["node_dicts"]
        tensors = state["tensor_dicts"]

        # Recreate node/tensor objects
        nodes = {name: Node(**node) for name, node in nodes.items()}
        tensors = {name: Tensor(**tensor) for name, tensor in tensors.items()}
        state["nodes"] = nodes
        state["tensors"] = tensors

        # Nodes: Reinsert actual tensor references
        for name, node in nodes.items():
            node.inputs = [tensors[tensor_name] if tensor_name is not None else None for tensor_name in node.inputs]
            node.outputs = [tensors[tensor_name] for tensor_name in node.outputs]

        # Tensors: Reinersert actual node references
        for name, tensor in tensors.items():
            tensor.consumers = [nodes[node_name] for node_name in tensor.consumers]
            tensor.producer = nodes[tensor.producer] if tensor.producer is not None else None
        self.__dict__.update(state)

    def __eq__(self, other: object) -> bool:
        """Compare two graph helper objects for equality.

        Args:
            other (object): Other graph helper object to compare to

        Returns:
            bool: Whether two objects represent the same onnx graph or not.

        Raises:
            ValueError: When other object is not of type ONNXGraphHelper
        """
        if not isinstance(other, ONNXGraphHelper):
            raise ValueError("Can only compare two ONNXGrapherHelper objects")

        for key in self.tensors.keys():
            if self.tensors[key] != other.tensors[key]:
                return False
        for key in self.nodes.keys():
            if self.nodes[key] != other.nodes[key]:
                return False
        return True
