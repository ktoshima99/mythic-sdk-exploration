from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from vnnmap.capnproto_interface import Layer, LayerType, Network, Tensor, TensorType
from vnnmap.utils import np_dtype_to_capnproto_dtype


class CapnprotoNetwork:
    """Helper class wrapping the Capnproto interface and providing convenience function to serialize a network.

    Note: Before adding a new layer, make sure to add all of the corresponding input and output tensors beforehand
    using .add_tensor()!

    Example:
        network = CapnprotoNetwork("test_network")
        network.add_tensor(...)
        network.add_tensor(...)
        network.add_layer(...)
        network.save("test_network.vidir")

    """

    def __init__(self, network_name: str, consume_data: bool = True) -> None:
        """Initialize the capnproto network.

        Args:
            network_name (str): Name of the network
            consume_data (bool, optional): Whether to consume the data provided in tensors or not.
                If True, the buffer of numpy arrays provided as data will be invalidated, right after the are added.
                Note that this probably makes these arrays unusuable afterwards. Defaults to True.
        """
        # Keep track of intermediate layers and tensors until final message is written
        self._layers: OrderedDict[str, Any] = OrderedDict()  # layer_name to Layer object
        self._tensors: OrderedDict[str, Any] = OrderedDict()  # tensor name to Tensor object

        # Keep track of layer inputs and outputs to be able to write tensor consumers and producers
        self._layer_inputs: dict[int, list[str]] = {}  # layer name to list of input tensor names
        self._layer_outputs: dict[int, list[str]] = {}  # layer name to list of output tensor names

        # Initialize main network message
        self._network = Network.new_message(name=network_name)
        self._network.name = network_name

        # Initialize the metainfo object with default values stored in schema
        # For some reason this is accomplished by accessing the members without assignment
        self._network.metaInfo.vidortCommit
        self._network.metaInfo.vidaimapCommit

        self._consume_data = consume_data

        # Fields for debugging data. Are set with add_debugging_fields()
        self._debug_floating_point_tensor_data: dict[str, NDArray[np.float32]] | None = None
        self._debug_quantized_floating_point_tensor_data: dict[str, NDArray[np.float32]] | None = None

    def save(self, file_path: str) -> None:
        """Serialize the model using capnproto and save it to file_path.

        Args:
            file_path (str): path to save the model

        Raises:
            ValueError: if no layers or tensors have been added.

        Returns:
            None: This method does not return a value.
        """
        if len(self._layers) == 0 or len(self._tensors) == 0:
            raise ValueError("You need to add at least one layer and tensor")

        self._write_input_output_tensor_info()
        self._write_tensor_consumer_producer_info()
        self._write_tensor_messages()
        self._write_layer_messages()

        # Write message
        with open(file_path, "w+b") as f:
            self._network.write(f)

    def add_model_flow_config(self, flow_configuration_string: str) -> None:
        """Add the model flow config string to the network MetaInfo.

        Args:
            flow_configuration_string (str): A string containing a representation of the vidort ModelFlowConfig used
                to create this model, which will be added to the MetaInfo object of the Network message.

        Returns:
            None: This method does not return a value.
        """
        self._network.metaInfo.vidortConfig = flow_configuration_string

    def add_debugging_fields(
        self,
        floating_point_tensor_data: dict[str, NDArray[np.float32]],
        quantized_floating_point_tensor_data: dict[str, NDArray[np.float32]],
    ) -> None:
        """Add debugging fields to the network message.

        Args:
            floating_point_tensor_data (dict[str, NDArray[np.float32]]): A dictionary mapping tensor names to their
                floating point data.
            quantized_floating_point_tensor_data (dict[str, NDArray[np.float32]]): A dictionary mapping tensor names to their
                quantized floating point data.

        Raises:
            ValueError: If a tensor name is not found in floating_point_tensor_data.

        Returns:
            None: This method does not return a value.
        """
        # Check that data entries match existing tensors
        for tensor_name in self._tensors.keys():
            if tensor_name not in floating_point_tensor_data.keys():
                raise ValueError(
                    f"Tensor {tensor_name} not found in floating point debug data. Cannot add debugging data"
                )
            if tensor_name not in quantized_floating_point_tensor_data.keys():
                # Some integer tensors do not have debugging data
                # In that case just add floating point data agin in that case
                quantized_floating_point_tensor_data[tensor_name] = floating_point_tensor_data[tensor_name]
        self._debug_floating_point_tensor_data = floating_point_tensor_data
        self._debug_quantized_floating_point_tensor_data = quantized_floating_point_tensor_data

    def _write_input_output_tensor_info(self) -> None:
        # Collect IDs of input and output tensors
        input_tensor_ids = []
        output_tensor_ids = []
        for index, tensor in enumerate(self._tensors.values()):
            if tensor.tensorType == TensorType.graphInput:
                input_tensor_ids.append(index)
            elif tensor.tensorType == TensorType.graphOutput:
                output_tensor_ids.append(index)

        # Write into network message
        self._network.init("inputs", len(input_tensor_ids))
        self._network.inputs = input_tensor_ids
        self._network.init("outputs", len(output_tensor_ids))
        self._network.outputs = output_tensor_ids

    def _write_tensor_consumer_producer_info(self) -> None:
        # Keep track of layer inputs
        tensor_list = [tensor for tensor in self._tensors.values()]
        tensor_consumers: dict[int, list[int]] = {tensor.idx: [] for tensor in tensor_list}
        for layer_id, input_tensor_names in self._layer_inputs.items():
            # Go over layer inputs and track which tensors are being used
            for input_tensor_name in input_tensor_names:
                input_tensor_id = self.get_tensor(input_tensor_name).idx
                tensor_consumers[input_tensor_id].append(layer_id)

        # Write tensor consumers into tensor message
        for tensor_idx, layer_consumer_ids in tensor_consumers.items():
            tensor_list[tensor_idx].init("consumers", len(layer_consumer_ids))
            tensor_list[tensor_idx].consumers = layer_consumer_ids

        # Go over layer outputs and write this layer as producer into tensor message
        for layer_id, output_tensor_names in self._layer_outputs.items():
            output_tensor_id = self.get_tensor(output_tensor_names[0]).idx
            tensor_list[output_tensor_id].producer = layer_id

    def _write_tensor_messages(self) -> None:
        # Add tensor message objects to network message
        self._network.init("tensors", len(self._tensors))
        tensors = [t for t in self._tensors.values()]
        tensors.sort(key=lambda t: t.idx)

        if self._debug_floating_point_tensor_data is not None:
            self._network.init("debugFloatingPointTensorData", len(tensors))
            self._network.init("debugQuantizedFloatingPointTensorData", len(tensors))

        for index, tensor in enumerate(tensors):
            self._network.tensors[index] = tensor

            tensor_name = tensor.name

            if self._debug_floating_point_tensor_data is not None:
                self._network.debugFloatingPointTensorData[index] = self._debug_floating_point_tensor_data[
                    tensor_name
                ].tobytes()
            if self._debug_quantized_floating_point_tensor_data is not None:
                self._network.debugQuantizedFloatingPointTensorData[index] = (
                    self._debug_quantized_floating_point_tensor_data[tensor_name].tobytes()
                )

    def _write_layer_messages(self) -> None:
        # Add layer message objects to network message
        self._network.init("layers", len(self._layers))
        layers = [t for t in self._layers.values()]
        layers.sort(key=lambda layer: layer.idx)
        for index, layer in enumerate(layers):
            self._network.layers[index] = layer

    def get_tensor(self, name: str) -> Any:
        """Return the Capnproto Tensor representation of an already added tensor given its name.

        Args:
            name (str): name of the tensor

        Raises:
            ValueError: if the tensor has not been added yet.

        Returns:
            Any: The capnproto tensor representation
        """
        if name not in self._tensors:
            raise ValueError(f"There is no tensor named {name}. Check that it was already added")
        return self._tensors[name]

    def get_layer(self, name: str) -> Any:
        """Return the Capnproto Layer representation of an already added layer given its name.

        Args:
            name (str): name of the layer

        Raises:
            ValueError: if the layer has not been added yet.

        Returns:
            Any: The capnproto layer representation
        """
        if name not in self._layers:
            raise ValueError(f"There is no layer named {name}. Check that it was already added")
        return self._layers[name]

    def add_tensor(  # noqa: C901
        self,
        name: str,
        tensor_type: TensorType,
        data: NDArray[Any] | None,
        fixed_point_data: NDArray[np.int16] | None,
        max_exponents: NDArray[np.int8] | None,
        adjusted_max_exponents: NDArray[np.int8] | None,
        shape: Sequence[int],
        n_bits: int | None,
        quant_axis: int | None,
    ) -> Tensor:
        """Add a new tensor to the network.

        Args:
            name (str): Name of the tensor
            tensor_type (TensorType): What kind of tensor this is.
            data (NDArray[Any] | None): A numpy array containing a single sample in fp32
            fixed_point_data (NDArray[np.int16] | None): A numpy array containing a single sample in int16
            max_exponents (NDArray[np.int8] | None): A numpy array containing the max exponents if this tensor
            adjusted_max_exponents (NDArray[np.int8] | None): A numpy array containing the adjusted
                max exponents (debugging field)
            shape (Sequence[int]): Shape of the tensor
            n_bits (int | None): N bits used to quantize this tensor
            quant_axis (int | None): The axis along which the tensor is quantized with max_exponents

        Raises:
            ValueError: Value error if the tensor was already added

        Returns:
            Tensor: The capnproto representation of this tensor
        """
        # There must not be a tensor present with the same name
        if name in self._tensors:
            raise ValueError(f"A tensor with name {name} was already added.")

        tensor_idx = len(self._tensors)
        tensor = Tensor.new_message(
            idx=tensor_idx,
            name=name,
            tensorType=tensor_type,
            shape=shape,
        )
        if n_bits is not None:
            tensor.nBits = n_bits
        if quant_axis is not None:
            tensor.quantAxis = quant_axis

        if max_exponents is not None:
            max_exponents_bytes = max_exponents.tobytes()
            tensor.maxExponents = max_exponents_bytes
            del max_exponents_bytes

        if adjusted_max_exponents is not None:
            adjusted_max_exponents_bytes = adjusted_max_exponents.tobytes()
            tensor.adjustedMaxExponents = adjusted_max_exponents_bytes
            del adjusted_max_exponents_bytes

        # These are optional and only apply only for static tensors
        if data is not None:
            data_bytes = data.tobytes()
            tensor.data = data_bytes
            data_type = np_dtype_to_capnproto_dtype(data.dtype)
            tensor.dtype = data_type
        if fixed_point_data is not None:
            fixed_point_data_bytes = fixed_point_data.tobytes()
            tensor.fixedPointData = fixed_point_data_bytes
            del fixed_point_data_bytes
        # Clear data from numpy buffers to reduce memory load
        if self._consume_data:
            # Use resize(0), to reduce effective memory size to zero
            if max_exponents is not None:
                max_exponents.resize(0, refcheck=False)
            if fixed_point_data is not None:
                fixed_point_data.resize(0, refcheck=False)
            if data is not None:
                data.resize(0, refcheck=False)

        self._tensors[name] = tensor
        return tensor

    def add_layer(
        self,
        name: str,
        layer_type: LayerType,
        inputs: list[str],
        outputs: list[str],
        attributes: Any,
    ) -> Layer:
        """Add a new layer to the network.

        Args:
            name (str): Name of the layer
            layer_type (LayerType): Type of this layer
            inputs (list[str]): List of tensor names, which are the inputs of this layer
            outputs (list[str]): List of tensor names, which are the outputs of this layer
            attributes (Any): An instance of a capnproto Attribute object, containing the
                attributes of this layer. For example, if the layertype is conv, this should be a
                ConvAttributes object.

        Raises:
            ValueError: If not all required tensors have been added

        Returns:
            Layer: The created layer
        """
        # There must not be a layer present with the same name
        if name in self._layers:
            raise ValueError(f"A layer with name {name} was already added.")

        # Check that input tensors have been added
        if not all(t in self._tensors for t in inputs):
            raise ValueError(f"Not all input tensors have been added for layer {name}")
        if not all(t in self._tensors for t in outputs):
            raise ValueError(f"Not all output tensors have been added for layer {name}")

        layer_idx = len(self._layers)
        layer = Layer.new_message(idx=layer_idx, name=name, layerType=layer_type)

        # Keep track of layer inputs and outputs to be able to write tensor consumers and producers
        self._layer_inputs[layer_idx] = inputs
        self._layer_outputs[layer_idx] = outputs

        # Initialize the attribute union field
        attribute_schema_name = attributes.schema.node.displayName.split(":")[-1].split("Attributes")[0].lower()
        layer.attributes.init(attribute_schema_name)
        setattr(layer.attributes, attribute_schema_name, attributes)

        self._layers[name] = layer
        return layer
