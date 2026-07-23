import contextlib
import dataclasses
import inspect
import logging
import re
from typing import List, TypedDict
import warnings
import numpy as np
import torch
import torch.utils.checkpoint as torch_checkpoint
from itertools import chain
from funcy import some, omit, walk_values

from munc import _naming_utils, _node_utils, _o2t_ops
from munc._constants import (MMA_TYPE, MUNC_INTERNAL_ONNX_OPSET, NODE_OPSET_ATTR_NAME)

logger = logging.getLogger(__name__)


def print_hook(name):
    """Print hook used for debugging."""
    def hook(grad):
        logger.debug(name + ' max=' + str(np.max(grad.numpy())))

    return hook


@dataclasses.dataclass
class LayerInfo:
    """Layer information that is used during TorchNet inference."""

    name: str
    input: List[str]
    output: List[str]


class ValueGenerator:
    """Marker base class for config values that are generated at layer creation time.

    A config entry whose value is a ValueGenerator instance will be replaced by calling
    ``value(node, torchnet)`` inside :func:`default_layer_factory` before the layer is
    constructed.  Subclasses must implement ``__call__(self, node, torchnet)``.
    """

    def __call__(self, node, torchnet):  # noqa: D102
        raise NotImplementedError


class DeduplicatedValueGenerator(ValueGenerator):
    """A ValueGenerator that creates at most one value per key.

    Values are stored in ``torchnet.value_registry`` (a ``torch.nn.ModuleDict``).
    If the registry already contains an entry for the key returned by ``key(node)``,
    that entry is returned unchanged; otherwise ``create(node, torchnet)`` is called,
    its result is stored, and then returned.

    Subclasses must implement :meth:`key` and :meth:`create`.
    """

    def key(self, node):
        """Return the value registry key.

        Parameters
        ----------
        node : onnx.NodeProto
            ONNX node being converted. Its attributes can be used to generate a key.

        Returns
        -------
        str
            String key that identifies the desired shared value.
        """
        raise NotImplementedError

    def create(self, node, torchnet):
        """Create and return a new value (``torch.nn.Module``).

        Parameters
        ----------
        node : onnx.NodeProto
            ONNX node being converted.
        torchnet : TorchNet
            TorchNet instance under construction.

        Returns
        -------
        torch.nn.Module
            The new module to register and return.
        """
        raise NotImplementedError

    def __call__(self, node, torchnet):  # noqa: D102
        k = self.key(node)
        if k not in torchnet.value_registry:
            torchnet.value_registry[k] = self.create(node, torchnet)
        return torchnet.value_registry[k]


class StaticKeyDeduplicatedValueGenerator(DeduplicatedValueGenerator):
    """A DeduplicatedValueGenerator with a fixed key and a provided create callable.

    Parameters
    ----------
    key : any
        Registry key used for all nodes.  Converted to a string via ``str()``
        for use as a ``ModuleDict`` key.  Any value that produces a unique
        string representation is valid (e.g. a tuple, list, or plain string).
    create : Callable[[onnx.NodeProto, TorchNet], torch.nn.Module]
        Callable invoked the first time the key is not found in the registry.
        Receives the same ``(node, torchnet)`` arguments as
        :meth:`DeduplicatedValueGenerator.create`.
    """

    def __init__(self, key, create):
        self._key = key
        self._create = create
        _positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        # Check if `create` takes `node` and `torchnet`.
        self._pass_args = any(p.kind in _positional and p.default is inspect.Parameter.empty
                              for p in inspect.signature(create).parameters.values())

    def key(self, node):  # noqa: D102
        return str(self._key)

    def create(self, node, torchnet):  # noqa: D102
        return self._create(node, torchnet) if self._pass_args else self._create()


def default_layer_factory(node, torchnet, configs=()):
    """Create a torch layer representing ONNX `node`.

    The layer class is determined by the `op_type` attribute of the node. It can be overridden by providing an
    `o2t_op` key in a configuration.

    Parameters
    ----------
    node : onnx.NodeProto
        ONNX node to convert to a torch layer.
    torchnet : munc._torchnet.TorchNet
        TorchNet object to which the layer belongs.
    configs : List[Callable[[onnx.NodeProto], Union[Dict,None]]], optional
        List of layer configuration functions. Each function takes a node and returns a non-empty
        dictionary of parameters to configure a corresponding layer or an empty dictionary if it does not want
        to handle the node. The result of the first function that returns a non-empty dictionary will be used.
        It will be passed as `kwargs` to the layer's `__init__`. A value of the `o2t_op` key will be used as the layer
        class name if specified.

    Returns
    -------
    torch.nn.Module
        Torch layer representing the ONNX node.
    """
    # Call each config function on the node to create a list of config dictionaries.
    config_dicts = (config_func(node) for config_func in configs)
    # Take the first non-empty config dictionary as our configuration.
    config = some(config_dicts) or {}
    # Merge all the applicable configuration dictionaries. This may be better, we will see.
    # config = join(config_dicts) or {}
    if config:
        logger.debug(f'{node.name}: configuration: {config}')
    o2t_op = config.get("o2t_op", node.op_type)
    try:
        layer_class = getattr(_o2t_ops, o2t_op) if isinstance(o2t_op, str) else o2t_op
    except AttributeError:
        raise Exception(f'Op type "{o2t_op}" is not supported by onnx2torch session.')

    # Create layer
    params = omit(config, ["o2t_op"])
    params = walk_values(lambda v: v(node, torchnet) if isinstance(v, ValueGenerator) else v, params)
    return layer_class(node, torchnet, **params)


class ActivationCkptConfig(TypedDict, total=False):
    """Config for per-layer activation checkpointing (trades extra compute for less memory during training).

    TorchNet must be in training mode (`.train()`) for checkpointing to take effect

    The `pattern` regex is matched against `"ClassName::layer_name"`, which allows
    selection by node type (as o2t op class name), layer name, or both, e.g:

        'MythicConv2d::'                       # all MythicConv2d layers (class anchor)
        '::.*layer[12]_'                       # any layer whose name contains layer1_ or layer2_
        'MythicConv2d::(conv1|.*layer1_)'      # combined

    Warning — stateful forward passes
    --------------------------------
    `torch.utils.checkpoint.checkpoint` re-executes the wrapped forward during backward.
    Only RNG state is preserved automatically; any other in-forward state mutation
    (buffer updates, EMAs, counters) will fire twice per training step and corrupt that
    state. Known risks in this codebase:
      - `MythicMMA` (parent of `MythicConv2d`/`MythicLinear`/`MythicSum`) updates
        `self.weight_scale` via momentum when `compute_weight_scale` is set. Safe when
        `compute_weight_scale=None` (the default); otherwise the EMA will double-update.
      - `BCMConv2d`/`BCMLinear` call `self.mma.randomize()` per forward and will
        re-randomize on recompute.
    Verify training convergence after enabling. With a `null` pattern (match all),
    be especially careful — prefer a targeted pattern that scopes checkpointing to
    layers verified to be recompute-safe.
    """

    enable: bool
    pattern: str | None  # None or omitted -> match all layers


class TorchNet(torch.nn.Module):
    """Class to convert from ONNX to PyTorch."""

    def __init__(self,
                 model,
                 device_name=None,
                 debug_hooks=False,
                 debug_devices=False,
                 debug_parameters=False,
                 debug_forward=False,
                 store_model=False,
                 layer_factory=None,
                 activation_ckpt_config=None,
                 ):
        """Create an executable Pytorch model for an ONNX model.

        Parameters
        ----------
        model : ONNXModel
            An ONNX model to be converted to Pytorch.
        layer_factory : Optional[Callable[[onnx.NodeProto, TorchNet], torch.nn.Module]]
            A layer (nn.Module) factory. It's called for every node and returns a torch module that represents
            the node. By default `default_layer_factory` is used.
        activation_ckpt_config : ActivationCkptConfig, optional
            Per-layer activation checkpointing config. Passed through to TorchNet.

        Returns
        -------
        TorchNet
            a TorchNet instance representing the ONNX model.
        """
        super(TorchNet, self).__init__()

        assert model.get_opset_version() == MUNC_INTERNAL_ONNX_OPSET

        if layer_factory is None:
            layer_factory = default_layer_factory

        self._model = model
        self.set_edge_hooks = []

        self.external_input_names = model.get_external_input_names()
        self.output_names = model.get_output_names()
        # Determine desired device if not provided
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(device_name)
        self.register_buffer('_device_tensor', torch.tensor(0.0, device=device))

        nodes = model.get_nodes()

        def make_layer_info(node):
            return LayerInfo(_generate_pytorch_layer_name(model, node),
                             list(map(_naming_utils.name2attr, node.input)),
                             list(map(_naming_utils.name2attr, node.output)))

        # Create the LayerInfo for all nodes in the graph.
        self.layers = list(map(make_layer_info, model.get_nodes()))

        # Create edges, this must be a list and not a set
        edges = list(model.get_input_names())
        self.precomputed_get_parameter_group_dict = {}
        for node in nodes:
            node_input = node.input
            is_trainable_value_list = _node_utils.fetch_is_trainable_value_list(node)

            if is_trainable_value_list is not None:
                # we need to remove all edges that are not coming from initializers
                def get_initializer_edge(edge):
                    return model.find_initializer(edge) is not None and edge
                initializer_edges = list(filter(None, map(get_initializer_edge, node_input)))
                assert len(initializer_edges) == len(is_trainable_value_list), 'size mismatch'
                # choose node trainable edges by the mask
                self.precomputed_get_parameter_group_dict.update(zip(initializer_edges, is_trainable_value_list))
            elif _node_utils.get_attribute_value(node, _node_utils.NO_GRADIENT_ATTRIBUTE_NAME, False) == "":
                # Mark all the edges as not trainable
                for inp in node_input:
                    self.precomputed_get_parameter_group_dict[inp] = 0
            edges += node.input
            edges += node.output
        edges = np.unique(edges)

        self.attr2edge = {_naming_utils.name2attr(edge): edge for edge in edges}

        # Assign initializers
        for edge in edges:
            # Fetch initializer (may be None)
            initializer = model.get_initializer_np(edge)

            att_name = _naming_utils.name2attr(edge)
            # Convert to torch tensor, if a tensor
            if initializer is not None:
                initializer = torch.tensor(initializer).to(device)

                if self._get_parameter_group(model, edge):
                    initializer = torch.nn.Parameter(initializer)

                    if debug_parameters:
                        logger.debug(f'Parameter: {edge}')
                else:
                    self.register_buffer(att_name, initializer)
                if debug_hooks:
                    if initializer.requires_grad:
                        initializer.register_hook(print_hook(edge))

            # Set attribute
            setattr(self, att_name, initializer)

            # Diagnostics
            if debug_devices:
                if initializer is not None:
                    logger.debug(f"{att_name} device: {initializer.device.type}")

        # Get list of initializers
        initializer_names = model.get_initializer_names()

        # Get a list of all node inputs and an index to iterate through this list.
        all_node_inputs = [input_ for node in nodes for input_ in node.input]
        input_index = 0

        # Registry for values shared across layers (used by DeduplicatedValueGenerator).
        self.value_registry = torch.nn.ModuleDict()

        # Create nodes
        self._delete_after_update = {}
        outputs_to_keep = set(initializer_names) | set(self.output_names)
        for node in nodes:
            layer = layer_factory(node, self)
            layer = layer.to(self.device)
            self._check_node_opset_is_supported(node, layer)

            # Store the layer
            layer_name = _generate_pytorch_layer_name(model, node)
            setattr(self, layer_name, layer)

            # Future inputs (for updating the delete set)
            input_index += len(node.input)
            future_inputs = all_node_inputs[input_index:]

            # Update delete set
            self._delete_after_update[layer_name] = set(map(_naming_utils.name2attr,
                                                            set(node.input) - outputs_to_keep - set(future_inputs)))

        # Move registry modules to the same device as the rest of the network.
        self.value_registry.to(self.device)

        # Save model
        if not store_model:
            self._model = None

        # Execute inference once to generate layers
        self.forward_random_inputs(model, debug=debug_forward)

        # Build set of names of layers to apply activation checkpointing to
        self._checkpoint_layer_names: set[str] = self._compute_checkpoint_layer_names(
            activation_ckpt_config
        )

    @property
    def device(self):
        """Get the device of the module."""
        return self._device_tensor.device

    def _check_node_opset_is_supported(self, node, layer):
        layer_class = type(layer)
        node_opset = _node_utils.get_attribute_value(node, NODE_OPSET_ATTR_NAME)
        supported_opsets = hasattr(layer_class, "_supported_opsets") and layer_class._supported_opsets
        supported = (
            # No special support required
            node_opset is None
            # All opsets are supported
            or supported_opsets is True
            # The node opset is supported
            or (supported_opsets is not False and node_opset in supported_opsets))
        assert supported, f"{layer_class.__name__} does not support opset {node_opset}"

    def forward(self, *x_arr, delete_unused_edges=True, debug=False, debug_hooks=False, debug_devices=False):
        """Forward pass solver."""
        for module in self.value_registry.values():
            if getattr(module, 'auto_step', False):
                module.step()

        edge_values = {}

        def get_edge(name):
            return edge_values[name] if name in edge_values else getattr(self, name)

        def set_edge(name, value):
            edge_values[name] = value
            edge_name = self.attr2edge[name]
            for hook in self.set_edge_hooks:
                hook(edge_name, value)

        def del_edge(name):
            if name in edge_values:
                del edge_values[name]

        # Assign input names (dict input)
        if len(x_arr) == 1 and isinstance(x_arr[0], dict):
            x = x_arr[0]
        # (tuple input)
        else:
            # Check for common errors
            if any([isinstance(x_arr[i], np.ndarray) for i, _ in enumerate(self.external_input_names)]):
                raise Exception('Input to onnx2torch object is numpy array. It needed to be tensor.')
            # Create dictionary of inputs
            x = {external_input_name: x_arr[i] for i, external_input_name in enumerate(self.external_input_names)}

        # Assign inputs
        for key in x:
            # Fetch name of attribute
            input_attr_name = _naming_utils.name2attr(key)

            # Set attribute from user provided inputs
            x_key = x[key].to(self.device)
            set_edge(input_attr_name, x_key)

        # Loop over nodes and create computation
        for layer in self.layers:
            # Fetch layer
            layer_fcn = getattr(self, layer.name)

            # Print debug info
            if debug:
                logger.debug(f'Layer name: {layer.name}')

            # Inputs list
            inputs_list = list(map(get_edge, layer.input))

            # Print debug info
            if debug:
                inputs_joined = ", ".join(layer.input)
                input_sizes_joined = ", ".join([str(input_.shape) for input_ in inputs_list])

                logger.debug(f'Inputs are: {inputs_joined}')
                logger.debug(f'Input sizes are: {input_sizes_joined}')

            # Verify inputs
            for input_val, input_name in zip(inputs_list, layer.input):
                if input_val is None and input_name != _naming_utils.name2attr(''):
                    raise Exception(f"Input {input_name} is None.")

            # Print debug info
            if debug:
                output_names_joined = ", ".join(layer.output)
                logger.debug(f'Output names are: {output_names_joined}')

            # Create layer, if layer comes with a generator
            if hasattr(layer_fcn, 'layer_gen'):
                if getattr(layer_fcn, 'layer') is None:  # if not created yet
                    layer_gen = getattr(layer_fcn, 'layer_gen')
                    layer_gen(*inputs_list)
                    layer_fcn = layer_fcn.to(self.device)

            # Eval
            if self.training and layer.name in self._checkpoint_layer_names:
                result = torch_checkpoint.checkpoint(layer_fcn, *inputs_list, use_reentrant=False)
            else:
                result = layer_fcn(*inputs_list)

            # Set output
            if type(result) is list:  # list output
                for i_output in range(len(layer.output)):
                    set_edge(layer.output[i_output], result[i_output])
                    if debug_devices:
                        logger.debug(f"{layer.output[i_output]} device: {result[i_output].device.type}")
                    if debug_hooks:
                        result[i_output].register_hook(print_hook(layer.output[i_output]))

            else:  # singular output
                assert (len(layer.output) == 1)
                output_name = layer.output[0]
                set_edge(output_name, result)
                if debug_devices:
                    logger.debug(f"{output_name} device: {result.device.type}")
                if debug_hooks:
                    if result.requires_grad:
                        result.register_hook(print_hook(output_name))

            # Delete edges that will not be used in the future to save up memory
            if delete_unused_edges:
                for edge in self._delete_after_update[layer.name]:
                    del_edge(edge)

            # Diagnostics
            if debug:
                logger.debug(' ')

        # Concatenate results
        outputs = []
        for output_name in self.output_names:
            att_name = _naming_utils.name2attr(output_name)
            outputs.append(get_edge(att_name))

        # No list for singular output
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def get_hardware_models(self, layer_names=None):
        """Return hardware models of the specified layers.

        Parameters
        ----------
        layer_names : Optional[List[str]]
            List of the name of layers to fetch hardware models from. If None (default), all layers will be queried.

        Returns
        -------
        Dict[str, BaseAnalogModel]
            A dictionary of layer names to their hardware models.
        """
        layer_names = ([layer.name for layer in self.layers if hasattr(getattr(self, layer.name), "analog_model")]
                       if layer_names is None else layer_names)
        return {layer_name: getattr(self, layer_name).analog_model for layer_name in layer_names}

    @contextlib.contextmanager
    def register_forward_debug(self, layer_names=None):
        """Register the forward debugging interface as a context manager.

        This is used by debug and visualization scripts to return the intermediate tensors of our hardware models.
        This is enabled for the duration of the context manager afterwhich the context manager is removed.

        Parameters
        ----------
        layer_names : str, or list, optional
            List of the name of layers to update.
            If set to None (default), all layers with register_forward_debug will be enabled.
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        if layer_names is None:
            layer_names = [layer.name for layer in self.layers]

        debug_container = {}
        for layer_name in layer_names:
            # Fetch layer
            layer_fcn = getattr(self, layer_name)

            if hasattr(layer_fcn, "register_forward_debug"):
                layer_container = layer_fcn.register_forward_debug()
                debug_container[layer_name] = layer_container
            else:
                logger.debug(f"Layer {layer_name} does not have register_forward_debug()")

        yield debug_container

        # Unregister the debug hooks
        for layer_name in layer_names:
            # Fetch layer
            layer_fcn = getattr(self, layer_name)

            if hasattr(layer_fcn, "unregister_forward_debug"):
                layer_fcn.unregister_forward_debug()
            else:
                logger.debug(f"Layer {layer_name} does not have unregister_forward_debug()")

    def save_torch_to_onnx_object(self, model=None):
        """Save all torch data into the underlying ONNX object."""
        # Fetch nodes of the model
        if model is None:
            warnings.warn("Calling save_torch_to_onnx_object without an ONNX model is deprecated",
                          category=DeprecationWarning)
            model = self._model
        nodes = model.get_nodes()

        # Save node info
        count_saved_parameters = 0
        for node in nodes:
            for edge in node.input:
                if self._get_parameter_group(model, edge):
                    # Get torch data
                    att_name = _naming_utils.name2attr(edge)
                    tensor = getattr(self, att_name)

                    # Set ONNX data with Torch data
                    model.set_initializer_np(edge, tensor.cpu().detach().numpy())
                    count_saved_parameters += 1

            # update the onnx node if any values have changed during training
            count_saved_parameters += self.update_node(model, node)

        # Check if all trained parameters are saved
        count_torch_parameters = len(list(self.parameters()))
        if count_torch_parameters != count_saved_parameters:
            raise Exception(f"Missed parameters during save. Only {count_saved_parameters} out of"
                            f" {count_torch_parameters} torch parameters saved.")

    def update_node(self, model, node):
        """Update onnx node with information from _pytorch layer.

        If attribute or initializer values have changed during training, update the onnx values to reflect this.

        A pytorch layer that needs to write updated values back to its ONNX node may provide the following methods:
            update_back_to_onnx()
            update_parameters_back_to_onnx(model, node)
        The first one should return a dictionary of attribute names and their new values. This function will save the
        new values to the corresponding attributes.
        The second method is for more general cases. It should save updated values to ONNX attributes and initializers,
        and return the number of saved parameters.

        Parameters
        ----------
        net : munc._torchnet.TorchNet
            Network from which updated attributes will be copied.
        node : onnx.NodeProto
            Node to update attributes with pytorch layer information.

        Returns
        -------
        int
            The number of saved parameters.
        """
        layer_name = _generate_pytorch_layer_name(model, node)
        layer = getattr(self, layer_name)

        if hasattr(layer, 'update_back_to_onnx'):
            for attr, value in layer.update_back_to_onnx().items():
                _node_utils.set_attribute_value(node, attr, value, create=True)

        if hasattr(layer, 'update_parameters_back_to_onnx'):
            return layer.update_parameters_back_to_onnx(model, node)
        else:
            return 0

    def _get_parameter_group(self, model, edge):
        """Check nodes to see if this edge connects any Gemm or Conv node to its weights or bias.

        In onnx, a Gemm or Conv node has a field input, e.g., by name of the input
            node.input = [new283, conv1_weight, new1]
        where the order is [input, weight, bias], i.e. node.input[0] is input to the node (activations or input)
        node.input[1] is the weight and node.input[2] is the bias.
        The elements in node.input are the id's the edge, but am using the names in this explanation.

        1) The algorithm retrieves all nodes that have this edge as an input, i.e.,
        edge appears in node.input.

        2) Loop over these nodes and at first instance of when 'edge' in position 1 or 2
            of node.input and if the node is a Gemm or Conv it returns True

        4) If edge not a wnb and was not used as input either, it may be a constant-edge or an initializer-edge
            but it is not a wnb edge for sure.

        5) If the edge is an input (position 0 in node.input), then check if the output of the node
            is a connection to a weight or bias via a recursive call on the outputs (edge) of every
            node that has the output as input.

            Make a recursive call to check if any of its outputs are a weight or bias to a node.
            Note: This can only be true if there are ops that modify the weights or bias.
            which is typically true for the work in MUNC models.

        Parameters
        ----------
        edge : str
            An edge name representing an input into a onnx node.

        Returns
        -------
        bool
            Zero if the edge is not a model parameter, otherwise a parameter group number (greater than zero).
            1 is for weights, 2 is for bias.

        Note
        ----
        - Makes the assumption an edge is a weight or bias input if it is an
        input to an mma operation and it is not the first input
        - An explicit check is added to see if the edge is for the global chip
        temperature
        """
        if edge in self.precomputed_get_parameter_group_dict:
            return self.precomputed_get_parameter_group_dict[edge]
        # Get output nodes of edge
        nodes = model.get_nodes_with_input_name(edge)

        # For any of those nodes is a conv or gemm
        at_least_one_first_input = False
        for node in nodes:
            # Get the edge's position in the list node.input
            i_edge = list(node.input).index(edge)

            # Keep track of whether this edge is the first input to at least one node
            # i.e., if edge is a image input or activation to at least one node
            if i_edge == 0:
                at_least_one_first_input = True

            # Check if the node is a Gemm or Conv op
            if node.op_type in MMA_TYPE:
                # If this edge is a weight of bias, return true
                if 0 < i_edge < 3:
                    self.precomputed_get_parameter_group_dict[edge] = i_edge
                    return i_edge

        # At this point, the edge was not associated with a node as weight or bias edge
        if not at_least_one_first_input:
            # and edge not associated as an input either
            # If no nodes that with this edge as first input, fail
            self.precomputed_get_parameter_group_dict[edge] = 0
            return 0

        # Check the edge outputs of nodes
        output_parameter_groups = (self._get_parameter_group(model, output_edge)
                                   for node in nodes for output_edge in node.output)
        result = next((group for group in output_parameter_groups if group > 0), 0)
        self.precomputed_get_parameter_group_dict[edge] = result
        return result

    def get_group_parameters(self, group):
        """Return model parameters that are members of `group`.

        Parameters
        ----------
        group : int
          A parameter group number. Currently 3 groups are defined 1 - weights, 2 - biases, 3 - DSFs.

        Returns : list[nn.Parameter]
          A list of model parameters that are members of `group`.
        """
        assert group > 0

        def get_parameter(name):
            return getattr(self, _naming_utils.name2attr(name))

        def get_layer_internal_parameters(layer):
            layer_fcn = getattr(self, layer.name)
            return layer_fcn.get_group_parameters(group) if hasattr(layer_fcn, "get_group_parameters") else []

        internal_params = list(chain.from_iterable(map(get_layer_internal_parameters, self.layers)))
        initializer_params = [get_parameter(p) for p, g in self.precomputed_get_parameter_group_dict.items()
                              if g == group]
        return initializer_params + internal_params

    def forward_random_inputs(self, model, **kargs):
        """Initialize the torch model layers by invoking forward function on dummy input.

        Parameters
        ----------
        model : ONNXModel
            The onnx model used to create this TorchNet instance.

        Returns
        -------
        Tensor
            Output prediction from dummy input.
        """
        with torch.no_grad():
            # Create dummy input data
            dummy_inputs_np = model.create_dummy_inputs()
            dummy_inputs_torch = [{k: torch.as_tensor(v).to(self.device) for k, v in dummy_inputs_np.items()}]

            # Execute inference
            self.eval()
            outputs = self.forward(*dummy_inputs_torch, **kargs)

        return outputs

    def register_set_edge_hook(self, hook):
        """Register a hook to be called when an edge value is set."""
        if hook not in self.set_edge_hooks:
            self.set_edge_hooks.append(hook)

    def unregister_set_edge_hook(self, hook):
        """Unregister a hook to be called when an edge is value set."""
        if hook in self.set_edge_hooks:
            self.set_edge_hooks.remove(hook)

    @contextlib.contextmanager
    def set_edge_hook(self, hook):
        """Register a set-edge hook for the duration of the context."""
        self.register_set_edge_hook(hook)
        try:
            yield
        finally:
            self.unregister_set_edge_hook(hook)

    def get_layer(self, layer_name):
        """Get a torch Module representing a layer by its ONNX node name."""
        return getattr(self, layer_name)

    def get_initializer_value(self, name):
        """Get the value of a torch parameter representing an initializer."""
        return getattr(self, _naming_utils.name2attr(name))

    def predict_np(self, *x_arr):
        """Make inference using numpy arrays.

        Parameters
        ----------
        x_arr : list of numpy.ndarray
            Batch of data.

        Returns
        -------
        numpy.ndarray or list of numpy.ndarray
            Resulting model predictions.
        """
        x_arr_torch = [torch.from_numpy(x).to(device=self.device) if isinstance(x, np.ndarray) else x for x in x_arr]
        res_torch = self(*x_arr_torch)
        if isinstance(res_torch, torch.Tensor):
            return res_torch.detach().cpu().numpy()
        else:
            return [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in res_torch]

    def _compute_checkpoint_layer_names(
        self, config: ActivationCkptConfig | None
    ) -> set[str]:
        """Return the set of layer names that should use activation checkpointing.

        Called once at the end of `__init__`, after `forward_random_inputs` has
        materialized any lazy `layer_gen` modules so `type(layer_fcn)` is stable.
        The regex in `config["pattern"]` is matched against
        `"ClassName::layer_name"` — see `ActivationCkptConfig` for examples.

        See https://docs.pytorch.org/docs/stable/checkpoint.html

        Parameters
        ----------
        config
            Layer selection activation config
        """
        if not config or not config.get("enable"):
            return set()
        config_pattern = config.get("pattern")
        pattern = re.compile(config_pattern) if config_pattern else None
        names: set[str] = set()
        for layer in self.layers:
            qualified = f"{type(getattr(self, layer.name)).__name__}::{layer.name}"
            if pattern is None or pattern.search(qualified):
                names.add(layer.name)
        return names


def _generate_pytorch_layer_name(model, node):
    return node.name or f'node{model.get_node_index(node)}'
