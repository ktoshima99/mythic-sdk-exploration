# This file is distributed under the terms of Mythic Inc's Software Licence Agreement
# Copyright (C) 2021, Mythic Inc. All rights reserved.
#
"""
A little library for creating L0 crates from Python.
"""

from collections import defaultdict
import operator

from collections.abc import Sequence
from functools import reduce
import typing

import numpy as np

from mythic.target_spec.resources_pb2 import BitSpreadingMode, SaluParameters
from mythic.target_spec.target_pb2 import SaluSpec, TileId

from mythic.irs.l0.ir_pb2 import (BaseLauncher, Buffer, Cluster, Copy, Crate, DbgDump, Infeed, Launcher, MmaDot,
                                  Outfeed, Pad, SaluCommand)
from mythic.irs.l0.shape_pb2 import IterationSpec
from mythic.irs.l0.vector_processing_pb2 import Precision, InStreamSpec, InStreamParameters, InPlaceSpec, InPlaceParameters

Shape = typing.Sequence[int]


def flat_size(shape: Shape) -> int:
    """
    Size of the `Shape` if it were flattened to be 1-dimensional.
    """

    return reduce(operator.mul, shape, 1)


def iteration_domain(num_iterations: Shape, filter: Shape, stride: Shape) -> Shape:
    """
    Get the domain `Shape` for a given number of iterations, filter, and stride.
    """

    return [(i - 1) * s + f for i, f, s in zip(num_iterations, filter, stride)]


def tight_stride(domain: Shape, filter: Shape, stride: Shape) -> Shape:
    """
    Get a tightened stride `Shape` for a given domain, filter, and stride.
    """

    return [f if (s > 0 and d == f) else s for d, f, s in zip(domain, filter, stride)]


def domain_iterations(domain: Shape, filter: Shape, stride: Shape) -> Shape:
    """
    Get the number of iterations `Shape` for a given domain, filter, and stride.
    """
    assert all(s > 0 for s in stride), 'Cannot compute number of iterations for stride of 0!'

    return [(d - f) // s + 1 for d, f, s in zip(domain, filter, stride)]


def line_buffer_size(spec: IterationSpec) -> int:
    """
    Minimal size needed for a `Buffer` to fit a `IterationSpec`.
    """
    shape = np.array(spec.buffer_shape_view.dims)
    filter_ = np.array(spec.filter.dims) - np.ones(len(spec.filter.dims), dtype=int)

    assert all(filter_ <= shape), f'IterationSpec filter does not fit into its BufferShapeView!\n\t{spec}'

    area = 1
    size = filter_[-1]
    for shape_dim, filter_dim in zip(reversed(shape[1:]), reversed(filter_[:-1])):
        area *= shape_dim
        size += filter_dim * area

    return size + 1


def _set_tile_id(protobuf, tile_id):
    if tile_id is None:
        return

    if isinstance(tile_id, Sequence):
        protobuf.tile_id.y = tile_id[0]
        protobuf.tile_id.x = tile_id[1]
        if len(tile_id) > 2:
            protobuf.tile_id.type = tile_id[2]
    else:
        protobuf.tile_id.CopyFrom(tile_id)


def build_iteration_spec(*,
                         domain: Shape,
                         buffer_shape_view: Shape | None = None,
                         offset: Shape | None = None,
                         filter: Shape | None = None,
                         stride: Shape | None = None,
                         wq_iterations: Shape | None = None,
                         wq_sub_filter: Shape | None = None,
                         wq_sub_stride: Shape | None = None,
                         sub_iterations: Shape | None = None,
                         sub_stride: Shape | None = None,
                         dim_order: Shape | None = None,
                         stay_count: int = 1) -> IterationSpec:
    """
    Create a Protobuf `IterationSpec`.

    The logic here mirrors that found in mythic::irs::l0::IterationSpec::Builder::Build().

    TODO: It would be better if this could somehow be directly implemented as calls to
    mythic::irs::l0::IterationSpec::Builder or if both could reference the same underlying
    implementation to avoid the possibility of them getting out of sync.
    """
    has_filter = filter is not None
    has_wq_loops = wq_iterations is not None
    has_sub_iterations = sub_iterations is not None
    assert has_filter or has_wq_loops or has_sub_iterations, 'At least one of filter, WQ loop, or sub-filter parameters must be specified!'
    assert not (has_filter and has_wq_loops), 'Filter and WQ loop parameters cannot both be specified!'
    assert not (has_filter and has_sub_iterations), 'Filter and sub-filter parameters cannot both be specified!'
    assert not (has_wq_loops and has_sub_iterations), 'WQ loop and sub-filter parameters cannot both be specified!'

    assert (wq_sub_filter is not None) == has_wq_loops, 'wq_sub_filter must be set if and only if WQ loops are used!'
    assert wq_sub_stride is None or has_wq_loops, 'wq_sub_stride may only be set if WQ loops are used!'
    assert not (stride is not None and has_wq_loops), 'stride may only be set if WQ loops are not used!'

    assert sub_stride is None or has_sub_iterations, 'sub_stride may only be set if sub-filter parameters are used!'

    rank = len(domain)
    zeros = [0] * rank
    ones = [1] * rank

    if not has_filter:
        filter = ones

    if has_sub_iterations:
        if sub_stride is None:
            sub_stride = ones
        filter = iteration_domain(sub_iterations, ones, sub_stride)

    if stride is None:
        stride = filter

    if has_wq_loops:
        if wq_sub_stride is None:
            wq_sub_stride = wq_sub_filter
        filter = iteration_domain(wq_iterations, wq_sub_filter, wq_sub_stride)
        stride = [i * s for i, s in zip(wq_iterations, wq_sub_stride)]
    else:
        wq_iterations = ones
        wq_sub_filter = filter
        wq_sub_stride = wq_sub_filter

    if not has_sub_iterations:
        sub_iterations = filter
        sub_stride = ones

    # TODO: Support setting num_iterations when using strides of 0 and tighten the domain accordingly.

    # Tighten the stride.
    stride = tight_stride(domain, filter, stride)

    if not has_wq_loops:
        wq_sub_stride = stride

    # TODO: Support passing num_iterations as a parameter.
    num_iterations = domain_iterations(domain, filter, stride)

    # TODO: Support passing domain_padding as a parameter.
    domain_padding = zeros * 2

    if offset is None:
        offset = zeros

    if buffer_shape_view is None:
        buffer_shape_view = [d + o for d, o in zip(domain, offset)]

    if dim_order is None:
        dim_order = list(range(len(domain) - 1, -1, -1))

    # TODO: Support passing windows_per_enqueue and windows_per_tail_token as parameters.
    windows_per_enqueue = flat_size(wq_iterations)
    windows_per_tail_token = flat_size(wq_iterations)

    # TODO: Support passing reset_axis as a parameter.
    reset_axis = 0

    # Checks correlated to IterationSpec::AssertValid().
    assert all(v > 0 for v in buffer_shape_view)
    assert all(d > 0 for d in domain)
    assert all(o >= 0 for o in offset)
    assert all(f > 0 for f in filter)
    assert all(s >= 0 for s in stride)
    assert all(i > 0 for i in num_iterations)
    assert all(dp >= 0 for dp in domain_padding)
    assert all(wi > 0 for wi in wq_iterations)
    assert all(wf > 0 for wf in wq_sub_filter)
    assert all(ws >= 0 for ws in wq_sub_stride)
    assert all(si > 0 for si in sub_iterations)
    assert all(ss >= 0 for ss in sub_stride)
    assert all(0 <= a < rank for a in dim_order)
    assert stay_count > 0
    assert windows_per_enqueue > 0
    assert windows_per_tail_token > 0
    assert 0 <= reset_axis < rank

    assert rank == len(filter) == len(stride) == len(domain) == len(num_iterations) == len(buffer_shape_view) == len(
        offset) == len(wq_iterations) == len(wq_sub_filter) == len(wq_sub_stride) == len(sub_iterations) == len(
            sub_stride) == len(dim_order)
    assert rank * 2 == len(domain_padding)

    assert all((d + o) <= v for d, o, v in zip(domain, offset, buffer_shape_view))
    assert all(f <= d for f, d in zip(filter, domain))
    assert all(s <= d for s, d in zip(stride, domain))

    assert filter == iteration_domain(wq_iterations, wq_sub_filter, wq_sub_stride)
    assert filter == iteration_domain(sub_iterations, ones, sub_stride)

    result = IterationSpec()
    result.domain.dims[:] = domain
    result.offset.dims[:] = offset
    result.filter.dims[:] = filter
    result.stride.dims[:] = stride
    result.num_iterations.dims[:] = num_iterations
    result.domain_padding.dims[:] = domain_padding
    result.buffer_shape_view.dims[:] = buffer_shape_view
    result.wq_iterations.dims[:] = wq_iterations
    result.wq_sub_filter.dims[:] = wq_sub_filter
    result.wq_sub_stride.dims[:] = wq_sub_stride
    result.sub_iterations.dims[:] = sub_iterations
    result.sub_stride.dims[:] = sub_stride
    result.dim_order.dims[:] = dim_order
    result.stay_count = stay_count
    result.windows_per_enqueue = windows_per_enqueue
    result.windows_per_tail_token = windows_per_tail_token
    result.reset_axis = reset_axis

    return result


def build_salu_parameters(ops,
                          *,
                          vector_mode: SaluSpec.VectorMode = SaluSpec.EightBit,
                          signed_mode: SaluSpec.SignedMode = SaluSpec.UU,
                          num_beats: int = 1,
                          shift_amount: int = 0,
                          multiply_constant: int = 0) -> SaluParameters:
    """
    Create parameters for a SALU operation.
    """

    params = SaluParameters()
    for op in ops:
        p_op = params.ops.add()
        if isinstance(op, SaluParameters.Op):
            p_op.CopyFrom(op)
        else:
            p_op.op = op[0]
            p_op.num_operands = op[1]

    params.vector_mode = vector_mode
    params.signed_mode = signed_mode
    params.num_beats = num_beats

    params.shift_amount = shift_amount
    params.multiply_constant = multiply_constant

    return params


def build_in_stream_parameters(*,
                               op: InStreamSpec.Op = InStreamSpec.Op.NoOperation,
                               activation: InStreamSpec.Activation = InStreamSpec.Activation.NoActivation,
                               constant: int = 0,
                               shift: int = 0,
                               input_precision: Precision = Precision.InvalidPrecision,
                               output_precision: Precision = Precision.InvalidPrecision) -> InStreamParameters:
    """
    Create a Protobuf `InStreamParameters`.
    """

    params = InStreamParameters()
    params.op = op
    params.activation = activation
    params.constant = constant
    params.shift = shift
    params.input_precision = input_precision
    params.output_precision = output_precision

    return params


def build_in_place_parameters(*,
                              op: InPlaceSpec.Op = InPlaceSpec.Op.NoOperation,
                              input_precision: Precision = Precision.InvalidPrecision,
                              buffer_precision: Precision = Precision.InvalidPrecision) -> InStreamParameters:
    """
    Create a Protobuf `InPlaceParameters`.
    """

    params = InPlaceParameters()
    params.op = op
    params.input_precision = input_precision
    params.buffer_precision = buffer_precision

    return params


def launcher_to_subclass(launcher: Launcher) -> BaseLauncher:
    """
    Get the specific subclass of a `Launcher`.
    """
    if launcher.HasField("infeed"):
        return launcher.infeed
    elif launcher.HasField("outfeed"):
        return launcher.outfeed
    elif launcher.HasField("copy"):
        return launcher.copy
    elif launcher.HasField("salu"):
        return launcher.salu
    elif launcher.HasField("mma_dot"):
        return launcher.mma_dot
    elif launcher.HasField("pad"):
        return launcher.pad
    elif launcher.HasField("dbg_dump"):
        return launcher.dbg_dump
    else:
        raise TypeError(f"Unsupported Launcher! {launcher}")


def launcher_from_subclass(
        launcher_subclass: Infeed | Outfeed | Copy | SaluCommand | MmaDot | Pad | DbgDump) -> Launcher:
    """
    Get a `Launcher` superclass instance for a specific subclass.
    """
    if isinstance(launcher_subclass, Infeed):
        return Launcher(infeed=launcher_subclass)
    elif isinstance(launcher_subclass, Outfeed):
        return Launcher(outfeed=launcher_subclass)
    elif isinstance(launcher_subclass, Copy):
        return Launcher(copy=launcher_subclass)
    elif isinstance(launcher_subclass, SaluCommand):
        return Launcher(salu=launcher_subclass)
    elif isinstance(launcher_subclass, MmaDot):
        return Launcher(mma_dot=launcher_subclass)
    elif isinstance(launcher_subclass, Pad):
        return Launcher(pad=launcher_subclass)
    elif isinstance(launcher_subclass, DbgDump):
        return Launcher(dbg_dump=launcher_subclass)
    else:
        raise TypeError(f"Unsupported Launcher subclass! {launcher_subclass}")


def get_buffer(crate: Crate, name: str) -> Buffer:
    """
    Get the `Buffer` with the given name from a `Crate`.
    """
    return next(b for b in crate.buffers if b.elem.name == name)


def get_launcher(crate: Crate, name: str) -> Launcher:
    """
    Get the `Launcher` with the given name from a `Crate`.
    """
    return next(l for l in crate.launchers if launcher_to_subclass(l).base.elem.name == name)


def get_cluster_index(crate: Crate, cluster: Cluster) -> int:
    """
    Index of a `Cluster` within a `Crate`.
    """
    return next(i for i, c in enumerate(crate.clusters) if c == cluster)


def get_buffer_index(crate: Crate, buffer: Buffer) -> int:
    """
    Index of a `Buffer` within the `Crate`.
    """
    return next(i for i, b in enumerate(crate.buffers) if b == buffer)


def get_launcher_index(crate: Crate, launcher: Launcher) -> int:
    """
    Index of a `Launcher` within the `Crate`.
    """
    return next(i for i, l in enumerate(crate.launchers) if l == launcher)


def get_buffer_cluster(crate: Crate, buffer: Buffer) -> Cluster:
    """
    Get the `Cluster` of a `Crate` that a `Buffer` belongs to.
    """
    index = get_buffer_index(crate, buffer)
    return next(cluster for cluster in crate.clusters if index in cluster.buffers)


def get_launcher_cluster(crate: Crate, launcher: Launcher) -> Cluster:
    """
    Get the `Cluster` of a `Crate` that a `Launcher` belongs to.
    """
    index = get_launcher_index(crate, launcher)
    return next(cluster for cluster in crate.clusters if index in cluster.launchers)


class CrateBuilder:
    """
    An interface for creating a protobuf `Crate`.

    Note that little to no checking is done regarding parameter validity.
    """
    def __init__(self, name: str = 'Crate'):
        self.crate = Crate()
        self.crate.name = name

        self._launcher_builders = {
            Infeed: self.infeed,
            Outfeed: self.outfeed,
            Copy: self.copy,
            SaluCommand: self.salu,
            MmaDot: self.mma_dot,
            Pad: self.pad,
            DbgDump: self.dbg_dump,
        }

        self._clusters = {}

        # Track the offset needed per section per Cluster for SRAM memory manager allocations.
        # The following structure is roughly Dict[int, Dict[int, int]] where the keys of the outer dictionary are the
        # indices of the Clusters with memory manager allocations, the keys of the inner dictionary are section IDs, and
        # the values of the inner dictionary are the total bytes allocated for that section of that Cluster.
        self._memory_manager_offsets = defaultdict(lambda: defaultdict(lambda: 0))

    def get_cluster(self, tile_id: TileId) -> Cluster:
        """
        Get the `Cluster` associated with a `TileId`, creating a new one if needed.
        """
        # Need to use the string representation since Protobuf message objects are not hashable.
        key = str(tile_id)

        if key not in self._clusters:
            cluster = self.crate.clusters.add()
            cluster.tile_id.CopyFrom(tile_id)
            self._clusters[key] = cluster

        return self._clusters[key]

    def add_memory_manager_allocation(self, cluster: Cluster, section: int, bytes: int) -> Cluster.SramAllocation:
        """
        Add a `Cluster.SramAllocation` that utilizes the memory manager for a `Cluster`.

        The memory manager allocations are assumed to all be shared, each utilizing a different offset into the same
        base SRAM allocation in the `Cluster` for each section.

        NOTE: Assumes that `Cluster`s and their allocations are never removed or modified from the `Builder` between
        calls to this method. Doing so would invalidate the internal tracking structure.
        """
        index = get_cluster_index(self.crate, cluster)
        offset = self._memory_manager_offsets[index][section]
        self._memory_manager_offsets[index][section] += bytes

        return Cluster.SramAllocation(
            type=Cluster.SramAllocation.Type.MemoryManager,
            bytes=bytes,
            offset=offset,
            sharable=True,
        )

    def buffer(
        self,
        *,
        shape,
        padding=None,
        tile_id=None,
        physical_size=None,
        name='',
        initialization_data=None,
        section=0,
    ):
        """
        Add a `Buffer`.
        """
        if physical_size is None:
            physical_size = flat_size(shape)
        if padding is None:
            padding = [0] * len(shape) * 2
        if not name:
            name = 'buffer_' + str(len(self.crate.buffers))
        if initialization_data is None:
            initialization_data = []

        result = self.crate.buffers.add()

        result.elem.name = name
        result.elem.section = section

        _set_tile_id(result, tile_id)

        result.physical_size = physical_size
        result.shape.dims[:] = shape
        result.padding.dims[:] = padding

        result.initialization_data = bytes(initialization_data)

        return result

    def _base_launcher(self, tile_id, name, section):
        """
        Create a `BaseLauncher`.
        """
        if not name:
            name = 'launcher_' + str(len(self.crate.launchers))

        result = BaseLauncher()

        result.elem.name = name
        result.elem.section = section

        _set_tile_id(result, tile_id)

        return result

    def _set_input(self, launcher, input_):
        """
        Add an input to a `Launcher`.
        """
        index = get_buffer_index(self.crate, input_[0])
        launcher.inputs.add()
        launcher.inputs[0].parameter_info.buffer_index = index
        launcher.inputs[0].parameter_info.iteration_spec.CopyFrom(input_[1])
        if len(input_) > 2:
            launcher.inputs[0].parameter_info.in_stream_params.CopyFrom(input_[2])
        if len(input_) > 3:
            launcher.inputs[0].parameter_info.in_place_params.CopyFrom(input_[3])

    def _set_output(self, launcher, output):
        """
        Add an output to a `Launcher`.
        """
        index = get_buffer_index(self.crate, output[0])
        launcher.outputs.add()
        launcher.outputs[0].parameter_info.buffer_index = index
        launcher.outputs[0].parameter_info.iteration_spec.CopyFrom(output[1])
        if len(output) > 2:
            launcher.outputs[0].parameter_info.in_stream_params.CopyFrom(output[2])
        if len(output) > 3:
            launcher.outputs[0].parameter_info.in_place_params.CopyFrom(output[3])

    def infeed(self, *, tile_id=None, name, section=0, output):
        """
        Add an `Infeed`.
        """
        result = self.crate.launchers.add().infeed
        result.base.CopyFrom(self._base_launcher(tile_id, name, section))

        self._set_output(result, output)

        return result

    def outfeed(self, *, tile_id=None, name, section=0, input):
        """
        Add an `Outfeed`.
        """
        result = self.crate.launchers.add().outfeed
        result.base.CopyFrom(self._base_launcher(tile_id, name, section))

        self._set_input(result, input)

        return result

    def copy(self, *, tile_id=None, name='', section=0, input, output):
        """
        Add a `Copy`.
        """
        result = self.crate.launchers.add().copy
        result.base.CopyFrom(self._base_launcher(tile_id, name, section))

        self._set_input(result, input)
        self._set_output(result, output)

        return result

    def salu(self, *, params, tile_id=None, name='', section=0, input, output):
        """
        Add a `SaluCommand`.
        """
        result = self.crate.launchers.add().salu
        result.base.CopyFrom(self._base_launcher(tile_id, name, section))

        result.params.CopyFrom(params)

        self._set_input(result, input)
        self._set_output(result, output)

        return result

    def mma_dot(self,
                *,
                tile_id=None,
                input,
                output,
                name='',
                section=0,
                size,
                bias_inputs=0,
                offset=(0, 0),
                data=None,
                biases=None,
                activation=0,
                ifsr=2,
                pfsr=2,
                multiplier=1,
                shift=0,
                bit_spreading=BitSpreadingMode.Normal,
                adc_cycles=[8, 8, 8, 8, 8, 8, 8, 8],
                input_start_bit=0,
                input_end_bit=7):
        """
        Add an `MmaDot`.
        """
        if data is None:
            data = []
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if biases is None:
            biases = []
        if not isinstance(biases, np.ndarray):
            biases = np.array(biases, dtype=np.float32)

        result = self.crate.launchers.add().mma_dot
        result.base.CopyFrom(self._base_launcher(tile_id, name, section))

        # Create an area to cover all weight data.
        area = result.weights.areas.add()
        area.bank_id = 0
        area.rect.inputs, area.rect.outputs = size
        area.rect.input_offset, area.rect.output_offset = offset

        area.data = bytes(data.astype(np.int8))
        area.float_data.extend(data.astype(np.float32).flatten())

        area.bias_inputs = bias_inputs
        area.biases = bytes(biases.astype(np.int8))
        area.float_biases.extend(biases.astype(np.float32).flatten())

        result.activation = activation
        result.ifsr = ifsr
        result.pfsr = pfsr
        result.multiplier = multiplier
        result.shift = shift
        result.bitspreading = bit_spreading
        result.adc_cycles.dims[:] = adc_cycles
        result.input_start_bit = input_start_bit
        result.input_end_bit = input_end_bit

        self._set_input(result, input)
        self._set_output(result, output)

        return result

    def pad(self, *, tile_id=None, name='', section=0, output):
        """
        Add a `Pad`.
        """
        result = self.crate.launchers.add().pad
        result.base.CopyFrom(self._base_launcher(tile_id, name, section))

        self._set_output(result, output)

        return result

    def dbg_dump(self, *, tile_id=None, input=None, name='', section=0, output):
        """
        Add a `DbgDump`.
        """
        result = self.crate.launchers.add().dbg_dump
        result.base.CopyFrom(self._base_launcher(tile_id, name, section))

        if input != None:
            self._set_input(result, input)
        else:
            result.inputs.add()
        self._set_output(result, output)

        return result

    def add_buffer(self,
                   *,
                   name: str,
                   cluster: Cluster,
                   allocation: Cluster.SramAllocation | None = None,
                   shape: Shape,
                   **kwargs) -> Buffer:
        """
        Build a `Buffer` and add it to a `Cluster`.

        TODO: Eventually, this should be consolidated with or replace the normal `Buffer` creation method.
        """
        if allocation is None:
            # Default to real SRAM allocations.
            allocation = Cluster.SramAllocation(type=Cluster.SramAllocation.Type.Sram, bytes=flat_size(shape))

        index = len(self.crate.buffers)
        buffer = self.buffer(
            name=name,
            tile_id=cluster.tile_id,
            shape=shape,
            physical_size=allocation.bytes,
            **kwargs,
        )

        cluster.buffers[index].CopyFrom(allocation)

        return buffer

    def add_launcher(self,
                     launcher_type: type,
                     *,
                     name: str,
                     cluster: Cluster,
                     allocation: Cluster.SramAllocation | None = None,
                     **kwargs):
        """
        Build a `Launcher` and add it to a `Cluster`.

        TODO: Eventually, this should be consolidated with or replace the normal `Launcher` creation methods.
        """
        builder = self._launcher_builders[launcher_type]

        if allocation is None:
            # Default to real SRAM allocations.
            # TODO: Assuming for now that Launcher allocations don't need code size. They are currently non-trivial to
            # calculate here and not really used when the Crate is deserialized.
            allocation = Cluster.SramAllocation(type=Cluster.SramAllocation.Type.Sram, bytes=0)

        index = len(self.crate.launchers)
        launcher = builder(name=name, tile_id=cluster.tile_id, **kwargs)

        cluster.launchers[index].CopyFrom(allocation)

        return launcher
