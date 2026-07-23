from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
from typing import Counter as CounterType
import logging

import onnx
from onnx import numpy_helper

logger = logging.getLogger(__name__)


@dataclass
class StatsAcc:
    """Accumulator for intermediate ONNX graph statistics."""

    init_bytes: int = 0
    init_count: int = 0
    n_nodes: int = 0
    macs_conv_total: int = 0


@dataclass
class OnnxStats:  # noqa: DOC601,DOC603
    """Container for aggregated ONNX model statistics.

    Parameters:
        ir_version (int): ONNX intermediate representation (IR) version of the model.
        opsets (List[Tuple[str, int]]): List of (domain, version) pairs describing
            the opset imports used by the model.
        n_nodes (int): Total number of nodes in the ONNX graph, including all
            subgraphs.
        n_initializers (int): Number of initializers (weights and constants)
            present in the model.
        initializer_total_bytes (int): Total size of all initializers in bytes.
        op_hist (Dict[str, int]): Histogram mapping ONNX operator types to their
            occurrence counts.
        macs_conv_total (int): Total number of multiply-accumulate operations
            (MACs) contributed by convolution layers.
    """

    ir_version: int
    opsets: List[Tuple[str, int]]
    n_nodes: int
    n_initializers: int
    initializer_total_bytes: int
    op_hist: Dict[str, int]
    macs_conv_total: int


def _tensor_payload_bytes(t: onnx.TensorProto) -> int:  # noqa
    """Return the payload size in bytes for an ONNX TensorProto.

    The size is determined in a robust way and supports both embedded and
    external tensor data.

    Resolution order:
        1. External data: use the ``length`` entry if present.
        2. Raw data: use ``len(raw_data)``.
        3. Typed fields: estimate size as element_size * element_count.
        4. Fallback: convert via ``numpy_helper.to_array`` and use ``nbytes``.

    Args:
        t: ONNX ``TensorProto`` whose payload size should be computed.

    Returns:
        Payload size in bytes. Returns 0 if the size cannot be determined.
    """
    if t.external_data:
        for e in t.external_data:
            if e.key == "length":
                try:
                    return int(e.value)
                except Exception as exc:
                    raise RuntimeError(f"Invalid external_data length for TensorProto '{t.name}': {e.value}") from exc

    if t.raw_data:
        return len(t.raw_data)

    if t.float_data:
        return 4 * len(t.float_data)
    if t.int32_data:
        return 4 * len(t.int32_data)
    if t.int64_data:
        return 8 * len(t.int64_data)
    if t.double_data:
        return 8 * len(t.double_data)
    if t.uint64_data:
        return 8 * len(t.uint64_data)
    if t.string_data:
        return sum(len(s) for s in t.string_data)

    try:
        return int(numpy_helper.to_array(t).nbytes)
    except Exception as exc:
        raise RuntimeError(f"Failed To determine payload size for TensorProto '{t.name}'") from exc


def generate_onnx_graph_stats(  # noqa
    model: onnx.ModelProto,
    infer_shapes: bool = True,
) -> OnnxStats:
    """Compute compact structural and memory statistics for an ONNX model.

    The function optionally runs ONNX shape inference, traverses the full
    computation graph including nested subgraphs (e.g. in If/Loop/Scan),
    and aggregates statistics about operators and initializers.

    Initializer payload sizes include support for external tensor data.

    If ``model_state`` indicates a quantized model, initializer sizes can be
    estimated under quantization assumptions rather than using the raw
    protobuf storage format.

    Args:
        model: ONNX model to analyze.
        infer_shapes: If True, run ONNX shape inference before collecting
            statistics. Failures are ignored gracefully.

    Returns:
        An ``OnnxStats`` instance containing aggregated graph statistics,
        including operator histogram, node count, and estimated initializer
        payload size in bytes.
    """
    if infer_shapes:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            raise

    opsets = [(op.domain or "", int(op.version)) for op in model.opset_import]

    op_hist: CounterType[str] = Counter()

    def _initializer_estimated_bytes(t: onnx.TensorProto) -> int:

        return _tensor_payload_bytes(t)

    def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
        for a in node.attribute:
            if a.name == name and a.type == onnx.AttributeProto.INT:
                return int(a.i)
        return default

    def _shape_env(g: onnx.GraphProto) -> Dict[str, List[int]]:
        """Build a static shape environment for tensors in an ONNX graph.

        The returned mapping associates tensor names with their static shape
        as a list of integers. Shapes are collected from graph inputs, outputs,
        value infos, and initializers.

        Tensors with symbolic or unknown dimensions are excluded to ensure
        that all recorded shapes are fully static.

        Args:
            g (onnx.GraphProto): ONNX graph from which to extract static tensor shapes.

        Returns:
            Dict[str, List[int]]: Mapping from tensor names to static shapes.
        """
        env: Dict[str, List[int]] = {}

        def add_vi(vi: onnx.ValueInfoProto) -> None:
            tt = vi.type.tensor_type
            if not tt.HasField("shape"):
                return
            dims: List[int] = []
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    return  # sybmbolic unknown -> skip
            env[vi.name] = dims

        for vi in list(g.input) + list(g.output) + list(g.value_info):
            add_vi(vi)

        for init in g.initializer:
            env[init.name] = list(init.dims) if len(init.dims) > 0 else [1]

        return env

    def _env_get(env: Dict[str, List[int]], name: str) -> List[int] | None:
        s = env.get(name)
        if s is not None:
            return s
        # pragmatic fallback for your naming convention
        if name.endswith("_quantized"):
            base = name[: -len("_quantized")]
            return env.get(base)
        return None

    def _conv_macs(node: onnx.NodeProto, env: Dict[str, List[int]]) -> int:
        """
        Estimate MACs for Conv-like nodes.

        Supported cases:
        1) Conv2D NCHW:
            x: [N, C_in, H, W]
            w: [C_out, C_in/groups, K_h, K_w]
            y: [N, C_out, H_out, W_out]

        2) FC / MatMul-like (appears as vidConv with 2D tensors):
            x: [N, K]
            w: [M, K]
            y: [N, M]

        Notes:
        - Dynamic batch (-1) is allowed and treated as N=1.
        - Naming convention '<name>_quantized' is resolved via _env_get.
        """
        macs = 0

        x = _env_get(env, node.input[0]) if len(node.input) >= 1 else None
        w = _env_get(env, node.input[1]) if len(node.input) >= 2 else None
        y = _env_get(env, node.output[0]) if len(node.output) >= 1 else None

        # ---------- Case 1: Conv2D ----------
        if x is not None and w is not None and y is not None:
            if len(x) == 4 and len(w) == 4 and len(y) == 4:
                # Batch handling
                n = 1 if x[0] == -1 else x[0]
                if n > 0 and not any(d <= 0 for d in x[1:] + w + y[1:]):
                    c_in = x[1]
                    c_out = y[1]
                    h_out = y[2]
                    w_out = y[3]
                    k_h = w[2]
                    k_w = w[3]

                    groups = _get_attr_int(node, "group", 1)
                    if groups > 0 and c_in % groups == 0:
                        macs = int(n) * int(h_out) * int(w_out) * int(c_out) * int(c_in // groups) * int(k_h) * int(k_w)

        # ---------- Case 2: FC / Linear ----------
        if macs == 0 and x is not None and w is not None and y is not None:
            if len(x) == 2 and len(w) == 2 and len(y) == 2:
                n = 1 if x[0] == -1 else x[0]
                k = x[1]
                m = w[0]

                if n > 0 and k > 0 and m > 0 and w[1] == k and y[1] == m:
                    macs = int(n) * int(m) * int(k)

        # ---------- Diagnostics ----------
        if macs == 0:
            logger.warning(
                "Conv MACs = 0 for node '%s' (op=%s). Shapes: X=%s, W=%s, Y=%s",
                node.name or "<unnamed>",
                node.op_type,
                x,
                w,
                y,
            )

        return macs

    def visit_graph(g: onnx.GraphProto, stats: StatsAcc) -> None:
        """Traverse an ONNX graph and accumulate global statistics.

        The function updates nonlocal counters for:
            - number of nodes
            - number and total size of initializers
            - total Conv MACs

        Subgraphs referenced by node attributes are visited recursively.
        """
        # Collect static tensor shapes once per graph
        env = _shape_env(g)

        # Account for all initializers in this graph
        for init in g.initializer:
            stats.init_bytes += _initializer_estimated_bytes(init)
            stats.init_count += 1

        # Traverse nodes in the graph
        for node in g.node:
            stats.n_nodes += 1
            op_hist[node.op_type] += 1

            # Accumulate MACs for convolution operators
            if "Conv" in node.op_type or "vidConv" in node.op_type:
                stats.macs_conv_total += _conv_macs(node, env)

            # Recurse into subgraphs (e.g. If, Loop, Scan)
            for a in node.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    visit_graph(a.g, stats)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs:
                        visit_graph(sg, stats)

    stats = StatsAcc()
    visit_graph(model.graph, stats)
    return OnnxStats(
        ir_version=int(model.ir_version),
        opsets=opsets,
        n_nodes=stats.n_nodes,
        n_initializers=stats.init_count,
        initializer_total_bytes=stats.init_bytes,
        op_hist=dict(op_hist),
        macs_conv_total=int(stats.macs_conv_total),
    )
