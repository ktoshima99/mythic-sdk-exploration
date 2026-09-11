"""Verify the source-generated encoder/decoder builders are equivalent to the shipped
hand-written ones, before trusting them for any swept layer count.

Uses a fixed-seed trick: every weight is drawn via bevformer.modeling.backbone
.initialize_weight() -> np.random.random(shape). Seeding identically before building the
hand-written version and the generated version (at the SAME num_layers as shipped: 3 for
the encoder, 6 for the decoder) makes every weight draw byte-identical, provided the build
order (which weight is drawn in which order) matches -- which is true by construction here
(see dynamic_transformer_builders.py docstrings).

Usage (inside the compilerd container):
    /mythic/pyvnnsdk-env/bin/python verify_dynamic_layers.py
Exits 0 and prints "OK" for both encoder and decoder on success; raises otherwise.
"""

import sys

sys.path.insert(0, "/mythic/vnnsdk/scripts")

import numpy as np  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from bevformer.bevformer_tiny import BevformerTiny  # noqa: E402
from bevformer.modeling.decoder import build_bevformer_tiny_decoder  # noqa: E402
from bevformer.modeling.encoder import build_bevformer_tiny_encoder  # noqa: E402
from dynamic_transformer_builders import (  # noqa: E402
    build_bevformer_tiny_decoder_n,
    build_bevformer_tiny_encoder_n,
)

config = BevformerTiny.config


def _assert_model_protos_equal(proto_a, proto_b, label):
    nodes_a = list(proto_a.graph.node)
    nodes_b = list(proto_b.graph.node)
    if len(nodes_a) != len(nodes_b):
        raise AssertionError(f"{label}: node count differs: {len(nodes_a)} vs {len(nodes_b)}")
    for i, (na, nb) in enumerate(zip(nodes_a, nodes_b)):
        if na.op_type != nb.op_type:
            raise AssertionError(f"{label}: node {i} op_type differs: {na.op_type} vs {nb.op_type}")
        if list(na.input) != list(nb.input):
            raise AssertionError(f"{label}: node {i} ({na.op_type}) inputs differ: {list(na.input)} vs {list(nb.input)}")
        if list(na.output) != list(nb.output):
            raise AssertionError(
                f"{label}: node {i} ({na.op_type}) outputs differ: {list(na.output)} vs {list(nb.output)}"
            )

    inits_a = {i.name: i for i in proto_a.graph.initializer}
    inits_b = {i.name: i for i in proto_b.graph.initializer}
    if inits_a.keys() != inits_b.keys():
        raise AssertionError(f"{label}: initializer name sets differ: {inits_a.keys() ^ inits_b.keys()}")
    for name in inits_a:
        arr_a = numpy_helper.to_array(inits_a[name])
        arr_b = numpy_helper.to_array(inits_b[name])
        if not np.array_equal(arr_a, arr_b):
            raise AssertionError(f"{label}: initializer '{name}' differs")

    ins_a = [(i.name, i.type.tensor_type.shape.SerializeToString()) for i in proto_a.graph.input]
    ins_b = [(i.name, i.type.tensor_type.shape.SerializeToString()) for i in proto_b.graph.input]
    if ins_a != ins_b:
        raise AssertionError(f"{label}: graph inputs differ: {ins_a} vs {ins_b}")

    print(f"{label}: OK ({len(nodes_a)} nodes, {len(inits_a)} initializers match byte-for-byte)")


def verify_encoder():
    n = 3  # shipped hardcoded value
    # BevformerTiny._input_shapes() is the single source of truth for shapes (derived from
    # `config` the same way the real graph-input declarations are) -- reuse it rather than
    # hand-deriving ref_pts_cam's channel count etc., which is easy to get wrong by hand.
    shapes = BevformerTiny._input_shapes()
    fixed_inputs = {
        name: np.random.default_rng(seed).random(shapes[name]).astype(np.float32)
        for seed, name in enumerate(
            ["bev_query", "bev_value_stacked", "img_features", "bev_mask", "ref_pts_cam", "count_normalizer"]
        )
    }

    np.random.seed(0)
    original = build_bevformer_tiny_encoder(config)
    original(**fixed_inputs)
    proto_original = original.to_model_proto()

    np.random.seed(0)
    generated = build_bevformer_tiny_encoder_n(config, n)
    generated(**fixed_inputs)
    proto_generated = generated.to_model_proto()

    _assert_model_protos_equal(proto_original, proto_generated, "encoder (N=3)")


def verify_decoder():
    n = 6  # shipped hardcoded value
    shapes = BevformerTiny._input_shapes()
    fixed_inputs = {
        "obj_queries": np.random.default_rng(10).random(shapes["obj_queries"]).astype(np.float32),
        # "bev_features" is an internal encoder-output edge, not a top-level ONNX input, so it
        # has no entry in _input_shapes() -- derive it directly: [B, C, 1, bev_h*bev_w].
        "bev_features": np.random.default_rng(11)
        .random([1, config._dim_, 1, config.bev_h_ * config.bev_w_])
        .astype(np.float32),
        "ref_points": np.random.default_rng(12).random(shapes["ref_points"]).astype(np.float32),
    }

    np.random.seed(0)
    original = build_bevformer_tiny_decoder(config)
    original(**fixed_inputs)
    proto_original = original.to_model_proto()

    np.random.seed(0)
    generated = build_bevformer_tiny_decoder_n(config, n)
    generated(**fixed_inputs)
    proto_generated = generated.to_model_proto()

    _assert_model_protos_equal(proto_original, proto_generated, "decoder (N=6)")


if __name__ == "__main__":
    verify_encoder()
    verify_decoder()
    print("ALL OK")
