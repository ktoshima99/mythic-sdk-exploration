# mypy: ignore-errors
import numpy as np
from onnx.numpy_helper import from_array
from onnxscript import FLOAT
from onnxscript import opset20 as op
from onnxscript import script

from .backbone import initialize_weight
from vnnort.optimizer.functions import Shortcut, vidConv, vidLayerNorm, vidSoftmax, vidGridSample


def build_multi_head_self_attention(config):
    """Build the multi-head self-attention sub-layer of the BEVFormer-tiny decoder."""
    # Notation: B=1, N_obj=900 (object queries), C=256, H=8 (heads), D=32 (C/H)
    C = config._dim_
    H = config.model["pts_bbox_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][0]["num_heads"]
    D = C // H
    N_obj = config.model["pts_bbox_head"]["num_query"]
    B = 1

    q_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1) * np.float32(
        D**-0.5
    )  # Absorb scaling into weight
    q_proj_b_np = initialize_weight([C], dtype=np.float32) * np.float32(D**-0.5)  # Absorb scaling into bias
    k_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    k_proj_b_np = initialize_weight([C], dtype=np.float32)
    v_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    v_proj_b_np = initialize_weight([C], dtype=np.float32)
    out_proj_w_np = initialize_weight([C, C], dtype=np.float32).reshape(C, C, 1, 1)
    out_proj_b_np = initialize_weight([C], dtype=np.float32)
    ln_scale_np = initialize_weight([C], dtype=np.float32)
    ln_bias_np = initialize_weight([C], dtype=np.float32)
    query_pos_enc_np = initialize_weight([N_obj, B, C], dtype=np.float32).transpose(1, 2, 0).reshape(B, C, 1, N_obj)
    key_pos_enc_np = initialize_weight([N_obj, B, C], dtype=np.float32).transpose(1, 2, 0).reshape(B, C, 1, N_obj)

    @script(default_opset=op)
    def MultiHeadSelfAttention(
        obj_queries: FLOAT["B", "C", 1, "N_obj"],
    ) -> FLOAT["B", "C", 1, "N_obj"]:
        q_proj_w = op.Constant(value=from_array(q_proj_w_np))
        q_proj_b = op.Constant(value=from_array(q_proj_b_np))
        k_proj_w = op.Constant(value=from_array(k_proj_w_np))
        k_proj_b = op.Constant(value=from_array(k_proj_b_np))
        v_proj_w = op.Constant(value=from_array(v_proj_w_np))
        v_proj_b = op.Constant(value=from_array(v_proj_b_np))
        out_proj_w = op.Constant(value=from_array(out_proj_w_np))
        out_proj_b = op.Constant(value=from_array(out_proj_b_np))
        ln_scale = op.Constant(value=from_array(ln_scale_np))
        ln_bias = op.Constant(value=from_array(ln_bias_np))
        query_pos_enc = op.Constant(value=from_array(query_pos_enc_np))
        key_pos_enc = op.Constant(value=from_array(key_pos_enc_np))

        # --- Q projection ---
        q = Shortcut(obj_queries, query_pos_enc, mode="addition")
        q = vidConv(q, q_proj_w, q_proj_b, kernel_shape=[1, 1])

        # # --- K projection ---
        k = Shortcut(obj_queries, key_pos_enc, mode="addition")
        k = vidConv(k, k_proj_w, k_proj_b, kernel_shape=[1, 1])

        # # --- V projection ---
        v = vidConv(obj_queries, v_proj_w, v_proj_b, kernel_shape=[1, 1])

        # # --- Scaled dot-product attention ---
        # Make k the weights of the dynamic conv
        k = op.Reshape(k, op.Constant(value_ints=[H, D, 1, N_obj]))
        k = op.Transpose(k, perm=[0, 2, 3, 1])  # [H, 1, N_obj, D]
        k = op.Reshape(k, op.Constant(value_ints=[-1, D, 1, 1]))
        qk = vidConv(q, k, None, kernel_shape=[1, 1], group=H)  # [1, ]
        attn = vidSoftmax(qk, group=[H])
        # --- Weighted aggregation and output projection ---
        # Make v the weights of the dynamic conv
        v = op.Transpose(v, perm=(1, 3, 0, 2))

        out = vidConv(attn, v, None, kernel_shape=[1, 1], group=H)
        out = vidConv(out, out_proj_w, out_proj_b, kernel_shape=[1, 1])

        out = Shortcut(out, obj_queries, mode="addition")  # residual
        out = vidLayerNorm(out, ln_scale, ln_bias)
        return out

    return MultiHeadSelfAttention


def build_decoder_deformable_attention(config):
    """Build the deformable cross-attention sub-layer of the BEVFormer-tiny decoder."""
    # Notation: B=1, N_obj=900, C=256, H=8, D=32 (C/H), L=1 (num_levels), P=4 (num_points)
    #           N_bev=2500 (50×50 BEV), H*L*P=32, H*L*P*2=64
    C = config._dim_
    H = config.model["pts_bbox_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][0]["num_heads"]
    D = C // H
    N_obj = config.model["pts_bbox_head"]["num_query"]
    L = config.model["pts_bbox_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][1]["num_levels"]
    P = config.model["pts_bbox_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][1].get(
        "num_points", 4
    )
    bev_h = config.bev_h_
    bev_w = config.bev_w_
    N_bev = bev_h * bev_w
    ffn_dim = config._ffn_dim_
    B = 1

    query_pos_enc_np = initialize_weight([N_obj, B, C], dtype=np.float32).transpose(1, 2, 0).reshape(B, C, 1, N_obj)
    sampling_offsets_w_np = initialize_weight([C, H * L * P * 2], dtype=np.float32).T.reshape(H * L * P * 2, C, 1, 1)
    sampling_offsets_b_np = initialize_weight([H * L * P * 2], dtype=np.float32)
    # Absorb Div by offset_normalizer=[bev_w, bev_h] into Conv weight and bias.
    m_per_channel = np.tile(np.array([1.0 / bev_w, 1.0 / bev_h], dtype=np.float32), H * L * P)
    sampling_offsets_w_np = sampling_offsets_w_np * m_per_channel[:, np.newaxis, np.newaxis, np.newaxis]
    sampling_offsets_b_np = sampling_offsets_b_np * m_per_channel
    value_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    value_proj_b_np = initialize_weight([C], dtype=np.float32)
    attn_weights_w_np = initialize_weight([C, H * L * P], dtype=np.float32).T.reshape(H * L * P, C, 1, 1)
    attn_weights_b_np = initialize_weight([H * L * P], dtype=np.float32)
    output_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    output_proj_b_np = initialize_weight([C], dtype=np.float32)
    ffn_fc1_w_np = initialize_weight([C, ffn_dim], dtype=np.float32).T.reshape(ffn_dim, C, 1, 1)
    ffn_fc1_b_np = initialize_weight([ffn_dim], dtype=np.float32)
    ffn_fc2_w_np = initialize_weight([ffn_dim, C], dtype=np.float32).T.reshape(C, ffn_dim, 1, 1)
    ffn_fc2_b_np = initialize_weight([C], dtype=np.float32)
    ln2_scale_np = initialize_weight([C], dtype=np.float32)
    ln2_bias_np = initialize_weight([C], dtype=np.float32)
    ln3_scale_np = initialize_weight([C], dtype=np.float32)
    ln3_bias_np = initialize_weight([C], dtype=np.float32)

    @script(default_opset=op)
    def DecoderDeformableAttention(
        obj_queries: FLOAT["B", "C", 1, "N_obj"],
        bev_features: FLOAT["B", "C", 1, "N_bev"],
        ref_points: FLOAT["B", 3, 1, "N_obj"],
    ) -> FLOAT["B", "C", 1, "N_obj"]:
        query_pos_enc = op.Constant(value=from_array(query_pos_enc_np))
        sampling_offsets_w = op.Constant(value=from_array(sampling_offsets_w_np))
        sampling_offsets_b = op.Constant(value=from_array(sampling_offsets_b_np))
        value_proj_w = op.Constant(value=from_array(value_proj_w_np))
        value_proj_b = op.Constant(value=from_array(value_proj_b_np))
        attn_weights_w = op.Constant(value=from_array(attn_weights_w_np))
        attn_weights_b = op.Constant(value=from_array(attn_weights_b_np))
        output_proj_w = op.Constant(value=from_array(output_proj_w_np))
        output_proj_b = op.Constant(value=from_array(output_proj_b_np))
        ffn_fc1_w = op.Constant(value=from_array(ffn_fc1_w_np))
        ffn_fc1_b = op.Constant(value=from_array(ffn_fc1_b_np))
        ffn_fc2_w = op.Constant(value=from_array(ffn_fc2_w_np))
        ffn_fc2_b = op.Constant(value=from_array(ffn_fc2_b_np))
        ln2_scale = op.Constant(value=from_array(ln2_scale_np))
        ln2_bias = op.Constant(value=from_array(ln2_bias_np))
        ln3_scale = op.Constant(value=from_array(ln3_scale_np))
        ln3_bias = op.Constant(value=from_array(ln3_bias_np))

        # --- Query preparation ---
        # Add positional encoding
        q_with_pos = Shortcut(obj_queries, query_pos_enc, mode="addition")  # [B, C, 1, N_obj]

        # # --- Attention weights ---
        # # Project to per-point weights, softmax over L*P points per head
        attn = vidConv(q_with_pos, attn_weights_w, attn_weights_b, kernel_shape=[1, 1])
        attn = vidSoftmax(attn, group=[H])  # softmax over L*P points per head # [B, H*L*P, 1, N_obj]
        attn = op.Reshape(attn, op.Constant(value_ints=[B, H, L * P, N_obj]))
        attn = op.Transpose(attn, perm=[0, 2, 1, 3])
        attn = op.Reshape(attn, op.Constant(value_ints=[B, L * P, 1, H * N_obj]))

        # # --- Sampling offsets ---
        # # Project to offsets, normalize by BEV spatial shape, add reference points, convert to [-1,1]
        off = vidConv(q_with_pos, sampling_offsets_w, sampling_offsets_b, kernel_shape=[1, 1])
        # Transform to [B, 2, H*L*P, N_obj] for easier manipulation
        off = op.Reshape(off, op.Constant(value_ints=[B, H, L * P, 2, N_obj]))
        off = op.Transpose(off, perm=[0, 3, 1, 2, 4])
        off = op.Reshape(off, op.Constant(value_ints=[B, 2, H * L * P, N_obj]))

        # Only refine xy
        ref_points = op.Slice(
            ref_points, op.Constant(value_ints=[0]), op.Constant(value_ints=[2]), op.Constant(value_ints=[1])
        )
        # FIXME unnecessary mul with 1.0 to allow sigmoid
        ref_points = Shortcut(
            ref_points, op.Constant(value=from_array(np.array([[[[1.0]]]], dtype=np.float32))), mode="multiplication"
        )
        ref_points = op.Sigmoid(ref_points)
        off = Shortcut(off, ref_points, mode="addition")

        off = Shortcut(
            off, op.Constant(value=from_array(np.array([[[[2.0]]]], dtype=np.float32))), mode="multiplication"
        )  # → [0, 2]
        off = Shortcut(
            off, op.Constant(value=from_array(np.array([[[[-1.0]]]], dtype=np.float32))), mode="addition"
        )  # → [-1, 1] for GridSample

        # Transform to [H, N_obj, L*P, 2] for GridSample
        grid = op.Reshape(off, op.Constant(value_ints=[B, 2, H, L * P, N_obj]))
        grid = op.Transpose(grid, perm=[0, 2, 1, 4, 3])
        grid = op.Reshape(grid, op.Constant(value_ints=[H, 2, N_obj, L * P]))

        # # --- Value projection from BEV features ---
        val = vidConv(bev_features, value_proj_w, value_proj_b, kernel_shape=[1, 1])  # [B, C, 1, N_bev]
        # Prepare for grid sample. Transform to [H, D, bev_h, bev_w]
        val = op.Reshape(val, op.Constant(value_ints=[H, D, bev_h, bev_w]))

        # --- GridSample feature aggregation ---
        # Sample BEV at deformable locations, weight by attention, sum over points
        feat = vidGridSample(val, grid, align_corners=0, mode="linear", padding_mode="zeros")  # [H, D, N_obj, P]

        # Transform to [B, L * P, D, H * N_obj]
        feat = op.Transpose(feat, perm=[3, 1, 0, 2])
        feat = op.Reshape(feat, op.Constant(value_ints=[B, L * P, D, H * N_obj]))
        out = Shortcut(feat, attn, mode="multiplication")  # [B, L*P, D, H*N_obj]

        # Express ReduceSum as Conv with kernel of ones
        out = vidConv(out, op.Constant(value=np.ones((1, L * P, 1, 1), dtype=np.float32)), None, kernel_shape=[1, 1])

        # # --- Output projection ---
        out = op.Reshape(out, op.Constant(value_ints=[B, D, H, N_obj]))
        out = op.Transpose(out, perm=[0, 2, 1, 3])
        out = op.Reshape(out, op.Constant(value_ints=[B, C, 1, N_obj]))
        out = vidConv(out, output_proj_w, output_proj_b, kernel_shape=[1, 1])

        # # --- Residual + LN2 + FFN + LN3 ---
        out = Shortcut(out, obj_queries, mode="addition")  # residual
        out = vidLayerNorm(out, ln2_scale, ln2_bias)
        res = out
        out = vidConv(out, ffn_fc1_w, ffn_fc1_b, kernel_shape=[1, 1])
        out = op.Relu(out)
        out = vidConv(out, ffn_fc2_w, ffn_fc2_b, kernel_shape=[1, 1])
        out = Shortcut(out, res, mode="addition")  # residual
        out = vidLayerNorm(out, ln3_scale, ln3_bias)
        return out

    return DecoderDeformableAttention


def build_bevformer_tiny_decoder(config):
    """Build the BEVFormer-tiny decoder (6 stacked self-attn + deformable cross-attn layers)."""
    # Build the 6 attention layers of each type
    layer_0_0 = build_multi_head_self_attention(config)
    layer_0_1 = build_decoder_deformable_attention(config)
    layer_1_0 = build_multi_head_self_attention(config)
    layer_1_1 = build_decoder_deformable_attention(config)
    layer_2_0 = build_multi_head_self_attention(config)
    layer_2_1 = build_decoder_deformable_attention(config)
    layer_3_0 = build_multi_head_self_attention(config)
    layer_3_1 = build_decoder_deformable_attention(config)
    layer_4_0 = build_multi_head_self_attention(config)
    layer_4_1 = build_decoder_deformable_attention(config)
    layer_5_0 = build_multi_head_self_attention(config)
    layer_5_1 = build_decoder_deformable_attention(config)

    # Reference point refinement regression branches
    reg_branch_0 = build_reg_branch(config)
    reg_branch_1 = build_reg_branch(config)
    reg_branch_2 = build_reg_branch(config)
    reg_branch_3 = build_reg_branch(config)
    reg_branch_4 = build_reg_branch(config)
    reg_branch_5 = build_reg_branch(config)

    @script(default_opset=op)
    def BevformerTinyDecoder(
        obj_queries: FLOAT["B", "C", 1, "N_obj"],  # noqa: F821
        bev_features: FLOAT["B", "C", 1, "N_bev"],  # noqa: F821
        ref_points: FLOAT["B", 3, 1, "N_obj"],  # noqa: F821
    ):
        out = obj_queries

        # layer 0
        out_0 = layer_0_0(out)
        out_0 = layer_0_1(out_0, bev_features, ref_points)
        ref_points_0 = op.Gather(reg_branch_0(out_0), op.Constant(value_ints=[0, 1, 4]), axis=1)
        ref_points_0 = Shortcut(ref_points_0, ref_points, mode="addition")  # residual refinement

        # layer 1
        out_1 = layer_1_0(out_0)
        out_1 = layer_1_1(out_1, bev_features, ref_points_0)
        ref_points_1 = op.Gather(reg_branch_1(out_1), op.Constant(value_ints=[0, 1, 4]), axis=1)
        ref_points_1 = Shortcut(ref_points_1, ref_points_0, mode="addition")  # residual refinement

        # layer 2
        out_2 = layer_2_0(out_1)
        out_2 = layer_2_1(out_2, bev_features, ref_points_1)
        ref_points_2 = op.Gather(reg_branch_2(out_2), op.Constant(value_ints=[0, 1, 4]), axis=1)
        ref_points_2 = Shortcut(ref_points_2, ref_points_1, mode="addition")  # residual refinement

        # layer 3
        out_3 = layer_3_0(out_2)
        out_3 = layer_3_1(out_3, bev_features, ref_points_2)
        ref_points_3 = op.Gather(reg_branch_3(out_3), op.Constant(value_ints=[0, 1, 4]), axis=1)
        ref_points_3 = Shortcut(ref_points_3, ref_points_2, mode="addition")  # residual refinement

        # layer 4
        out_4 = layer_4_0(out_3)
        out_4 = layer_4_1(out_4, bev_features, ref_points_3)
        ref_points_4 = op.Gather(reg_branch_4(out_4), op.Constant(value_ints=[0, 1, 4]), axis=1)
        ref_points_4 = Shortcut(ref_points_4, ref_points_3, mode="addition")  # residual refinement

        # layer 5
        out_5 = layer_5_0(out_4)
        out_5 = layer_5_1(out_5, bev_features, ref_points_4)
        ref_points_5 = op.Gather(reg_branch_5(out_5), op.Constant(value_ints=[0, 1, 4]), axis=1)
        ref_points_5 = Shortcut(ref_points_5, ref_points_4, mode="addition")  # residual refinement

        return out_5, ref_points_5

    return BevformerTinyDecoder


def build_reg_branch(config):
    """Build a 3D bounding-box regression branch (MLP) for one decoder layer."""
    # Notation: B=1, N_obj=900, C=256, H=8, D=32 (C/H), L=1 (num_levels), P=4 (num_points)
    #           N_bev=2500 (50×50 BEV), H*L*P=32, H*L*P*2=64
    C = config._dim_

    REG_OUT_CHANNELS = 10  # This is always 10 for 3D bounding branches

    w1_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    w2_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    w3_np = initialize_weight([C, REG_OUT_CHANNELS], dtype=np.float32).T.reshape(REG_OUT_CHANNELS, C, 1, 1)

    b1_np = initialize_weight([C], dtype=np.float32)
    b2_np = initialize_weight([C], dtype=np.float32)
    b3_np = initialize_weight([REG_OUT_CHANNELS], dtype=np.float32)

    @script(default_opset=op)
    def RegressionBranch(x: FLOAT["B", "C", 1, "N_obj"]) -> FLOAT["B", 10, 1, "N_obj"]:  # noqa: F821
        w1 = op.Constant(value=from_array(w1_np))
        w2 = op.Constant(value=from_array(w2_np))
        w3 = op.Constant(value=from_array(w3_np))

        b1 = op.Constant(value=from_array(b1_np))
        b2 = op.Constant(value=from_array(b2_np))
        b3 = op.Constant(value=from_array(b3_np))

        x = vidConv(x, w1, b1)
        x = op.Relu(x)
        x = vidConv(x, w2, b2)
        x = op.Relu(x)
        x = vidConv(x, w3, b3)

        return x

    return RegressionBranch
