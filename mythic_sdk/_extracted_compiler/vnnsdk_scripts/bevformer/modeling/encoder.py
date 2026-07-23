# mypy: ignore-errors
import numpy as np
from onnx.numpy_helper import from_array
from onnxscript import FLOAT
from onnxscript import opset20 as op
from onnxscript import script

from .backbone import initialize_weight
from vnnort.optimizer.functions import Shortcut, vidConv, vidLayerNorm, vidSoftmax, vidGridSample


def build_temporal_self_attention(config):
    """Build the temporal self-attention sub-layer of the BEVFormer-tiny encoder."""
    C = config._dim_
    bev_h = config.bev_h_
    bev_w = config.bev_w_
    N_q = bev_h * bev_w
    Q = 2  # num_bev_queue
    H = config.model["pts_bbox_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][0]["num_heads"]
    D = C // H
    L = config.model["pts_bbox_head"]["transformer"]["encoder"]["transformerlayers"]["attn_cfgs"][0]["num_levels"]
    P = config.model["pts_bbox_head"]["transformer"]["encoder"]["num_points_in_pillar"]
    B = 1  # batch size

    bev_pos_enc_np = initialize_weight([B, N_q, C], dtype=np.float32).transpose(0, 2, 1).reshape(B, C, 1, N_q)
    value_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    value_proj_b_np = initialize_weight([C], dtype=np.float32)
    sampling_offsets_w_np = initialize_weight([Q * C, Q * H * L * P * 2], dtype=np.float32).T.reshape(
        Q * H * L * P * 2, Q * C, 1, 1
    )
    sampling_offsets_b_np = initialize_weight([Q * H * L * P * 2], dtype=np.float32)
    attn_weights_w_np = initialize_weight([Q * C, Q * H * L * P], dtype=np.float32).T.reshape(
        Q * H * L * P, Q * C, 1, 1
    )
    attn_weights_b_np = initialize_weight([Q * H * L * P], dtype=np.float32)
    output_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    output_proj_b_np = initialize_weight([C], dtype=np.float32)
    ln_scale_np = initialize_weight([C], dtype=np.float32)
    ln_bias_np = initialize_weight([C], dtype=np.float32)

    reference_points_np = np.full([Q, N_q, H, L, P, 2], 0.5, dtype=np.float32)
    reference_points_np = reference_points_np.transpose([0, 2, 3, 4, 5, 1]).reshape([1, Q * H * L * P * 2, 1, N_q])

    # Absorb Div by offset_normalizer=[w_feat, h_feat] into the Conv weight and bias.
    # After reshape to [N_cam, H, L, P_total, 2, N_q] (N_q at end), output channel c always maps to
    # xy = c % 2, so scaling row c by 1/normalizer[c % 2] is equivalent to the element-wise Div.
    m_per_channel = np.tile(np.array([1.0 / bev_h, 1.0 / bev_w], dtype=np.float32), Q * H * L * P)
    sampling_offsets_w_np = sampling_offsets_w_np * m_per_channel[:, np.newaxis, np.newaxis, np.newaxis]
    sampling_offsets_b_np = sampling_offsets_b_np * m_per_channel

    @script(default_opset=op)
    def TemporalSelfAttention(
        bev_query: FLOAT["B", "C", 1, "N_q"], bev_value_stacked: FLOAT[1, "C", "Q", "N_q"]
    ) -> FLOAT["B", "C", 1, "N_q"]:
        bev_pos_enc = op.Constant(value=from_array(bev_pos_enc_np))
        value_proj_w = op.Constant(value=from_array(value_proj_w_np))
        value_proj_b = op.Constant(value=from_array(value_proj_b_np))
        sampling_offsets_w = op.Constant(value=from_array(sampling_offsets_w_np))
        sampling_offsets_b = op.Constant(value=from_array(sampling_offsets_b_np))
        attn_weights_w = op.Constant(value=from_array(attn_weights_w_np))
        attn_weights_b = op.Constant(value=from_array(attn_weights_b_np))
        output_proj_w = op.Constant(value=from_array(output_proj_w_np))
        output_proj_b = op.Constant(value=from_array(output_proj_b_np))
        ln_scale = op.Constant(value=from_array(ln_scale_np))
        ln_bias = op.Constant(value=from_array(ln_bias_np))
        reference_points = op.Constant(value=from_array(reference_points_np))

        # --- Query construction ---
        q = Shortcut(bev_query, bev_pos_enc)  # [B, C, 1, N_q]
        prev_bev = op.Slice(
            bev_value_stacked,
            op.Constant(value=from_array(np.array([0], dtype=np.int64))),
            op.Constant(value=from_array(np.array([1], dtype=np.int64))),
            op.Constant(value=from_array(np.array([2], dtype=np.int64))),
        )  # [1, C, 1, N_q]
        query_combined = op.Concat(prev_bev, q, axis=1)  # [B, 2*C, 1, N_q]

        # --- Value projection ---
        v = vidConv(
            bev_value_stacked, value_proj_w, value_proj_b, kernel_shape=[1, 1]
        )  # [B, C, Q, N_q] [1, 256, 2, 2500]
        v = op.Transpose(v, perm=[2, 0, 1, 3])  # [2, 1, 256, 2500]
        v = op.Reshape(v, op.Constant(value_ints=[Q * H, D, bev_h, bev_w]))  # [Q*H, D, bev_h, bev_w]

        # # --- Sampling offsets (m = scale/normalizer already absorbed into weight and bias) ---
        off = vidConv(query_combined, sampling_offsets_w, sampling_offsets_b, kernel_shape=[1, 1])
        off = Shortcut(off, reference_points)
        off = Shortcut(
            off, op.Constant(value=from_array(np.array([[[[2.0]]]], dtype=np.float32))), mode="multiplication"
        )  # → [0, 2]
        off = Shortcut(
            off, op.Constant(value=from_array(np.array([[[[-1.0]]]], dtype=np.float32))), mode="addition"
        )  # → [-1, 1] for GridSample
        off = op.Reshape(off, op.Constant(value_ints=[B, H, Q, L, P, 2, N_q]))
        off = op.Transpose(off, perm=[0, 2, 1, 5, 6, 3, 4])  # [B, H, Q, 2, N_q, L, P]
        grid = op.Reshape(off, op.Constant(value_ints=[B * H * Q, 2, N_q, L * P]))

        # --- Attention weights ---
        attn = vidConv(query_combined, attn_weights_w, attn_weights_b, kernel_shape=[1, 1])  # [B, Q*H*L*P, 1, N_q]
        attn = vidSoftmax(attn, group=[H * Q])
        attn = op.Reshape(attn, op.Constant(value_ints=[B, Q, H, L, P, N_q]))
        attn = op.Transpose(attn, perm=[0, 3, 4, 1, 2, 5])  # [B, L, P, Q, H, N_q]
        attn = op.Reshape(attn, op.Constant(value_ints=[B, L * P, 1, Q * H * N_q]))

        # --- GridSample feature aggregation ---
        feat = vidGridSample(v, grid, align_corners=0, mode="linear", padding_mode="zeros")  # [H*Q, D, N_q, P]

        feat = op.Reshape(feat, op.Constant(value_ints=[H, Q, D, N_q, P]))  # [H, Q, D, N_q, P]
        feat = op.Transpose(feat, perm=[4, 2, 0, 1, 3])  # [P, D, H, Q, N_q]
        feat = op.Reshape(feat, op.Constant(value_ints=[B, P, D, H * Q * N_q]))

        out = Shortcut(feat, attn, mode="multiplication")

        # Create [1, P, 1, 1] convolution to produce sum over number of point per pillar
        out = vidConv(
            out, op.Constant(value=from_array(np.ones((1, P, 1, 1), dtype=np.float32))), None, kernel_shape=[1, 1]
        )

        # Create [1, Q, 1, 1] convolution to produce mean over number prev bevs in queue
        out = op.Reshape(out, op.Constant(value_ints=[1, D, Q, H, N_q]))
        out = op.Transpose(out, perm=[0, 2, 3, 1, 4])  # [1, Q, H, D, N_q]
        out = op.Reshape(out, op.Constant(value_ints=[1, Q, H * D, N_q]))  # [1, Q*D, H*N_q]
        out = vidConv(
            out, op.Constant(value=from_array(np.ones((1, Q, 1, 1), dtype=np.float32) / Q)), None, kernel_shape=[1, 1]
        )

        # # Output projection
        out = op.Reshape(out, op.Constant(value_ints=[1, H * D, 1, N_q]))
        out = vidConv(
            out,
            output_proj_w,
            output_proj_b,
            kernel_shape=[1, 1],
        )  # [B, C, 1, N_q]

        # # Residual and LayerNorm
        out = Shortcut(out, bev_query)
        out = vidLayerNorm(out, ln_scale, ln_bias)
        return out

    return TemporalSelfAttention


def build_spatial_cross_attention(config):
    """Build the spatial cross-attention sub-layer of the BEVFormer-tiny encoder."""
    C = config._dim_
    bev_h = config.bev_h_
    bev_w = config.bev_w_
    N_q = bev_h * bev_w
    H = config.model["pts_bbox_head"]["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][0]["num_heads"]
    D = C // H
    L = config.model["pts_bbox_head"]["transformer"]["encoder"]["transformerlayers"]["attn_cfgs"][1][
        "deformable_attention"
    ]["num_levels"]
    P_total = config.model["pts_bbox_head"]["transformer"]["encoder"]["transformerlayers"]["attn_cfgs"][1][
        "deformable_attention"
    ]["num_points"]
    ffn_dim = config._ffn_dim_
    N_cam = 6
    h_feat, w_feat = 29, 50
    B = 1

    sampling_offsets_w_np = initialize_weight([C, H * L * P_total * 2], dtype=np.float32).T.reshape(
        H * L * P_total * 2, C, 1, 1
    )
    sampling_offsets_b_np = initialize_weight([H * L * P_total * 2], dtype=np.float32)
    # Absorb Div by offset_normalizer=[w_feat, h_feat] into the Conv weight and bias.
    # After reshape to [N_cam, H, L, P_total, 2, N_q] (N_q at end), output channel c always maps to
    # xy = c % 2, so scaling row c by 1/normalizer[c % 2] is equivalent to the element-wise Div.
    m_per_channel = np.tile(np.array([1.0 / w_feat, 1.0 / h_feat], dtype=np.float32), H * L * P_total)
    sampling_offsets_w_np = sampling_offsets_w_np * m_per_channel[:, np.newaxis, np.newaxis, np.newaxis]
    sampling_offsets_b_np = sampling_offsets_b_np * m_per_channel
    value_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    value_proj_b_np = initialize_weight([C], dtype=np.float32)
    attn_weights_w_np = initialize_weight([C, H * L * P_total], dtype=np.float32).T.reshape(H * L * P_total, C, 1, 1)
    attn_weights_b_np = initialize_weight([H * L * P_total], dtype=np.float32)
    output_proj_w_np = initialize_weight([C, C], dtype=np.float32).T.reshape(C, C, 1, 1)
    output_proj_b_np = initialize_weight([C], dtype=np.float32)
    ffn_fc1_w_np = initialize_weight([C, ffn_dim], dtype=np.float32).T.reshape(ffn_dim, C, 1, 1)
    ffn_fc1_b_np = initialize_weight([ffn_dim], dtype=np.float32)
    ffn_fc2_w_np = initialize_weight([ffn_dim, C], dtype=np.float32).T.reshape(C, ffn_dim, 1, 1)
    ffn_fc2_b_np = initialize_weight([C], dtype=np.float32)
    ln1_scale_np = initialize_weight([C], dtype=np.float32)
    ln1_bias_np = initialize_weight([C], dtype=np.float32)
    ln2_scale_np = initialize_weight([C], dtype=np.float32)
    ln2_bias_np = initialize_weight([C], dtype=np.float32)

    grid_range_scale_np = np.array([[[[2.0]]]], dtype=np.float32)
    grid_range_shift_np = np.array([[[[-1.0]]]], dtype=np.float32)  # Negative to transform sub into add

    @script(default_opset=op)
    def SpatialCrossAttention(
        bev_query: FLOAT["B", "C", 1, "N_q"],
        img_features: FLOAT["B", "C", "N_cam", "h_feat_w_feat"],  # noqa: F821
        bev_mask: FLOAT["B", 1, "N_cam", "N_q"],
        ref_pts_cam: FLOAT["B", "C", "N_cam", "N_q"],
        count_normalizer: FLOAT[
            "B",
            1,
            1,
            "N_q",
        ],
    ) -> FLOAT["B", "N_q", "C"]:
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
        ln1_scale = op.Constant(value=from_array(ln1_scale_np))
        ln1_bias = op.Constant(value=from_array(ln1_bias_np))
        ln2_scale = op.Constant(value=from_array(ln2_scale_np))
        ln2_bias = op.Constant(value=from_array(ln2_bias_np))
        grid_range_scale = op.Constant(value=from_array(grid_range_scale_np))
        grid_range_shift = op.Constant(value=from_array(grid_range_shift_np))

        # --- Query masking ---

        # # --- Sampling offsets ---
        ref = Shortcut(bev_query, bev_mask, mode="multiplication")  # [B, N_q, C] and [B, C, N_cam, N_q]
        off = vidConv(ref, sampling_offsets_w, sampling_offsets_b, kernel_shape=[1, 1])
        off = Shortcut(off, ref_pts_cam, mode="addition")
        off = Shortcut(off, grid_range_scale, mode="multiplication")
        off = Shortcut(off, grid_range_shift, mode="addition")
        grid = op.Reshape(off, op.Constant(value_ints=[B, H, L, P_total, 2, N_cam, N_q]))
        grid = op.Transpose(grid, perm=[0, 4, 5, 1, 6, 2, 3])
        grid = op.Reshape(grid, op.Constant(value_ints=[N_cam * H, 2, N_q, P_total]))

        # # --- Value projection ---
        v = vidConv(img_features, value_proj_w, value_proj_b, kernel_shape=[1, 1])  # [B, C, N_cam, h_feat*w_feat]
        v = op.Transpose(v, perm=[0, 2, 1, 3])  # [B, N_cam, C, h_feat*w_feat]
        v = op.Reshape(v, op.Constant(value_ints=[-1, D, h_feat, w_feat]))  # [N_cam, C, h_feat*w_feat]

        # # --- Attention weights ---
        attn = vidConv(ref, attn_weights_w, attn_weights_b, kernel_shape=[1, 1])  # [B, H*L*P_total, N_cam, N_q]
        attn = vidSoftmax(attn, group=[H])
        attn = op.Reshape(attn, op.Constant(value_ints=[B, H, P_total, N_cam, N_q]))
        attn = op.Transpose(attn, perm=[0, 2, 3, 1, 4])
        attn = op.Reshape(attn, op.Constant(value_ints=[B, P_total, 1, -1]))

        # # --- GridSample feature aggregation ---
        feat = vidGridSample(v, grid, align_corners=0, mode="linear", padding_mode="zeros")

        # Current: [48, 32, 2500, 8] = [N_cam*H, D, N_q, P_total]
        # Target [B, P_total, D, N_cam*H*N_q]
        feat = op.Transpose(feat, perm=[3, 1, 0, 2])
        feat = op.Reshape(feat, op.Constant(value_ints=[B, P_total, D, N_cam * H * N_q]))

        out = Shortcut(feat, attn, mode="multiplication")  # [1, P_total, D, N_cam*H*N_q]
        # Create [1, P_total, 1, 1] convolution to produce sum over number of point per pillar
        out = vidConv(
            out, op.Constant(value=from_array(np.ones((1, P_total, 1, 1), dtype=np.float32))), None, kernel_shape=[1, 1]
        )
        out = op.Reshape(out, op.Constant(value_ints=[B, D, N_cam, H, N_q]))
        out = op.Transpose(out, perm=[0, 2, 3, 1, 4])
        out = op.Reshape(out, op.Constant(value_ints=[B, N_cam, H * D, N_q]))

        # FIXME THIS IS ONLY TEMPORARY TO AVOID SKIP LAYER ISSUE IN AIMAP
        bev_mask = Shortcut(
            bev_mask, op.Constant(value=from_array(np.array([[[[1.0]]]], dtype=np.float32))), mode="multiplication"
        )

        bev_mask_reshaped = op.Reshape(bev_mask, op.Constant(value_ints=[B, N_cam, 1, N_q]))
        out = Shortcut(out, bev_mask_reshaped, mode="multiplication")  # zero out invisible cameras

        # Create [1, N_cam, 1, 1] convolution to produce sum over number of cameras
        out = vidConv(
            out, op.Constant(value=from_array(np.ones((1, N_cam, 1, 1), dtype=np.float32))), None, kernel_shape=[1, 1]
        )
        out = op.Reshape(out, op.Constant(value_ints=[B, C, 1, N_q]))
        out = Shortcut(out, count_normalizer, mode="division")

        # --- Output projection and residual ---
        out = vidConv(out, output_proj_w, output_proj_b, kernel_shape=[1, 1])  # [B, C, 1, N_q]
        out = Shortcut(out, bev_query, mode="addition")

        # # --- Feed-forward network ---
        out = vidLayerNorm(out, ln1_scale, ln1_bias)
        res = out
        out = vidConv(out, ffn_fc1_w, ffn_fc1_b, kernel_shape=[1, 1])  # [B, ffn_dim, 1, N_q]
        out = op.Relu(out)
        out = vidConv(out, ffn_fc2_w, ffn_fc2_b, kernel_shape=[1, 1])  # [B, C, 1, N_q]
        out = Shortcut(out, res, mode="addition")  # residual
        out = vidLayerNorm(out, ln2_scale, ln2_bias)
        return out

    return SpatialCrossAttention


def build_bevformer_tiny_encoder(config):
    """Build the BEVFormer-tiny encoder (3 stacked temporal + spatial attention layers)."""
    # Build the 3 layers
    layer_0_0 = build_temporal_self_attention(config)
    layer_0_1 = build_spatial_cross_attention(config)
    layer_1_0 = build_temporal_self_attention(config)
    layer_1_1 = build_spatial_cross_attention(config)
    layer_2_0 = build_temporal_self_attention(config)
    layer_2_1 = build_spatial_cross_attention(config)

    @script(default_opset=op)
    def BevformerTinyEncoder(
        bev_query: FLOAT["B", "C", 1, "N_q"],  # noqa: F821
        bev_value_stacked: FLOAT["1", "C", "Q", "N_q"],  # noqa: F821
        img_features: FLOAT["1", "C", "N_cam", "H_feat*W_feat"],  # noqa: F821
        bev_mask: FLOAT["B", "1", "N_cam", "N_q"],  # noqa: F821
        ref_pts_cam: FLOAT["1", "N_cam", "N_q", "-1"],  # noqa: F821
        count_normalizer: FLOAT["B", "1", "1", "N_q"],  # noqa: F821
    ) -> FLOAT["B", "C", 1, "N_q"]:  # noqa: F821
        out = bev_query
        out = layer_0_0(out, bev_value_stacked)
        out = layer_0_1(out, img_features, bev_mask, ref_pts_cam, count_normalizer)
        out = layer_1_0(out, bev_value_stacked)
        out = layer_1_1(out, img_features, bev_mask, ref_pts_cam, count_normalizer)
        out = layer_2_0(out, bev_value_stacked)
        out = layer_2_1(out, img_features, bev_mask, ref_pts_cam, count_normalizer)
        return out

    return BevformerTinyEncoder
