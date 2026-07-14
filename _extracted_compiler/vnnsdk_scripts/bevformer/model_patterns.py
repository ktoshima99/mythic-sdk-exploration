# flake8: noqa: D101, D102
# mypy: ignore-errors
"""Rewrite rule patterns for matching parts of the BEVFormer-tiny ONNX graph (not yet wired in)."""

from onnxscript.rewriter import RewriteRuleClassBase


class EncoderTemporalSelfAttentionPattern(RewriteRuleClassBase):
    # Notation: B=1, N_q=2500 (50×50 BEV), C=256, H=8, D=32 (C/H),
    #           Q=2 (num_bev_queue), L=1 (num_levels), P=4 (num_points)

    @classmethod
    def pattern(
        cls,
        op,
        bev_query,  # current BEV queries               [B, N_q, C]    = [1, 2500, 256]
        bev_value_stacked,  # prev+current BEV stacked           [Q, N_q, C]    = [2, 2500, 256]
        bev_pos_enc,  # BEV positional encoding             [B, N_q, C]    = [1, 2500, 256]
        value_proj_w,  # value_proj weight                   [C, C]         = [256, 256]
        value_proj_b,  # value_proj bias                     [C]            = [256]
        sampling_offsets_w,  # sampling_offsets weight             [Q*C, Q*H*L*P*2] = [512, 128]
        sampling_offsets_b,  # sampling_offsets bias               [Q*H*L*P*2]    = [128]
        offset_normalizer,  # spatial shape [bev_w, bev_h] for normalizing raw offsets to [0,1]
        reference_points,  # BEV reference points in [0,1]      broadcasts to [Q, N_q, H, L, P, 2]
        grid_range_scale,  # scalar 2.0: [0,1] → [0,2]
        grid_range_shift,  # scalar 1.0: [0,2] → [-1,1] for GridSample
        attn_weights_w,  # attention_weights weight            [Q*C, Q*H*L*P] = [512, 64]
        attn_weights_b,  # attention_weights bias              [Q*H*L*P]      = [64]
        output_proj_w,  # output_proj weight                  [C, C]         = [256, 256]
        output_proj_b,  # output_proj bias                    [C]            = [256]
        ln_scale,
        ln_bias,
    ):
        # --- Query construction ---
        # Add BEV pos encoding to current query, then concat with sliced prev_bev along the
        # channel dim to form the temporal context query fed into offset/attn projections.
        q = op.Add(bev_query, bev_pos_enc)  # [B, N_q, C]   = [1, 2500, 256]
        prev_bev = op.Slice(
            bev_value_stacked, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_q, C]   = [1, 2500, 256]
        query_combined = op.Concat(prev_bev, q, axis=-1)  # [B, N_q, Q*C] = [1, 2500, 512]

        # --- Value projection ---
        # Project prev+current BEV to per-head values and reshape into a spatial feature map
        # for GridSample (treating each head as an independent "camera").
        v = op.MatMul(bev_value_stacked, value_proj_w)  # [Q, N_q, C]      = [2, 2500, 256]
        v = op.Add(v, value_proj_b)  # [Q, N_q, C]      = [2, 2500, 256]
        v = op.Reshape(v, _allow_other_inputs=True, _allow_other_attributes=True)  # [Q, N_q, H, D]   = [2, 2500, 8, 32]
        v = op.Slice(v, _allow_other_inputs=True)  # [Q, N_q, H, D]   = [2, 2500, 8, 32]
        v = op.Reshape(v, _allow_other_inputs=True, _allow_other_attributes=True)  # [Q, N_q, C]      = [2, 2500, 256]
        v = op.Transpose(v, _allow_other_attributes=True)  # [Q, C, N_q]      = [2, 256, 2500]
        v = op.Reshape(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [Q*H, D, bev_h, bev_w] = [16, 32, 50, 50]

        # --- Sampling offsets ---
        # Project temporal context query to deformable sampling offsets, normalize by BEV
        # spatial shape, add reference points, then convert from [0,1] to [-1,1] for GridSample.
        off = op.MatMul(query_combined, sampling_offsets_w)  # [B, N_q, Q*H*L*P*2]    = [1, 2500, 128]
        off = op.Add(off, sampling_offsets_b)  # [B, N_q, Q*H*L*P*2]    = [1, 2500, 128]
        off = op.Reshape(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_q, H, Q, L, P, 2] = [1, 2500, 8, 2, 1, 4, 2]
        off = op.Transpose(off, _allow_other_attributes=True)  # [B, Q, N_q, H, L, P, 2] = [1, 2, 2500, 8, 1, 4, 2]
        off = op.Reshape(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B*Q, N_q, H, L, P, 2]  = [2, 2500, 8, 1, 4, 2]
        off = op.Div(off, offset_normalizer)  # normalize to [0,1] range
        off = op.Add(off, reference_points)  # sampling_locations in [0,1]
        off = op.Mul(off, grid_range_scale)  # → [0, 2]
        off = op.Sub(off, grid_range_shift)  # → [-1, 1] GridSample range
        grid = op.Gather(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B*Q, N_q, H, P, 2]     = [2, 2500, 8, 4, 2]
        grid = op.Transpose(grid, _allow_other_attributes=True)  # [B*Q, H, N_q, P, 2]     = [2, 8, 2500, 4, 2]
        grid = op.Reshape(
            grid, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [Q*H, N_q, P, 2]        = [16, 2500, 4, 2]

        # --- Attention weights ---
        # Project temporal context query to per-point attention weights, softmax over L*P=4 points,
        # then permute to [Q*H, 1, N_q, P] for element-wise multiplication with sampled features.
        attn = op.MatMul(query_combined, attn_weights_w)  # [B, N_q, Q*H*L*P]    = [1, 2500, 64]
        attn = op.Add(attn, attn_weights_b)  # [B, N_q, Q*H*L*P]    = [1, 2500, 64]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_q, H, Q, L*P]  = [1, 2500, 8, 2, 4]
        attn = op.Softmax(attn, axis=-1)  # [B, N_q, H, Q, L*P]  = [1, 2500, 8, 2, 4]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_q, H, Q, L, P] = [1, 2500, 8, 2, 1, 4]
        attn = op.Transpose(attn, _allow_other_attributes=True)  # [B, Q, N_q, H, L, P] = [1, 2, 2500, 8, 1, 4]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B*Q, N_q, H, L, P]  = [2, 2500, 8, 1, 4]
        attn = op.Transpose(attn, _allow_other_attributes=True)  # [Q*H, L, N_q, P]     = [16, 1, 2500, 4] (approx)
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [Q*H, 1, N_q, P]     = [16, 1, 2500, 4]

        # --- GridSample feature aggregation ---
        # Sample value features at deformable locations, weight by attention scores,
        # sum over sampling points, then average across the BEV queue (prev + current).
        feat = op.GridSample(v, grid, _allow_other_attributes=True)  # [Q*H, D, N_q, P]        = [16, 32, 2500, 4]
        feat = op.Unsqueeze(feat, _allow_other_inputs=True)  # [Q*H, D, N_q, L, P]     = [16, 32, 2500, 1, 4]
        feat = op.Concat(feat, axis=-2)  # [Q*H, D, N_q, L, P]     = [16, 32, 2500, 1, 4]
        feat = op.Reshape(feat, _allow_other_inputs=True, _allow_other_attributes=True)  # compatible shape for attn Mul
        out = op.Mul(feat, attn)  # broadcast attn [Q*H, 1, N_q, P] over D
        out = op.ReduceSum(out, _allow_other_inputs=True, keepdims=0)  # sum over P → [Q*H, D, N_q] = [16, 32, 2500]
        out = op.Reshape(
            out, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [Q, C, N_q]          = [2, 256, 2500]
        out = op.Transpose(out, _allow_other_attributes=True)  # [N_q, C, Q]          = [2500, 256, 2]
        out = op.Reshape(
            out, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_q, C, B, Q]       = [2500, 256, 1, 2]
        out = op.ReduceMean(out, _allow_other_inputs=True, keepdims=0)  # mean over Q → [N_q, C, B] = [2500, 256, 1]
        out = op.Transpose(out, _allow_other_attributes=True)  # [B, N_q, C]          = [1, 2500, 256]

        # --- Output projection and residual ---
        out = op.MatMul(out, output_proj_w)  # [B, N_q, C] = [1, 2500, 256]
        out = op.Add(out, output_proj_b)
        out = op.Add(out, bev_query)  # residual connection
        out = op.LayerNormalization(out, ln_scale, ln_bias)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        bev_query,
        bev_value_stacked,
        bev_pos_enc,
        value_proj_w,
        value_proj_b,
        sampling_offsets_w,
        sampling_offsets_b,
        offset_normalizer,
        reference_points,
        grid_range_scale,
        grid_range_shift,
        attn_weights_w,
        attn_weights_b,
        output_proj_w,
        output_proj_b,
        ln_scale,
        ln_bias,
    ):
        return op.EncoderTemporalSelfAttention(bev_query, bev_value_stacked, _domain="com.videantis", _version=1)


class EncoderSpatialCrossAttentionPattern(RewriteRuleClassBase):
    # Notation: B=1, N_q=2500 (50×50 BEV), C=256, H=8, D=32 (C/H), N_cam=6,
    #           L=1 (num_levels), P_total=8 (num_points), Z=4 (num_Z_anchors), P_per_Z=2 (P_total/Z)
    #           N_cam*H=48, H*L*P_total=64, H*L*P_total*2=128
    #           h_feat × w_feat: image feature spatial dims (e.g. 29×50=1450 for tiny)

    @classmethod
    def pattern(
        cls,
        op,
        bev_query,  # BEV queries (pos enc already added)           [B, N_q, C]              = [1, 2500, 256]
        img_features,  # multi-camera image features                    [N_cam, h_feat*w_feat, C] = [6, 1450, 256]
        bev_mask,  # camera validity mask                           [B, N_cam, N_q, 1]        = [1, 6, 2500, 1]
        ref_pts_cam,  # per-camera 3D→2D projected ref points in [0,1] broadcasts to [N_cam, N_q, H, L, P_per_Z, Z, 2]
        count_normalizer,  # visible-camera count per BEV query             [B, N_q, 1]              = [1, 2500, 1]
        sampling_offsets_w,  # sampling_offsets weight                        [C, H*L*P_total*2]       = [256, 128]
        sampling_offsets_b,  # sampling_offsets bias                          [H*L*P_total*2]          = [128]
        offset_normalizer,  # spatial shape [w_feat, h_feat] for normalizing offsets to [0,1]
        grid_range_scale,  # scalar 2.0: [0,1] → [0,2]
        grid_range_shift,  # scalar 1.0: [0,2] → [-1,1] for GridSample
        value_proj_w,  # value_proj weight                              [C, C]                   = [256, 256]
        value_proj_b,  # value_proj bias                                [C]                      = [256]
        attn_weights_w,  # attention_weights weight                       [C, H*L*P_total]         = [256, 64]
        attn_weights_b,  # attention_weights bias                         [H*L*P_total]            = [64]
        output_proj_w,  # output_proj weight                             [C, C]                   = [256, 256]
        output_proj_b,  # output_proj bias                               [C]                      = [256]
        ffn_fc1_w,  # FFN first  layer weight                        [C, ffn_dim]             = [256, 512]
        ffn_fc1_b,  # FFN first  layer bias                          [ffn_dim]                = [512]
        ffn_fc2_w,  # FFN second layer weight                        [ffn_dim, C]             = [512, 256]
        ffn_fc2_b,  # FFN second layer bias                          [C]                      = [256]
        ln1_scale,
        ln1_bias,
        ln2_scale,
        ln2_bias,
    ):
        # --- Query masking ---
        # Broadcast BEV queries across cameras, then zero out BEV positions not visible
        # from each camera (bev_mask acts as a per-camera visibility gate).
        ref = op.Expand(bev_query, _allow_other_inputs=True)  # [B, N_cam, N_q, C]      = [1, 6, 2500, 256]
        ref = op.Mul(ref, bev_mask)  # [B, N_cam, N_q, C]      = [1, 6, 2500, 256]
        ref = op.Reshape(
            ref, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, N_q, C]         = [6, 2500, 256]

        # --- Sampling offsets ---
        # Project masked queries to deformable offsets; reshape to expose the Z-anchor structure
        # (P_per_Z offsets per depth anchor × Z depth levels); add per-camera projected reference
        # points; collapse back to flat P_total; convert to [-1,1] for GridSample.
        off = op.MatMul(ref, sampling_offsets_w)  # [N_cam, N_q, H*L*P_total*2] = [6, 2500, 128]
        off = op.Add(off, sampling_offsets_b)  # [N_cam, N_q, H*L*P_total*2] = [6, 2500, 128]
        off = op.Reshape(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, N_q, H, L, P_total, 2]  = [6, 2500, 8, 1, 8, 2]
        off = op.Div(off, offset_normalizer)  # normalize by [w_feat, h_feat]
        off = op.Reshape(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, N_q, H, L, P_per_Z, Z, 2] = [6, 2500, 8, 1, 2, 4, 2]
        off = op.Add(off, ref_pts_cam)  # add 3D→2D ref points in [0,1]
        off = op.Reshape(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, N_q, H, L, P_total, 2]  = [6, 2500, 8, 1, 8, 2]
        off = op.Mul(off, grid_range_scale)  # → [0, 2]
        off = op.Sub(off, grid_range_shift)  # → [-1, 1] GridSample range
        grid = op.Gather(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, N_q, H, P_total, 2]     = [6, 2500, 8, 8, 2]
        grid = op.Transpose(grid, _allow_other_attributes=True)  # [N_cam, H, N_q, P_total, 2]     = [6, 8, 2500, 8, 2]
        grid = op.Reshape(
            grid, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam*H, N_q, P_total, 2]      = [48, 2500, 8, 2]

        # --- Value projection ---
        # Project camera features to per-head values and reshape into spatial feature maps
        # (each attention head treated as an independent sample channel).
        v = op.MatMul(img_features, value_proj_w)  # [N_cam, h_feat*w_feat, C]       = [6, 1450, 256]
        v = op.Add(v, value_proj_b)  # [N_cam, h_feat*w_feat, C]       = [6, 1450, 256]
        v = op.Reshape(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, h_feat*w_feat, H, D]    = [6, 1450, 8, 32]
        v = op.Slice(v, _allow_other_inputs=True)  # [N_cam, h_feat*w_feat, H, D]    = [6, 1450, 8, 32]
        v = op.Reshape(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, h_feat*w_feat, C]       = [6, 1450, 256]
        v = op.Transpose(v, _allow_other_attributes=True)  # [N_cam, C, h_feat*w_feat]       = [6, 256, 1450]
        v = op.Reshape(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam*H, D, h_feat, w_feat]    = [48, 32, 29, 50]

        # --- Attention weights ---
        # Project masked queries to per-point attention weights, softmax over L*P_total=8 points,
        # then permute to [N_cam*H, L, N_q, P_total] for element-wise multiplication with features.
        attn = op.MatMul(ref, attn_weights_w)  # [N_cam, N_q, H*L*P_total]       = [6, 2500, 64]
        attn = op.Add(attn, attn_weights_b)  # [N_cam, N_q, H*L*P_total]       = [6, 2500, 64]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, N_q, H, L*P_total]      = [6, 2500, 8, 8]
        attn = op.Softmax(attn, axis=-1)  # [N_cam, N_q, H, L*P_total]      = [6, 2500, 8, 8]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, N_q, H, L, P_total]     = [6, 2500, 8, 1, 8]
        attn = op.Transpose(attn, _allow_other_attributes=True)  # [N_cam, H, N_q, L, P_total]     = [6, 8, 2500, 1, 8]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam*H, L, N_q, P_total]      = [48, 1, 2500, 8]

        # --- GridSample feature aggregation ---
        # Sample per-head features at deformable locations, weight by attention scores, sum over
        # sampling points, then mask by camera visibility and sum over cameras, normalize by count.
        feat = op.GridSample(
            v, grid, _allow_other_attributes=True
        )  # [N_cam*H, D, N_q, P_total]      = [48, 32, 2500, 8]
        feat = op.Unsqueeze(feat, _allow_other_inputs=True)  # [N_cam*H, D, N_q, L, P_total]   = [48, 32, 2500, 1, 8]
        feat = op.Concat(feat, axis=-2)  # [N_cam*H, D, N_q, L, P_total]   = [48, 32, 2500, 1, 8]
        feat = op.Reshape(
            feat, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam*H, D, N_q, P_total]      = [48, 32, 2500, 8]
        out = op.Mul(feat, attn)  # broadcast attn [48, 1, 2500, 8] over D
        out = op.ReduceSum(
            out, _allow_other_inputs=True, keepdims=0
        )  # sum over P_total → [N_cam*H, D, N_q] = [48, 32, 2500]
        out = op.Reshape(
            out, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [N_cam, C, N_q]           = [6, 256, 2500]
        out = op.Transpose(out, _allow_other_attributes=True)  # [N_q, C, N_cam]           = [2500, 256, 6]
        out = op.Reshape(
            out, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_cam, N_q, C]        = [1, 6, 2500, 256]
        out = op.Mul(out, bev_mask)  # zero out invisible cameras
        out = op.ReduceSum(out, _allow_other_inputs=True, keepdims=0)  # sum over cameras → [B, N_q, C] = [1, 2500, 256]
        out = op.Div(out, count_normalizer)  # normalize by visible-camera count

        # --- Output projection and residual ---
        out = op.MatMul(out, output_proj_w)  # [B, N_q, C] = [1, 2500, 256]
        out = op.Add(out, output_proj_b)
        out = op.Add(out, bev_query)  # residual connection

        # --- Feed-forward network ---
        out = op.LayerNormalization(out, ln1_scale, ln1_bias)
        res = out
        out = op.MatMul(out, ffn_fc1_w)  # [B, N_q, ffn_dim] = [1, 2500, 512]
        out = op.Add(out, ffn_fc1_b)
        out = op.Relu(out)
        out = op.MatMul(out, ffn_fc2_w)  # [B, N_q, C] = [1, 2500, 256]
        out = op.Add(out, ffn_fc2_b)
        out = op.Add(out, res)  # residual connection
        out = op.LayerNormalization(out, ln2_scale, ln2_bias)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        bev_query,
        img_features,
        bev_mask,
        ref_pts_cam,
        count_normalizer,
        sampling_offsets_w,
        sampling_offsets_b,
        offset_normalizer,
        grid_range_scale,
        grid_range_shift,
        value_proj_w,
        value_proj_b,
        attn_weights_w,
        attn_weights_b,
        output_proj_w,
        output_proj_b,
        ffn_fc1_w,
        ffn_fc1_b,
        ffn_fc2_w,
        ffn_fc2_b,
        ln1_scale,
        ln1_bias,
        ln2_scale,
        ln2_bias,
    ):
        return op.EncoderSpatialSelfAttention(
            bev_query, img_features, bev_mask, ref_pts_cam, count_normalizer, _domain="com.videantis", _version=1
        )


class DecoderMultiHeadSelfAttentionPattern(RewriteRuleClassBase):
    # Notation: B=1, N_obj=900 (object queries), C=256, H=8 (heads), D=32 (C/H)
    # This is standard scaled dot-product self-attention; Q and K each receive
    # their own learned positional encoding, V uses the raw queries.

    @classmethod
    def pattern(
        cls,
        op,
        obj_queries,  # object queries                          [N_obj, B, C] = [900, 1, 256]
        q_proj_w,  # Q projection weight                     [C, C]        = [256, 256]
        query_pos_enc,  # learned pos encoding added to Q input   [N_obj, B, C] = [900, 1, 256]
        q_scale,  # head scaling 1/sqrt(D) ≈ 0.177          scalar
        key_pos_enc,  # learned pos encoding added to K input   [N_obj, B, C] = [900, 1, 256]
        k_proj_w,  # K projection weight                     [C, C]        = [256, 256]
        k_proj_b,  # K projection bias                       [C]           = [256]
        v_proj_b,  # V projection bias                       [C]           = [256]
        v_proj_w,  # V projection weight                     [C, C]        = [256, 256]
        q_proj_b,  # Q projection bias                       [C]           = [256]
        out_proj_w,  # output projection weight (Gemm, transB) [C, C]        = [256, 256]
        out_proj_b,  # output projection bias                  [C]           = [256]
        ln_scale,
        ln_bias,
    ):
        # --- Q projection ---
        # Add positional encoding, project to Q, split into heads, scale.
        q = op.Add(obj_queries, query_pos_enc)  # [N_obj, B, C]  = [900, 1, 256]
        q = op.MatMul(q, q_proj_w)  # [N_obj, B, C]  = [900, 1, 256]
        q = op.Add(q, q_proj_b)  # [N_obj, B, C]  = [900, 1, 256]
        q = op.Reshape(q, _allow_other_inputs=True, _allow_other_attributes=True)  # [N_obj, H, D]  = [900, 8, 32]
        q = op.Transpose(q, _allow_other_attributes=True)  # [H, N_obj, D]  = [8, 900, 32]
        q = op.Mul(q, q_scale)  # [H, N_obj, D]  = [8, 900, 32]

        # --- K projection ---
        # Add positional encoding, project to K, split into heads, transpose for QKᵀ.
        k = op.Add(obj_queries, key_pos_enc)  # [N_obj, B, C]  = [900, 1, 256]
        k = op.MatMul(k, k_proj_w)  # [N_obj, B, C]  = [900, 1, 256]
        k = op.Add(k, k_proj_b)  # [N_obj, B, C]  = [900, 1, 256]
        k = op.Reshape(k, _allow_other_inputs=True, _allow_other_attributes=True)  # [N_obj, H, D]  = [900, 8, 32]
        k = op.Transpose(k, _allow_other_attributes=True)  # [H, D, N_obj]  = [8, 32, 900]

        # --- V projection ---
        # Project raw queries to V (no positional encoding added), split into heads.
        v = op.MatMul(obj_queries, v_proj_w)  # [N_obj, B, C]  = [900, 1, 256]
        v = op.Add(v, v_proj_b)  # [N_obj, B, C]  = [900, 1, 256]
        v = op.Reshape(v, _allow_other_inputs=True, _allow_other_attributes=True)  # [N_obj, H, D]  = [900, 8, 32]
        v = op.Transpose(v, _allow_other_attributes=True)  # [H, N_obj, D]  = [8, 900, 32]

        # --- Scaled dot-product attention ---
        qk = op.MatMul(q, k)  # [H, N_obj, N_obj] = [8, 900, 900]
        attn = op.Softmax(qk, _allow_other_attributes=True)  # [H, N_obj, N_obj] = [8, 900, 900]

        # --- Weighted aggregation and output projection ---
        out = op.MatMul(attn, v)  # [H, N_obj, D]  = [8, 900, 32]
        out = op.Transpose(out, _allow_other_attributes=True)  # [N_obj, H, D]  = [900, 8, 32]
        out = op.Reshape(out, _allow_other_inputs=True, _allow_other_attributes=True)  # [N_obj, C]     = [900, 256]
        out = op.Gemm(out, out_proj_w, out_proj_b, _allow_other_attributes=True)  # [N_obj, C]     = [900, 256]
        out = op.Reshape(out, _allow_other_inputs=True, _allow_other_attributes=True)  # [N_obj, B, C]  = [900, 1, 256]
        out = op.Add(out, obj_queries)  # residual connection
        out = op.LayerNormalization(out, ln_scale, ln_bias)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        obj_queries,
        q_proj_w,
        query_pos_enc,
        q_scale,
        key_pos_enc,
        k_proj_w,
        k_proj_b,
        v_proj_b,
        v_proj_w,
        q_proj_b,
        out_proj_w,
        out_proj_b,
        ln_scale,
        ln_bias,
    ):
        return op.DecoderMultiHeadSelfAttention(obj_queries, _domain="com.videantis", _version=1)


class DecoderDeformableAttentionPattern(RewriteRuleClassBase):
    # Notation: B=1, N_obj=900, C=256, H=8, D=32 (C/H), L=1 (num_levels), P=4 (num_points)
    #           N_bev=2500 (50×50 BEV), H*L*P=32, H*L*P*2=64
    #
    # Variables marked _unused_* were planned for a reference-point refinement MLP and a
    # pre-block LayerNorm; both are commented out in the active pattern. They remain in the
    # signature so the matcher can bind them, but they do not constrain the match.

    @classmethod
    def pattern(
        cls,
        op,
        obj_queries,  # object queries (post-MHSA, pre-LN)          [N_obj, B, C]         = [900, 1, 256]
        bev_features,  # flattened BEV encoder output                 [B, N_bev, C]         = [1, 2500, 256]
        ref_points,  # precomputed 2D ref pts per query/head        [B, N_obj, 1, 1, 1, 2] = [1, 900, 1, 1, 1, 2]
        query_pos_enc,  # object query positional encoding             [N_obj, B, C]         = [900, 1, 256]
        sampling_offsets_w,  # sampling_offsets weight                      [C, H*L*P*2]          = [256, 64]
        sampling_offsets_b,  # sampling_offsets bias                        [H*L*P*2]             = [64]
        grid_range_scale,  # scalar 2.0: [0,1] → [0,2]
        grid_range_shift,  # scalar 1.0: [0,2] → [-1,1] for GridSample
        value_proj_w,  # value_proj weight                            [C, C]                = [256, 256]
        value_proj_b,  # value_proj bias                              [C]                   = [256]
        attn_weights_w,  # attention_weights weight                     [C, H*L*P]            = [256, 32]
        attn_weights_b,  # attention_weights bias                       [H*L*P]               = [32]
        output_proj_w,  # output projection weight                     [C, C]                = [256, 256]
        output_proj_b,  # output projection bias                       [C]                   = [256]
        offset_normalizer,  # spatial shape [bev_w, bev_h] for offset normalization
        ffn_fc1_w,  # FFN first  layer weight                      [C, ffn_dim]          = [256, 512]
        ffn_fc1_b,  # FFN first  layer bias                        [ffn_dim]             = [512]
        ffn_fc2_w,  # FFN second layer weight                      [ffn_dim, C]          = [512, 256]
        ffn_fc2_b,  # FFN second layer bias                        [C]                   = [256]
        ln2_scale,  # LayerNorm scale after deformable attention
        ln2_bias,
        ln3_scale,  # LayerNorm scale after FFN
        ln3_bias,
    ):
        # --- Query preparation ---
        # Add positional encoding and transpose from [N_obj,B,C] to batch-first [B,N_obj,C]
        # (the deformable attention operates batch-first internally).
        q_with_pos = op.Add(obj_queries, query_pos_enc)  # [N_obj, B, C]         = [900, 1, 256]
        q_with_pos = op.Transpose(q_with_pos, _allow_other_attributes=True)  # [B, N_obj, C]         = [1, 900, 256]

        # --- Attention weights ---
        # Project query to per-point weights, softmax over L*P=4 points per head,
        # then permute to [H, L, N_obj, P] for element-wise multiplication with features.
        attn = op.MatMul(q_with_pos, attn_weights_w)  # [B, N_obj, H*L*P]     = [1, 900, 32]
        attn = op.Add(attn, attn_weights_b)  # [B, N_obj, H*L*P]     = [1, 900, 32]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_obj, H, L*P]    = [1, 900, 8, 4]
        attn = op.Softmax(attn, _allow_other_attributes=True)  # [B, N_obj, H, L*P]    = [1, 900, 8, 4]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_obj, H, L, P]   = [1, 900, 8, 1, 4]
        attn = op.Transpose(attn, _allow_other_attributes=True)  # [B, H, N_obj, L, P]   = [1, 8, 900, 1, 4]
        attn = op.Reshape(
            attn, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [H, L, N_obj, P]      = [8, 1, 900, 4]

        # --- Sampling offsets ---
        # Project query to deformable offsets, normalize by BEV spatial shape,
        # add precomputed reference points, then convert to [-1,1] for GridSample.
        off = op.MatMul(q_with_pos, sampling_offsets_w)  # [B, N_obj, H*L*P*2]   = [1, 900, 64]
        off = op.Add(off, sampling_offsets_b)  # [B, N_obj, H*L*P*2]   = [1, 900, 64]
        off = op.Reshape(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_obj, H, L, P, 2] = [1, 900, 8, 1, 4, 2]
        off = op.Div(off, offset_normalizer)  # normalize by [bev_w, bev_h]
        off = op.Add(off, ref_points)  # sampling_locations in [0,1]
        off = op.Mul(off, grid_range_scale)  # → [0, 2]
        off = op.Sub(off, grid_range_shift)  # → [-1, 1] GridSample range
        grid = op.Gather(
            off, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_obj, H, P, 2]   = [1, 900, 8, 4, 2]
        grid = op.Transpose(grid, _allow_other_attributes=True)  # [B, H, N_obj, P, 2]   = [1, 8, 900, 4, 2]
        grid = op.Reshape(
            grid, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [H, N_obj, P, 2]      = [8, 900, 4, 2]

        # --- Value projection from BEV features ---
        # Project flattened BEV to per-head values, reshape into 2D spatial map for GridSample.
        v = op.MatMul(bev_features, value_proj_w)  # [B, N_bev, C]         = [1, 2500, 256]
        v = op.Add(v, value_proj_b)  # [B, N_bev, C]         = [1, 2500, 256]
        v = op.Reshape(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_bev, H, D]      = [1, 2500, 8, 32]
        v = op.Slice(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_bev, H, D]      = [1, 2500, 8, 32]
        v = op.Reshape(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, N_bev, C]         = [1, 2500, 256]
        v = op.Transpose(v, _allow_other_attributes=True)  # [B, C, N_bev]         = [1, 256, 2500]
        v = op.Reshape(
            v, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [H, D, bev_h, bev_w]  = [8, 32, 50, 50]

        # --- GridSample feature aggregation ---
        # Sample BEV features at deformable locations, weight by attention, sum over points.
        feat = op.GridSample(v, grid, _allow_other_attributes=True)  # [H, D, N_obj, P]      = [8, 32, 900, 4]
        feat = op.Unsqueeze(feat, _allow_other_inputs=True)  # [H, D, N_obj, L, P]   = [8, 32, 900, 1, 4]
        feat = op.Concat(feat, _allow_other_attributes=True)  # [H, D, N_obj, L, P]   = [8, 32, 900, 1, 4]
        feat = op.Reshape(
            feat, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [H, D, N_obj, P]      = [8, 32, 900, 4]
        out = op.Mul(feat, attn)  # broadcast attn [H,L,N_obj,P] over D
        out = op.ReduceSum(out, _allow_other_inputs=True, keepdims=0)  # sum over P → [H, D, N_obj] = [8, 32, 900]
        out = op.Reshape(
            out, _allow_other_inputs=True, _allow_other_attributes=True
        )  # [B, C, N_obj]         = [1, 256, 900]
        out = op.Transpose(out, _allow_other_attributes=True)  # [B, N_obj, C]         = [1, 900, 256]

        # --- Output projection ---
        out = op.MatMul(out, output_proj_w)  # [B, N_obj, C]         = [1, 900, 256]
        out = op.Add(out, output_proj_b)  # [B, N_obj, C]         = [1, 900, 256]
        out = op.Transpose(out, _allow_other_attributes=True)  # [N_obj, B, C]         = [900, 1, 256]

        # --- Residual + LayerNorm + FFN ---
        out = op.Add(out, obj_queries)  # residual connection
        out = op.LayerNormalization(out, ln2_scale, ln2_bias)  # [N_obj, B, C] = [900, 1, 256]
        res = out
        out = op.MatMul(out, ffn_fc1_w)  # [N_obj, B, ffn_dim]   = [900, 1, 512]
        out = op.Add(out, ffn_fc1_b)
        out = op.Relu(out)
        out = op.MatMul(out, ffn_fc2_w)  # [N_obj, B, C]         = [900, 1, 256]
        out = op.Add(out, ffn_fc2_b)
        out = op.Add(out, res)  # residual connection
        out = op.LayerNormalization(out, ln3_scale, ln3_bias)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        obj_queries,
        bev_features,
        ref_points,
        query_pos_enc,
        sampling_offsets_w,
        sampling_offsets_b,
        grid_range_scale,
        grid_range_shift,
        value_proj_w,
        value_proj_b,
        attn_weights_w,
        attn_weights_b,
        output_proj_w,
        output_proj_b,
        offset_normalizer,
        ffn_fc1_w,
        ffn_fc1_b,
        ffn_fc2_w,
        ffn_fc2_b,
        ln2_scale,
        ln2_bias,
        ln3_scale,
        ln3_bias,
    ):
        return op.DecoderDeformableAttention(obj_queries, bev_features, ref_points, _domain="com.videantis", _version=1)
