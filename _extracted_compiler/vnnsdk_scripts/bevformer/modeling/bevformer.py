# mypy: ignore-errors
from onnxscript import FLOAT
from onnxscript import opset20 as op
from onnxscript import script

from .backbone import (
    build_img_neck,
    build_resnet50_backbone,
)
from .decoder import build_bevformer_tiny_decoder
from .encoder import build_bevformer_tiny_encoder
from .postprocessing import build_postprocessing


def build_bevformer_tiny_transformer(config):
    """Build the BEVFormer-tiny transformer (encoder + decoder + postprocessing).

    Shape symbols (values for BEVFormer-tiny):
        B          batch size (1)
        C          hidden_size / embed_dims (256)
        N_q        BEV cells = bev_h * bev_w (2500)
        N_obj      object queries (900)
        N_cam      cameras (6)
        Q          temporal BEV queue length (N_BEV_QUEUE = 2)
        H_feat     image-feature height; W_feat: image-feature width
        H*L*P*2    encoder deformable-attn channels = num_heads * num_levels * num_points * 2 (64)
        num_classes  classification head channels (10)
    """
    encoder = build_bevformer_tiny_encoder(config)
    decoder = build_bevformer_tiny_decoder(config)
    postprocessing = build_postprocessing(config)

    @script(default_opset=op)
    def BevformerTinyTransformer(
        bev_query: FLOAT["B", "C", 1, "N_q"],  # noqa: F821
        bev_value_stacked: FLOAT["1", "C", "Q", "N_q"],  # noqa: F821
        img_features: FLOAT["1", "C", "H_feat*W_feat", "N_cam"],  # noqa: F821
        bev_mask: FLOAT["B", "1", "N_cam", "N_q"],  # noqa: F821
        ref_pts_cam: FLOAT["1", "H*L*P*2", "N_cam", "N_q"],  # noqa: F821
        count_normalizer: FLOAT["B", "1", "1", "N_q"],  # noqa: F821
        obj_queries: FLOAT["B", "C", 1, "N_obj"],  # noqa: F821
        ref_points: FLOAT["B", 3, 1, "N_obj"],  # noqa: F821
    ) -> tuple[
        FLOAT["B", "num_classes", 1, "N_obj"],  # noqa: F821
        FLOAT["B", 10, 1, "N_obj"],  # noqa: F821
        FLOAT["B", "C", 1, "N_q"],  # noqa: F821
    ]:
        """Run encoder → decoder → in-graph postprocessing.

        Inputs (see signature for shapes):
            * ``bev_query`` — learned BEV positional/query embeddings, one per BEV cell.
            * ``bev_value_stacked`` — temporal BEV value stack (previous + current frame's BEV)
              used by the encoder's temporal self-attention. Q = N_BEV_QUEUE.
            * ``img_features`` — multi-camera image features after backbone+FPN (or supplied
              directly when running the transformer-only graph). Per
              ``bevformer_tiny.BevformerTiny._input_shapes`` the cameras are stored in the last
              axis.
            * ``bev_mask`` — per-BEV-cell visibility (which of the N_cam cameras can see each
              cell).
            * ``ref_pts_cam`` — 3D BEV reference points projected into each camera's image
              plane, flattened across heads, levels, and points (H*L*P*2 channels: xy per
              head/level/point).
            * ``count_normalizer`` — reciprocal of the number of visible cameras per BEV cell,
              applied after spatial cross-attention to average contributions.
            * ``obj_queries`` — learned object query embeddings (N_obj = 900 detection slots).
            * ``ref_points`` — initial 3D reference points (x, y, z) for each object query.

        Outputs (tuple):
            * ``cls_out`` — per-query class logits (num_classes = 10 for the nuScenes config).
            * ``output_coords`` — per-query 10-channel box parameters [cx, cy, cz, log(w),
              log(l), log(h), sin(rot), cos(rot), vx, vy] (post-sigmoid for cx/cy/cz, rescaled
              into the point-cloud range).
            * ``prev_bev`` — updated BEV embedding for the next frame's temporal queue.
        """
        prev_bev = encoder(bev_query, bev_value_stacked, img_features, bev_mask, ref_pts_cam, count_normalizer)
        dec_out, ref_points_out = decoder(obj_queries, prev_bev, ref_points)
        cls_out, output_coords = postprocessing(dec_out, ref_points_out)
        return cls_out, output_coords, prev_bev

    return BevformerTinyTransformer


def build_bevformer_tiny(config):
    """Build the full BEVFormer-tiny model (backbone + neck + transformer).

    Same shape symbols as `build_bevformer_tiny_transformer`. The difference is that the six
    camera images are passed in (concatenated along width) and `img_features` is produced
    internally by the ResNet-50 backbone + FPN neck.
    """
    img_backbone = build_resnet50_backbone()
    img_neck = build_img_neck(config)
    transformer = build_bevformer_tiny_transformer(config)

    @script(default_opset=op)
    def BevformerTiny(
        images: FLOAT["B", 3, "H_img", "N_cam*W_img"],  # noqa: F821
        bev_query: FLOAT["B", "C", 1, "N_q"],  # noqa: F821
        bev_value_stacked: FLOAT["1", "C", "Q", "N_q"],  # noqa: F821
        bev_mask: FLOAT["B", "1", "N_cam", "N_q"],  # noqa: F821
        ref_pts_cam: FLOAT["1", "H*L*P*2", "N_cam", "N_q"],  # noqa: F821
        count_normalizer: FLOAT["B", "1", "1", "N_q"],  # noqa: F821
        obj_queries: FLOAT["B", "C", 1, "N_obj"],  # noqa: F821
        ref_points: FLOAT["B", 3, 1, "N_obj"],  # noqa: F821
    ) -> tuple[
        FLOAT["B", "num_classes", 1, "N_obj"],  # noqa: F821
        FLOAT["B", 10, 1, "N_obj"],  # noqa: F821
        FLOAT["B", "C", 1, "N_q"],  # noqa: F821
    ]:
        """Run backbone+neck on the six-camera image, then the transformer.

        ``images`` is six camera frames stacked along width into a single
        Bx3xH_imgx(N_cam·W_img) tensor (the layout expected by ``Resnet50`` in this codebase).
        See ``BevformerTinyTransformer`` for the remaining inputs and outputs.
        """
        img_feats = img_backbone(images)
        img_feats = img_neck(img_feats)
        cls_out, output_coords, prev_bev = transformer(
            bev_query, bev_value_stacked, img_feats, bev_mask, ref_pts_cam, count_normalizer, obj_queries, ref_points
        )
        return cls_out, output_coords, prev_bev

    return BevformerTiny
