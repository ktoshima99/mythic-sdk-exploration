# mypy: ignore-errors
import numpy as np
from onnx.numpy_helper import from_array
from onnxscript import FLOAT
from onnxscript import opset20 as op
from onnxscript import script

from .backbone import initialize_weight
from .decoder import build_reg_branch
from vnnort.optimizer.functions import Shortcut, vidConv, vidLayerNorm


def build_classification_branch(config):
    """Build the classification branch (MLP from decoder embeddings to per-class logits)."""
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

    ln1_scale_np = initialize_weight([C], dtype=np.float32)
    ln1_bias_np = initialize_weight([C], dtype=np.float32)
    ln2_scale_np = initialize_weight([C], dtype=np.float32)
    ln2_bias_np = initialize_weight([C], dtype=np.float32)

    @script(default_opset=op)
    def ClassificationBranch(x: FLOAT["B", "C", 1, "N_obj"]) -> FLOAT["B", 10, 1, "N_obj"]:  # noqa: F821
        w1 = op.Constant(value=from_array(w1_np))
        w2 = op.Constant(value=from_array(w2_np))
        w3 = op.Constant(value=from_array(w3_np))

        b1 = op.Constant(value=from_array(b1_np))
        b2 = op.Constant(value=from_array(b2_np))
        b3 = op.Constant(value=from_array(b3_np))

        ln1_scale = op.Constant(value=from_array(ln1_scale_np))
        ln1_bias = op.Constant(value=from_array(ln1_bias_np))
        ln2_scale = op.Constant(value=from_array(ln2_scale_np))
        ln2_bias = op.Constant(value=from_array(ln2_bias_np))

        x = vidConv(x, w1, b1)
        x = vidLayerNorm(x, ln1_scale, ln1_bias)
        # FIXME Add multiplication with 1 [1, 1, 1, 1] to merge Activation Function into Shortcut
        x = Shortcut(x, op.Constant(value=from_array(np.ones((1, 1, 1, 1), dtype=np.float32))), mode="multiplication")
        x = op.Relu(x)
        x = vidConv(x, w2, b2)
        x = vidLayerNorm(x, ln2_scale, ln2_bias)
        # FIXME Add multiplication with 1 [1, 1, 1, 1] to merge Activation Function into Shortcut
        x = Shortcut(x, op.Constant(value=from_array(np.ones((1, 1, 1, 1), dtype=np.float32))), mode="multiplication")
        x = op.Relu(x)

        x = vidConv(x, w3, b3)

        return x

    return ClassificationBranch


def build_postprocessing(config):
    """Build the in-graph postprocessing (classification head + bbox-coord conversion)."""
    # Notation: B=1, N_obj=900, C=256, H=8, D=32 (C/H), L=1 (num_levels), P=4 (num_points)
    #           N_bev=2500 (50×50 BEV), H*L*P=32, H*L*P*2=64
    classificiation_branch = build_classification_branch(config)
    regression_branch = build_reg_branch(config)

    x_min, y_min, z_min, x_max, y_max, z_max = [float(x) for x in config.point_cloud_range]

    x_scale_np = np.array([[[[x_max - x_min]]]], dtype=np.float32)
    y_scale_np = np.array([[[[y_max - y_min]]]], dtype=np.float32)
    z_scale_np = np.array([[[[z_max - z_min]]]], dtype=np.float32)
    x_min_np = np.array([[[[x_min]]]], dtype=np.float32)
    y_min_np = np.array([[[[y_min]]]], dtype=np.float32)
    z_min_np = np.array([[[[z_min]]]], dtype=np.float32)

    @script(default_opset=op)
    def Postprocessing(dec_out: FLOAT["B", "C", 1, "N_obj"], ref_points: FLOAT["B", 3, 1, "N_obj"]):  # noqa: F821
        cls_out = classificiation_branch(dec_out)
        reg_out = regression_branch(dec_out)

        # In the original code, the outputs of the two branches are concatenated and then split again.
        # Here we can just return the classification output, since that's what the postprocessing needs.

        # meaning of the ten regression output channels:
        # [cx, cy, w, l, cz, h, sin(rot), cos(rot), vx, vy]   # raw, unbounded floats
        """
        0-1	cx, cy (BEV center)	logit / inverse-sigmoid (delta on reference)
        2-3	log(w), log(l)	log-size
        4	cz	logit / inverse-sigmoid (delta on reference)
        5	log(h)	log-size
        6-7	sin(rot), cos(rot)	raw
        8-9	vx, vy	raw (m/s)
        """

        cx_cy = op.Slice(reg_out, op.Constant(value_ints=[0]), op.Constant(value_ints=[2]), op.Constant(value_ints=[1]))
        w_l = op.Slice(reg_out, op.Constant(value_ints=[2]), op.Constant(value_ints=[4]), op.Constant(value_ints=[1]))
        cz = op.Slice(reg_out, op.Constant(value_ints=[4]), op.Constant(value_ints=[5]), op.Constant(value_ints=[1]))
        rest = op.Slice(reg_out, op.Constant(value_ints=[5]), op.Constant(value_ints=[10]), op.Constant(value_ints=[1]))

        xy = op.Slice(ref_points, op.Constant(value_ints=[0]), op.Constant(value_ints=[2]), op.Constant(value_ints=[1]))
        z = op.Slice(ref_points, op.Constant(value_ints=[2]), op.Constant(value_ints=[3]), op.Constant(value_ints=[1]))

        cx_cy = Shortcut(cx_cy, xy, mode="addition")
        cx_cy = op.Sigmoid(cx_cy)
        cx = op.Slice(cx_cy, op.Constant(value_ints=[0]), op.Constant(value_ints=[1]), op.Constant(value_ints=[1]))
        cy = op.Slice(cx_cy, op.Constant(value_ints=[1]), op.Constant(value_ints=[2]), op.Constant(value_ints=[1]))
        cx = Shortcut(cx, op.Constant(value=from_array(x_scale_np)), mode="multiplication")
        cx = Shortcut(cx, op.Constant(value=from_array(x_min_np)), mode="addition")
        cy = Shortcut(cy, op.Constant(value=from_array(y_scale_np)), mode="multiplication")
        cy = Shortcut(cy, op.Constant(value=from_array(y_min_np)), mode="addition")

        cz = Shortcut(cz, z, mode="addition")
        cz = op.Sigmoid(cz)
        cz = Shortcut(cz, op.Constant(value=from_array(z_scale_np)), mode="multiplication")
        cz = Shortcut(cz, op.Constant(value=from_array(z_min_np)), mode="addition")

        output_coords = op.Concat(cx, cy, cz, w_l, rest, axis=1)
        return cls_out, output_coords

    return Postprocessing
