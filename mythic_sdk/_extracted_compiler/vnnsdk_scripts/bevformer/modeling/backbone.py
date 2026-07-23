# mypy: ignore-errors
import numpy as np
from onnx.numpy_helper import from_array
from onnxscript import FLOAT
from onnxscript import opset20 as op
from onnxscript import script, values

from vnnort.optimizer.functions import Shortcut, vidConv, vidMaxPool


def initialize_weight(shape, dtype):
    """Return a deterministic numpy weight tensor of the given shape and dtype."""
    return np.random.random(shape).astype(dtype)


def build_resnet_block(c_in, c_mid, c_out, downsample, strides=(1, 1)):
    """Build one ResNet bottleneck block as an onnxscript function."""
    if not hasattr(build_resnet_block, "index"):
        build_resnet_block.index = 0
    else:
        build_resnet_block.index += 1

    if not downsample:
        w1_np = initialize_weight([c_mid, c_in, 1, 1], dtype=np.float32)
        b1_np = initialize_weight([c_mid], dtype=np.float32)

        w2_np = initialize_weight([c_mid, c_mid, 3, 3], dtype=np.float32)
        b2_np = initialize_weight([c_mid], dtype=np.float32)

        w3_np = initialize_weight([c_out, c_mid, 1, 1], dtype=np.float32)
        b3_np = initialize_weight([c_out], dtype=np.float32)

        @script(values.Opset(f"com.videantis.dynamic_functions.ResnetBlock{build_resnet_block.index}", 1))
        def ResnetBlock(x: FLOAT) -> FLOAT:
            identity = x

            w1 = op.Constant(value=from_array(w1_np))
            b1 = op.Constant(value=from_array(b1_np))
            x = vidConv(x, w1, b1, kernel_shape=(1, 1))
            x = op.Relu(x)

            w2 = op.Constant(value=from_array(w2_np))
            b2 = op.Constant(value=from_array(b2_np))
            x = vidConv(x, w2, b2, kernel_shape=(3, 3), pads=(1, 1, 1, 1))
            x = op.Relu(x)

            w3 = op.Constant(value=from_array(w3_np))
            b3 = op.Constant(value=from_array(b3_np))
            x = vidConv(x, w3, b3, kernel_shape=(1, 1))

            x = Shortcut(x, identity)
            x = op.Relu(x)
            return x

        return ResnetBlock

    else:
        w1_np = initialize_weight([c_mid, c_in, 1, 1], dtype=np.float32)
        b1_np = initialize_weight([c_mid], dtype=np.float32)

        w2_np = initialize_weight([c_mid, c_mid, 3, 3], dtype=np.float32)
        b2_np = initialize_weight([c_mid], dtype=np.float32)

        w3_np = initialize_weight([c_out, c_mid, 1, 1], dtype=np.float32)
        b3_np = initialize_weight([c_out], dtype=np.float32)

        w4_np = initialize_weight([c_out, c_in, 1, 1], dtype=np.float32)
        b4_np = initialize_weight([c_out], dtype=np.float32)

        @script(values.Opset(f"com.videantis.dynamic_functions.ResnetBlock{build_resnet_block.index}", 1))
        def ResnetBlock(x: FLOAT) -> FLOAT:
            identity = x

            w1 = op.Constant(value=from_array(w1_np))
            b1 = op.Constant(value=from_array(b1_np))
            x = vidConv(x, w1, b1, kernel_shape=(1, 1))
            x = op.Relu(x)

            w2 = op.Constant(value=from_array(w2_np))
            b2 = op.Constant(value=from_array(b2_np))
            x = vidConv(x, w2, b2, kernel_shape=(3, 3), pads=(1, 1, 1, 1), strides=strides)
            x = op.Relu(x)

            w3 = op.Constant(value=from_array(w3_np))
            b3 = op.Constant(value=from_array(b3_np))
            x = vidConv(x, w3, b3, kernel_shape=(1, 1))

            w4 = op.Constant(value=from_array(w4_np))
            b4 = op.Constant(value=from_array(b4_np))
            identity = vidConv(identity, w4, b4, kernel_shape=(1, 1), strides=strides)

            x = Shortcut(x, identity)
            x = op.Relu(x)
            return x

        return ResnetBlock


def build_resnet50_backbone():
    """Build the ResNet-50 image backbone as an onnxscript function."""
    w1_np = initialize_weight([64, 3, 7, 7], dtype=np.float32)
    b1_np = initialize_weight([64], dtype=np.float32)

    # Stage0 (3 layers)
    block0 = build_resnet_block(64, 64, 256, downsample=True, strides=(1, 1))
    block1 = build_resnet_block(256, 64, 256, downsample=False)
    block2 = build_resnet_block(256, 64, 256, downsample=False)

    # Stage1 (4 layers)
    block3 = build_resnet_block(256, 128, 512, downsample=True, strides=(2, 2))
    block4 = build_resnet_block(512, 128, 512, downsample=False)
    block5 = build_resnet_block(512, 128, 512, downsample=False)
    block6 = build_resnet_block(512, 128, 512, downsample=False)

    # Stage2 (6 layers)
    block7 = build_resnet_block(512, 256, 1024, downsample=True, strides=(2, 2))
    block8 = build_resnet_block(1024, 256, 1024, downsample=False)
    block9 = build_resnet_block(1024, 256, 1024, downsample=False)
    block10 = build_resnet_block(1024, 256, 1024, downsample=False)
    block11 = build_resnet_block(1024, 256, 1024, downsample=False)
    block12 = build_resnet_block(1024, 256, 1024, downsample=False)

    #  Stage3 (3 layers)
    block13 = build_resnet_block(1024, 512, 2048, downsample=True, strides=(2, 2))
    block14 = build_resnet_block(2048, 1024, 2048, downsample=False)
    block15 = build_resnet_block(2048, 1024, 2048, downsample=False)

    @script(default_opset=op)
    def Resnet50(x: FLOAT) -> FLOAT:
        """Run the resnet50 bevformer backbone.

        Args:
            x (FLOAT): [1, 3, H, 6*W]

        Returns:
            FLOAT: [1, C, Hout, 6*Wout]
        """
        w1 = op.Constant(value=from_array(w1_np))
        b1 = op.Constant(value=from_array(b1_np))
        x = vidConv(x, w1, b1, pads=[3, 3, 3, 3], strides=[2, 2], kernel_shape=(7, 7))
        x = op.Relu(x)
        x = vidMaxPool(x, kernel_shape=(3, 3), pads=(0, 0, 1, 1), strides=(2, 2))

        # Stage 0
        x = block0(x)
        x = block1(x)
        x = block2(x)

        # Stage 1
        x = block3(x)
        x = block4(x)
        x = block5(x)
        x = block6(x)

        # Stage 2
        x = block7(x)
        x = block8(x)
        x = block9(x)
        x = block10(x)
        x = block11(x)
        x = block12(x)

        # Stage 3
        x = block13(x)
        x = block14(x)
        x = block15(x)

        return x

    return Resnet50


def build_resnet101_backbone():
    """Build the ResNet-101 image backbone as an onnxscript function."""
    # Similar to build_resnet50_backbone but with more blocks in stage 2 and stage 3
    # Stage0 (3 layers)
    block0 = build_resnet_block(64, 64, 256, downsample=True, strides=(1, 1))
    block1 = build_resnet_block(256, 64, 256, downsample=False)
    block2 = build_resnet_block(256, 64, 256, downsample=False)

    # Stage1 (4 layers)
    block3 = build_resnet_block(256, 128, 512, downsample=True, strides=(2, 2))
    block4 = build_resnet_block(512, 128, 512, downsample=False)
    block5 = build_resnet_block(512, 128, 512, downsample=False)
    block6 = build_resnet_block(512, 128, 512, downsample=False)

    # Stage2 (23 layers)
    block7 = build_resnet_block(512, 256, 1024, downsample=True, strides=(2, 2))
    block8 = build_resnet_block(1024, 256, 1024, downsample=False)
    block9 = build_resnet_block(1024, 256, 1024, downsample=False)
    block10 = build_resnet_block(1024, 256, 1024, downsample=False)
    block11 = build_resnet_block(1024, 256, 1024, downsample=False)
    block12 = build_resnet_block(1024, 256, 1024, downsample=False)
    block13 = build_resnet_block(1024, 256, 1024, downsample=False)
    block14 = build_resnet_block(1024, 256, 1024, downsample=False)
    block15 = build_resnet_block(1024, 256, 1024, downsample=False)
    block16 = build_resnet_block(1024, 256, 1024, downsample=False)
    block17 = build_resnet_block(1024, 256, 1024, downsample=False)
    block18 = build_resnet_block(1024, 256, 1024, downsample=False)
    block19 = build_resnet_block(1024, 256, 1024, downsample=False)
    block20 = build_resnet_block(1024, 256, 1024, downsample=False)
    block21 = build_resnet_block(1024, 256, 1024, downsample=False)
    block22 = build_resnet_block(1024, 256, 1024, downsample=False)
    block23 = build_resnet_block(1024, 256, 1024, downsample=False)
    block24 = build_resnet_block(1024, 256, 1024, downsample=False)
    block25 = build_resnet_block(1024, 256, 1024, downsample=False)
    block26 = build_resnet_block(1024, 256, 1024, downsample=False)
    block27 = build_resnet_block(1024, 256, 1024, downsample=False)
    block28 = build_resnet_block(1024, 256, 1024, downsample=False)
    block29 = build_resnet_block(1024, 256, 1024, downsample=False)

    # Stage3 (3 layers)
    block30 = build_resnet_block(1024, 512, 2048, downsample=True, strides=(2, 2))
    block31 = build_resnet_block(2048, 1024, 2048, downsample=False)
    block32 = build_resnet_block(2048, 1024, 2048, downsample=False)

    w1_np = initialize_weight([64, 3, 7, 7], dtype=np.float32)
    b1_np = initialize_weight([64], dtype=np.float32)

    @script(default_opset=op)
    def Resnet101(x: FLOAT) -> FLOAT:
        """Run the resnet101 bevformer backbone.

        Args:
            x (FLOAT): [1, 3, H, 6*W]

        Returns:
            FLOAT: [1, C, Hout, Wout]
        """
        w1 = op.Constant(value=from_array(w1_np))
        b1 = op.Constant(value=from_array(b1_np))
        x = vidConv(x, w1, b1, pads=[3, 3, 3, 3], strides=[2, 2], kernel_shape=(7, 7))
        x = op.Relu(x)
        x = vidMaxPool(x, kernel_shape=(3, 3), pads=(0, 0, 1, 1), strides=(2, 2))

        # Stage 0
        x = block0(x)
        x = block1(x)
        x = block2(x)

        # Stage 1
        x = block3(x)
        x = block4(x)
        x = block5(x)
        x = block6(x)

        # Stage 2
        x = block7(x)
        x = block8(x)
        x = block9(x)
        x = block10(x)
        x = block11(x)
        x = block12(x)
        x = block13(x)
        x = block14(x)
        x = block15(x)
        x = block16(x)
        x = block17(x)
        x = block18(x)
        x = block19(x)
        x = block20(x)
        x = block21(x)
        x = block22(x)
        x = block23(x)
        x = block24(x)
        x = block25(x)
        x = block26(x)
        x = block27(x)
        x = block28(x)
        x = block29(x)

        # Stage 3
        x = block30(x)
        x = block31(x)
        x = block32(x)

        return x

    return Resnet101


def build_img_neck(config):
    """Build the FPN-style image neck on top of the backbone."""
    w1_np = initialize_weight([256, 2048, 1, 1], dtype=np.float32)
    b1_np = initialize_weight([256], dtype=np.float32)

    w2_np = initialize_weight([256, 256, 3, 3], dtype=np.float32)
    b2_np = initialize_weight([256], dtype=np.float32)

    hidden_size = config._dim_

    @script(default_opset=op)
    def FPN(x: FLOAT) -> FLOAT:
        w1 = op.Constant(value=from_array(w1_np))
        b1 = op.Constant(value=from_array(b1_np))
        x = vidConv(x, w1, b1)

        w2 = op.Constant(value=from_array(w2_np))
        b2 = op.Constant(value=from_array(b2_np))

        # FIXME THIS PADDING IS HERE TO RECREATE ORIGINAL DATA SIZES
        x = vidConv(x, w2, b2, kernel_shape=(3, 3), pads=(1, 1, 1, 1))

        x = op.Reshape(x, op.Constant(value=[1, hidden_size, -1, 6]))
        return x

    return FPN
