@0xbf5147a2c3bd1f5d;
const uint32MaxValue :UInt32 = 4294967295;

struct Network {
    name @0: Text;
    layers @1 :List(Layer);
    tensors @2: List(Tensor);

    inputs @3: List(UInt32); # Indices of tensors, which are network inputs (Network.tensors[index])
    outputs @4: List(UInt32); # Indices of tensors, which are network outputs (Network.tensors[index])

    metaInfo @5: MetaInfo;  # Struct containing meta information about the network

    sysConfiguration @6: SysConfiguration;  # System configuration used for mapping

    inferenceProfile @9: InferenceProfile;  # Inference profiling results

    # Debugging fields
    # These fields are filled with data when running vid flow with --export_debug_data
    debugFloatingPointTensorData @7 :List(Data);  # Debugging field containing unquantized full precision tensor data
    debugQuantizedFloatingPointTensorData @8 :List(Data);  # Debugging field containing quantized floating tensor data
}

# System configuration used for mapping
# This includes the number of clusters, their sizes, and groups of fused layers that share the same tile(es)
struct SysConfiguration {
    nClusters @0 :UInt32;  # number of clusters including memory clusters and processing clusters, (currently memory cluster at index 0)
    mCluster @1 :UInt32;   # number of memory clusters

    clusters @2 :List(Cluster);  # List of clusters

    # groups of fused layers that share the same tile(es)
    finalFusedLayerIdx @3 :List(UInt32); # Layer.idx of the final fused layer of each fused group of layers

    frequency @4 :UInt64 = 0; # Core frequency in Hz
}

# Inference profiling results
struct InferenceProfile {
    clusterProfiles @0 :List(ClusterProfile);  # List of clusterProfiles
}

# Profiling per cluster
# Including a summary of all cycles per inference TBC
struct ClusterProfile {
    idx @0 :UInt32; # # Unique index for this cluster corresponding to its position in Network.clusters
    cycleSummary @1 :CycleSummary; # cycles per inference
}

# Summary of cycles per inference
struct CycleSummary {
    mac @0 :UInt64; # MAC cycles
    nonMac @1 :UInt64; # non MAC cycles
    exposedDma @2 :UInt64; # exposed DMA cycles (wait)
    total @3 :UInt64; # total cycles per inference
}

struct MetaInfo {
    vidortCommit @0: Text;  # Commit ID of the vidort repository at the time this message was created
    vidaimapCommit @1: Text;   # Commit ID of the vidnnmap repository at the time this message was created

    vidortConfig @2: Text;  # vidort configuration parameters used to create this model (formatted as YAML)
    vidaimapArgs @3: Text;  # vidnnmap CLI parameters used to create this model
}

struct Layer {
    idx @0 :UInt32;        # Unique index for the layer corresponding to its position in Network.layers
    name @1 :Text;        # Name for the layer
    layerType @2 :LayerType;  # Type of this layer  # TODO: This may be obsolete, since it is encoded in attributes

    # Union field so that attributes specific to a new layer can be set during runtime
    # Only one of these will be "active" at a time!
    attributes :union {
        conv @3 :ConvAttributes;
        maxpool @4 :MaxPoolAttributes;
        shortcut @5 :ShortcutAttributes;
        averagepool @6 :AveragePoolAttributes;
        resize @7 :ResizeAttributes;
        layernorm @8 :LayerNormAttributes;
        softmax @9: SoftmaxAttributes;
        split @10: SplitAttributes;
        squeeze @11: SqueezeAttributes;
        reshape @12: ReshapeAttributes;
        transpose @13: TransposeAttributes;
        gather @14: GatherAttributes;
        slice @15: SliceAttributes;
        flatten @16: FlattenAttributes;
        concat @17: ConcatAttributes;
        rmsnormalization @19: RMSNormalizationAttributes;
        rope @20: RopeAttributes;
        expand @21: ExpandAttributes;

        retrtransformation @22: RETRTransformationAttributes;
        rtrtransformation @23: RTRTransformationAttributes;
        rertransformation @24: RERTransformationAttributes;
        scatter @25: ScatterAttributes;
        convtranspose @26: ConvTransposeAttributes;
        gridsample @27: GridSampleAttributes;
        attention @28: AttentionAttributes;
    }

    mapping @18: Mapping;  # Data required for hardware processing
}


# A cluster is a group of processing cores (v-MPs) with shared memory
# A cluster may be a memory cluster (OCR) or a processing cluster (v-MP)
# A cluster may contain multiple v-MPs, which are the actual processing cores
struct Cluster {
    idx @0 :UInt32;  # Unique index for this cluster corresponding to its position in Network.clusters
    memoryCluster @1 :Bool;  # Type of this cluster (processing or memory)
    nMPs @2 :UInt32;  # number of processing cores in this cluster
    sizeDMEM @3 :UInt32; # number of DMEM blocks in each v-MP of this cluster
    sizeDMEM2 @4 :UInt32; # number of DMEM2 blocks in each v-MP of this cluster
    sizeDMEM3 @5 :UInt32; # number of DMEM3 blocks in each v-MP of this cluster
    sizeOCR @6 :UInt64; # size of OCR memory in this cluster
    sizeDDR @7 :UInt64; # size of DDR memory in this cluster
}

# A tile is a spacial part of a layer that needs to be assigned to a cluster for the actual processing
# A layer is partitioned into nTile tiles. Each tile is assigned to a processing clusters (pCluster) for actual processing.
# A processing cluster always processes one tile at a time. If there are not enough tiles left for all processing clusters some clusters are idle
# Each tile is further partitioned into partH * partW * partN vPartition blocks
# vInp...vWgtEDMA: tile buffers reside in cluster memory and may be allocated or referenced
# groups0...maxMPs: Parameters to define the partitioning scheme in N-direction, each group comprises all used MPs (goal:used MPs == available MPs)

struct Tile {
    idx @0 :UInt32;  # Unique index for this tile corresponding to its position in Layer.tiles
    inpY @1 :Int32;  # y position of the tile w.r.t the Layer-Input
    inpX @2 :Int32;  # x position of the tile w.r.t the Layer-Input
    outY @3 :Int32;  # y position of the tile w.r.t the Layer-Output
    outX @4 :Int32;  # x position of the tile w.r.t the Layer-Output

    partN @5 :UInt32; # number of partitions in N-direction
    partH @6 :UInt32; # number of partitions in H-direction
    partW @7 :UInt32; # number of partitions in W-direction
    numN @8 :UInt32;  # number of output pixel in dimension N <= numN of the corresponding layer
    numH @9 :UInt32;  # number of output pixel in dimension H <= numH of the corresponding layer
    numW @10 :UInt32; # number of output pixel in dimension W <= numW of the corresponding layer
    # uint32_t partD;        ///< TODO

    partition @11 :List(Partition); # all partitions per tile [partN*partH*partW*partD]

    # processing tile buffers
    # buffers reside in cluster memory and may be allocated or referenced
    buffers @12 :Buffers; # processing buffers for the tile
}

# Parameters of a partition
# The layer input and output buffers are split into smaller work-units called partitions.
# An input partition is a contiguous subset of the layer input buffer,
# an output partition is a contiguous subset of the layer output buffer respectively.
struct Partition {
    mpIdx @0 :UInt32; # Index of associated v-MP core
    inpX @1 :Int32;   # rel. X-position of part. in input buffer in X respective W direction
    inpY @2 :Int32;   # rel. Y-position of part. in input buffer in Y respective H direction
    inpZ @3 :Int32;   # rel. Z-position of part. in input buffer in Z respective D direction
    outX @4 :UInt32;  # rel. X-position of part. in output buffer in X respective W direction
    outY @5 :UInt32;  # rel. Y-position of part. in output buffer in Y respective H direction
    outZ @6 :UInt32;  # rel. Z-position of part. in output buffer in Z respective N direction

    numW @7 :UInt32;  # output partition size W in x direction
    numH @8 :UInt32;  # output partition size H in y direction
    numD @9 :UInt32;  # input  partition size D in z direction
    numF @10 :UInt32; # number of groups of filters of size numL, i.e, numN = numF*numL filters are processed in parallel in N direction (output)
    numL @11 :UInt32; # number of lanes used, i.e., the minimum parallel processing level; currently always numL = 8

    # additional parameters for codegen
    numWPad @12 :UInt32;  # padded output partition size in W direction
    mpinpWPad @13 :UInt32; # padded input buffer payload width in v-MP DMEM2

    mpinpW @14 :UInt32; # local input buffer payload width in v-MP DMEM2
    mpinpH @15 :UInt32; # local input buffer payload width in v-MP DMEM2
    mpinpRowOfs @16 :UInt32; # word-offset between two consecutive input buffer rows in v-MP DMEM2
    mpresH @17 :UInt32;

    wgtsAddr @18 :UInt64;   # ext. address of weights
    wgtsLen @19 :UInt32;    # length (bytes) to load from wgtsAddr. If 0: do not load
    cdataAddr @20 :UInt64;  # ext. address of cdata
    cdataLen @21 :UInt32;   # length (bytes) to load from cdataAddr. If 0: do not load
    auxInpAddr @22 :UInt64; # ext. auxiliary input address
    auxOutAddr @23 :UInt64; # ext. auxiliary output address

    concatIdx @24 :UInt32;  # index of associated concat structure
}


struct Buffers {
    vInp @0 :Buffer; # fixed point buffer struct for the input tensor
    vOut @1 :Buffer; # fixed point buffer struct for the output tensor
    vWgt @2 :Buffer; # fixed point buffer struct for the weights tensor
    vBas @3 :Buffer; # fixed point buffer struct for the bias tensor
    vAux @4 :Buffer; # fixed point buffer struct in split D case and for a merged Layer (splitD & merged are mutually exclusive)
    vSft @5 :Buffer; # fixed point buffer struct for the shifts, i.e., the fixed-point scaling factor equivalent in case of power-of-two quantization

    vInpEDMA @6 :Buffer; # input preload buffer filled by EDMA, always in memory cluster OCR  or NULL
    vWgtEDMA @7 :Buffer; # weight preload buffer filled by EDMA, always in memory cluster OCR  or NULL
}

struct Buffer {
    mt @0 :MemType; # memory type (OCR, DDR, ...)
    n @1 :UInt32;
    h @2 :UInt32;
    w @3 :UInt32;
    c @4 :UInt32;
    size @5 :UInt64;       # # of bytes (word aligned)
    address @6 :UInt64;    # linear address in OCR / DDR mem (starting from 0)
    cIdx @7 :UInt32;       # cluster index
    maxExp @8 :List(Int8); # re-ordered exponents for the fixed-point number format
                           # maxExp may be moved to Layer
    fixedPointData @9 :Data; # array containing re-ordered fixed point mantissas, element size depends on buffer type (Mapping.inpWW, outWW, wgtWW, or 8 (vSft) or 32 (vAux))
}

enum MemType {
    nomem @0; # no memory
    ddr @1;   # global  DDR
    ocr @2;   # on-chip SRAM
}

# Data required for hardware processing
# and mapping to clusters of v-MP cores
struct Mapping {
    tiles @0: List(Tile);  # List of tiles, which are part of this layer
    buffers @1: Buffers;  # processing buffers for the layer, which are used to store input, output, weights and bias tensors
    mergedLayerIdx @2 :UInt32; # layer index, which is merged with this layer, UINT32_MAX if no layer is merged
    mergedWithPrev @3: Bool;  # Current layer is merged with some previous layer, e.g., SHORTCUT
    skipped @4: Bool;  # Layer is skipped, i.e., not executed

    inpWW @5 :UInt32;         # input-buffer word width (in bytes)
    outWW @6 :UInt32;         # output-buffer word width (in bytes)
    wgtWW @7 :UInt32;         # weights word width (in bytes)
    shortcutDelta @8 :Int32;  # shortcut delta value
    activation @9: UInt32;    # Which activation to apply on the output  FIXME shall use original Layer.activation
    activationAux @10 :UInt32;  # activation auxiliary value
    flags @11 :UInt32;  # flags for the layer
    lIdx @12 :UInt32; # index of this layer of the original network including merged and skipped layers
    nIdx @13 :UInt32; # index of this layer excluding merged or skipped layers, nIdx <= lIdx
}

struct Tensor {
    idx @0 :UInt32;  # Unique index for this tensor corresponding to its position in Network.tensors
    name @1 :Text;  # Name of this tensor
    tensorType @2 :TensorType;

    floatingPointData @3 :Data;  # fp-32 array of shape(@6) (ToDo: Remove together with a complete modelZoo recreation)

    # Raw data of type `dtype` and shape `shape`
    # For TensorType static this contains the weight data of the corresponding tensor. Otherwise this is not set.
    data @12 :Data;
    dtype @13 :TensorDataType;

    fixedPointData @4 :Data;  # int-16 array containing fixed point mantissas of shape(@6), only nBits(@7) are used
    maxExponents @5 :Data;  # 1D int-8 array of size shape[quantAxis] containing the maximum exponents of each channel

    shape @6 :List(UInt32);  # Array shape of floatData and mantissas
    nBits @7 :UInt32;  # Number of bits used to quantize floatData into mantisassas and maxExponents
    quantAxis @8 :UInt32;  # Indicates axis over which all channels are quantized individually

    producer @9: UInt32;  # Index of the layer, which produces this tensor (Network.layer[producer])
    consumers @10: List(UInt32);  # Indices of layers, which consume this tensor


    adjustedMaxExponents @11 :Data;  # Debugging field. This contains the vidort adjusted max exponents
}

enum TensorType {
    dynamic @0;  # Tensors being produced by layers as intermediate outputs
    static @1;  # Fixed tensors like weights and biases, which do do not change during runtime
    graphInput @2;  # Special kind of dynamic tensor, which represents network inputs
    graphOutput @3;  # Special kind of dynamic tensor, which represents network outputs
    none @4; # Uninitialized state
}

# Supported data types for Tensor objects
enum TensorDataType {
    float32 @0;
    int64 @1;
}



struct ConvAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__Conv.html
    # Note that this OP refers to the 2D Conv case (ONNX supports Nd convolution)

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, Cin, Hin, Win]
    weight @1: UInt32;  # Weight tensor of shape [Cout, Cin, kernel_height, kernel_width]
    bias @2: UInt32 = .uint32MaxValue;  # Optional: Bias 1D vector of shape [Cout].
    output @3: UInt32; # Output tensor after activation and data transformation
    preActivationOutput @14: UInt32 = .uint32MaxValue; # Direct conv output tensor BEFORE activation of shape [B, Cout, Hout, Wout]

    # Attributes
    enum ReshapeMode {
        none @0;
        mulExpand @1;
        transformerQK @2;
        transformerV @3;
        flattenW @4;
    }
    kernelShape @6 :List(UInt32);  # Height and width of the kernel
    activation @10: ActivationType;  # Which activation to apply on the output

    # Optional
    dilations @4: List(UInt32) = [1, 1];  # The size of the kernel along each axis.
    group @5: UInt32 = 1;  # Number of groups for the convolution.
    pads @7: List(UInt32) = [0, 0, 0, 0];  # Padding in the form (pad_top, pad_left, pad_bottom, pad_right).
    strides @8 :List(UInt32) = [1, 1];      # Strides for the convolution.
    dim @9: UInt32 = 4;  #  Dimensionality of the convolution (e.g., 2D or 4D).
    reshapeMode @11: ReshapeMode = none;  #  Reshape mode, such as "MUL_EXPAND" or "TRANSFORMER_QK".
    reshapeModeGroups @12: List(UInt32) = [1];  #  Group sizes for reshape in "TRANSFORMER_QK" mode.
    hardSigmoidAlpha @13: Float32 = 0.2;  # Alpha value for the `hardsigmoid` activation.
}



struct MaxPoolAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__MaxPool.html
    # Note that this OP refers to the 2D max pool case (ONNX supports Nd max pool)

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, Cin, Hin, Win]
    output @1: UInt32; # Output tensor of shape [B, Cout, Hout, Wout]

    # Attributes
    enum AutoPadMode {
        notSet @0;
        sameUpper @1;
        sameLower @2;
        valid @3;
    }
    enum CeilMode{
        floor @0;  # corresponds to ceil_mode=0 in ONNX
        ceil @1; # corresponds to ceil_mode=1 in ONNX
    }
    kernelShape @2 :List(UInt32) ;  # The size of the kernel along each axis.

    # Optional
    autoPad @3: AutoPadMode = notSet;  # Padding Mode
    ceilMode @4: CeilMode = floor;  #  Whether to use ceil or floor to compute the output shape.
    dilations @5: List(UInt32) = [1, 1];  # Dilation value along each spatial axis of the filter.
    pads @6: List(UInt32) = [0, 0, 0, 0];  # Padding in the form (pad_top, pad_left, pad_bottom, pad_right).
    strides @7: List(UInt32) = [1, 1];  # Stride along each spatial axis.
}

struct ShortcutAttributes {
    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input0 @0: UInt32;  # Left operand tensor of shape [B, Cin, Hin, Win]
    input1 @1: UInt32;  # Right operand tensor. Must be broadcastable with input0

    output @2: UInt32; # Output tensor of shape [B, Cout, Hout, Wout] AFTER activation is applied
    preActivationOutput @7: UInt32 = .uint32MaxValue; # Direct shortcut output tensor BEFORE activation of shape [B, Cout, Hout, Wout]


    # Attributes
    enum Mode {
        addition @0;
        multiplication @1;
        division @2;
    }
    enum ReshapeMode {
        none @0;
        transformerQK @1;
    }

    mode @3: Mode;  # possible options are 'addition' and 'multiplication'

    # Optional
    activation @4: ActivationType = linear;  # Which activation to apply on the output
    reshapeMode @5: ReshapeMode = none;  # Reshape mode, such as "TRANSFORMER_QK" for transformer operations.
    reshapeModeGroups @6: List(UInt32) = [1]; # Group size for reshaping in "TRANSFORMER_QK" mode.

}

struct AveragePoolAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__AveragePool.html
    # Note that this OP refers to the 2D average pool case (ONNX supports Nd average pool)


    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, Cin, Hin, Win]
    output @1: UInt32; # Output tensor of shape [B, Cin, Hout, Wout]

    # Attributes
    enum AutoPadMode {
        notSet @0;
        sameUpper @1;
        sameLower @2;
        valid @3;
    }
    enum CeilMode{
        floor @0;  # corresponds to ceil_mode=0 in ONNX
        ceil @1; # corresponds to ceil_mode=1 in ONNX
    }
    kernelShape @2: List(UInt32);  # The size of the kernel along each axis.

    # Optional
    autoPad @3: AutoPadMode = notSet; # Padding Mode
    ceilMode @4: CeilMode  = floor;  #  Whether to use ceil or floor to compute the output shape.
    countIncludePad @5: Bool = false;  #  Whether to include pad pixels when calculating values for the edges.
    dilations @6: List(UInt32) = [1, 1];  # # Dilation value along each spatial axis of the filter.
    pads @7: List(UInt32) = [0, 0, 0, 0];  # Padding in the form (pad_top, pad_left, pad_bottom, pad_right).
    strides @8: List(UInt32) = [1, 1];  # Stride along each spatial axis.
    outputDim @9: UInt32 = 4;  #  adds additional unsqueezes internally if dim !=4.
}

struct FlattenAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__Flatten.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [D1, D2, ... Dn]
    output @1: UInt32;  # Flattened input tensor. E.g. axis=1 would be [D1, D2 * D3 * ... * Dn]

    # Attributes
    # Optional:
    axis @2: UInt32 = 1;  # The axis along which to flatten the tensor.
}

struct ConcatAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__Concat.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    inputs @0: List(UInt32);  # List of input tensors to be concatenated. Must share the same shape except at axis.
    output @1: UInt32;  # Output tensor with same shape as inputs, except at axis.

    # Attributes
    axis @2: UInt32; # The Concat Axis
}

struct ResizeAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__Resize.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, C, H, W]
    output @1: UInt32;  # Output tensor of shape [B, C, Hnew, Wnew]

    enum InterpolationMode {
        nearest @0;
        linear @1;
    }

    enum CoordinateTransformationMode {
        asymmetric @0;
        halfPixel @1;
        alignCorners @2;
    }

    # Attributes
    sizes @2: List(UInt32);  # List with [Hnew, Wnew]
    mode @3: InterpolationMode;
    coordinateTransformationMode @4: CoordinateTransformationMode;
}


struct LayerNormAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, C, H, W]
    scale @1: UInt32;  # Scale tensor of shape [C]
    bias @2: UInt32 = .uint32MaxValue;  # Optional: Bias tensor of shape [C]
    output @3: UInt32;  # Output tensor of shape [B, C, H, W]

    # No attributes
}

struct SplitAttributes {
    # Reference https://onnx.ai/onnx/operators/onnx__Split.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [D1, D2, ..., Dn]
    output @1: UInt32;  # For axis=1 it would be [D1, split, ..., Dn]

    # Attributes
    split @2: List(UInt32);  # Length of each segment after the split.
    pos @3:  List(UInt32); # The index of the segment to retrieve from the sequence.
    axis @4: UInt32;  # The axis along which to split the input tensor.
}

struct SoftmaxAttributes {
    # Reference https://onnx.ai/onnx/operators/onnx__Softmax.html  with axis=2

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, C, H, W]
    output @1: UInt32;  # Output tensor of shape [B, C, H, W]]

    # Attributes
    reshapeModeGroups @2: List(UInt32) = [1];  # Group dimensions for reshaping in transformer operations
}

struct SqueezeAttributes {
    # Reference https://onnx.ai/onnx/operators/onnx__Squeeze.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape, where dim=1 at denoted axes
    output @1: UInt32;  # Squeezed output tensor. [1, D1, D2, 1] with axes=[0, 3] becomes [D1, D2]

    # Attributes
    axes @2: List(UInt32);  # A list of dimension indices with size 1 to be squeezed.
}

struct ReshapeAttributes {
    # Reference https://onnx.ai/onnx/operators/onnx__Reshape.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape but same number of entries as `shape`
    output @1: UInt32;  # Reshaped input tensor with shape `shape`

    # Attributes
    shape @2: List(UInt32);  # Shape of the new tensor.
}

struct TransposeAttributes {
    # Reference https://onnx.ai/onnx/operators/onnx__Transpose.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape
    output @1: UInt32;  # Output tensor with shape according to input shape and perm.

    # Attributes
    perm @2: List(UInt32);  # Permute the axes according to the values given. Its length must be equal to the rank of the input.
}

struct GatherAttributes {
    # Reference https://onnx.ai/onnx/operators/onnx__Gather.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [D1, D2, ..., Dn]

    deprecatedDoNotUse @1: List(UInt32); # This is no longer used and should removed #DEPRECATED

    indexTensor @4: UInt32; # Index tensor containing indices to gather from axis.

    output @2: UInt32;  # Output tensor. Example: axis = 1 and indices [N]: [D1, N, ..., Dn]

    # Attributes
    axis @3: UInt32;  # Axis on which to gather with indices
}
struct SliceAttributes {
    # Reference https://onnx.ai/onnx/operators/onnx__Slice.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape
    output @1: UInt32;  # Slice of the input tensor according to the provided attributes

    # Attributes
    starts @2: List(UInt32);  # 1-D tensor of starting indices of corresponding axis in axes
    ends @3: List(UInt32);  # 1-D tensor of ending indices (exclusive) of corresponding axis in axes
    axes @4: List(UInt32);  # 1-D tensor of axes that starts and ends apply to.
    steps @5: List(UInt32);  # 1-D tensor of slice step of corresponding axis in axes.
}

struct RMSNormalizationAttributes {
    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    # TODO: Fix annotations
    input @0: UInt32;  # Input tensor of shape [B, C, H, W]
    scale @1: UInt32;  # Scale tensor of shape [C]
    output @2: UInt32;  # Output tensor of shape [B, C, H, W]

    # Attributes
    axis @3: UInt32;  # Axis of which to normalize. Defaults to 0
    epsilon @4: Float32;  # Epsilon to apply during divison, to avoid div by 0.

}

struct RopeAttributes {
    # TODO: fix annotations
    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, C, H, W]
    cos @1: UInt32;
    sin @2: UInt32;
    output @3: UInt32;  # Output tensor of shape [B, C, H, W]
}

struct ExpandAttributes {
    # TODO: Amend annotations
    # Reference https://onnx.ai/onnx/operators/onnx__Expand.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape but same number of entries as `shape`
    output @1: UInt32;  # Reshaped input tensor with shape `shape`

    # Attributes
    shape @2: List(UInt32);  # Shape of the new tensor.
}

struct RETRTransformationAttributes {
    # Container for consecutive Reshape, Expand, Transpose, Reshape transformation

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape but same number of entries as `shape`
    output @1: UInt32;  # Reshaped input tensor with shape `shape`

    # Attributes
    reshape1Shape @2: List(UInt32);
    expandShape @3: List(UInt32);
    transposePerm @4: List(UInt32);
    reshape2Shape @5: List(UInt32);
}

struct RERTransformationAttributes {
    # Container for consecutive Reshape, Expand, Reshape transformation

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape but same number of entries as `shape`
    output @1: UInt32;  # Reshaped input tensor with shape `shape`

    # Attributes
    reshape1Shape @2: List(UInt32);
    expandShape @3: List(UInt32);
    reshape2Shape @4: List(UInt32);
}

struct RTRTransformationAttributes {
    # Container for consecutive Reshape, Transpose, Reshape transformation

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of any shape but same number of entries as `shape`
    output @1: UInt32;  # Reshaped input tensor with shape `shape`

    # Attributes
    reshape1Shape @2: List(UInt32);
    transposePerm @3: List(UInt32);
    reshape2Shape @4: List(UInt32);
}

struct ScatterAttributes {
    # Performs a specialized scatter operation. For now this is only used to update the KV cache tensors
    # in transformer models. Given the data tensor, containg the cached key/value vectors, the update tensor,
    # containing the new key/value vectors to be inserted, and the index tensor, containing the index at which
    # to insert the new vectors.

    data @0: UInt32;  # Input tensor of shape [1, dim, 1, sequence_length]
    update @1: UInt32;  # The update vectors of shape [1, dim, 1, 1]
    index @2: UInt32;  # The index tensor containing the position to update of shape [1]

    output @3: UInt32;  # The updated data tensor of shape [1, dim, 1, sequence_length]
}

struct AttentionAttributes {
    # Multi-head scaled dot-product attention (MHA / GQA / MQA).
    # Mirrors vidAttention in the v-NN ORT optimizer function library.
    # Reference: https://onnx.ai/onnx/operators/onnx__Attention.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    query @0: UInt32;   # Query tensor [B, q_heads*head_dim, 1, seq_q]
    key @1: UInt32;     # Key tensor [B, kv_heads*head_dim, 1, seq_k]
    value @2: UInt32;   # Value tensor [B, kv_heads*head_dim, 1, seq_k]
    output @3: UInt32;  # Attention output tensor [B, q_heads*head_dim, 1, seq_q]
    qkOutput @4: UInt32 = .uint32MaxValue;  # Optional raw QK^T MatMul output [B, q_heads, seq_q, seq_k]
    attentionScores @5: UInt32 = .uint32MaxValue;  # Optional softmax attention weights [B, q_heads, seq_q, seq_k]

    # Attributes
    qNumHeads @6: UInt32;    # Number of query heads
    kvNumHeads @7: UInt32;   # Number of key/value heads (< qNumHeads for GQA/MQA, == qNumHeads for MHA)
    scale @8: Float32;       # Scaling factor applied to QK logits
    isCausal @9: UInt32 = 0; # When 1, the backend applies a causal constraint using qStartOffset

    # Sequence-position inputs — allow masking without a 2-D mask tensor.
    # All values are tensor IDs; .uint32MaxValue means the input is absent.
    qStartOffset @10: UInt32 = .uint32MaxValue;  # [B] absolute index of first query token in KV seq
    qSeqLens @11: UInt32 = .uint32MaxValue;      # [B] number of valid query tokens per batch element
    kvStartOffset @12: UInt32 = .uint32MaxValue; # [B] index of first valid key/value token (left padding)
    kvSeqLens @13: UInt32 = .uint32MaxValue;     # [B] number of valid key/value tokens per batch element
}

struct ConvTransposeAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    # Note that this OP refers to the 2D Conv case (ONNX supports Nd convolution)

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, Cin, Hin, Win]
    weight @1: UInt32;  # Weight tensor of shape [Cout, Cin, kernel_height, kernel_width]
    bias @2: UInt32 = .uint32MaxValue;  # Optional: Bias 1D vector of shape [Cout].
    output @3: UInt32; # Output tensor after activation and data transformation
    preActivationOutput @4: UInt32 = .uint32MaxValue; # Direct conv output tensor BEFORE activation of shape [B, Cout, Hout, Wout]

    kernelShape @5 :List(UInt32);  # Height and width of the kernel
    activation @6: ActivationType;  # Which activation to apply on the output

    # Optional
    dilations @7: List(UInt32) = [1, 1];  # The size of the kernel along each axis.
    group @8: UInt32 = 1;  # Number of groups for the convolution.
    pads @9: List(UInt32) = [0, 0, 0, 0];  # Padding in the form (pad_top, pad_left, pad_bottom, pad_right).
    strides @10 :List(UInt32) = [1, 1];      # Strides for the convolution.
}

struct GridSampleAttributes {
    # Reference: https://onnx.ai/onnx/operators/onnx__GridSample.html

    # IDs of input/output tensors (index refers to position of tensor in Network.tensors list)
    input @0: UInt32;  # Input tensor of shape [B, C, Hin, Win]
    grid @1: UInt32;   # Grid tensor of shape [B, 2, Hout, Wout] containing the sampling coordinates
    output @2: UInt32; # Output tensor of shape [B, C, Hout, Wout]

    enum InterpolationMode {
        linear @0;
        nearest @1;
    }

    # Attributes
    alignCorners @3: Int32;  # 1 if the extrema (-1 and 1) are considered as referring to the center of the corner pixels, 0 otherwise.
    mode @4: InterpolationMode;  # Interpolation mode to use for sampling
}


# Keep track of all layers we support. This may be obsolete with the attributes union in the Layer struct
enum LayerType {
    conv @0;
    maxPool @1;
    shortcut @2;
    averagePool @3;
    resize @4;
    layerNorm @5;
    softmax @6;
    split @7;
    squeeze @8;
    reshape @9;
    transpose @10;
    gather @11;
    slice @12;
    flatten @13;
    concat @14;
    rmsNormalization @15;
    rope @16;
    expand @17;
    retr @18;
    rer @19;
    rtr @20;
    scatter @21;
    convTranspose @22;
    gridSample @23;
    attention @24;
}


# Enum for activation functions
enum ActivationType {
    linear @0;
    relu @1;
    clip @2;
    hardswish @3;
    relu6 @4;
    swish @5;
    sigmoid @6;
    hardsigmoid @7;
    gelu @8;
    mish @9;
    leakyrelu @10;
}
