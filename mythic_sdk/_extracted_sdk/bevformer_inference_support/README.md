# BEVFormer

- BEVFormer paper: [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270)
- Original BEVFormer repo: https://github.com/fundamentalvision/BEVFormer/
- Onnx Export support code: https://github.com/AXERA-TECH/bevformer.axera

## Variants

**Integrated**
- BEVFormer Tiny 
  - 800x450 resolution
  - 1600x900 resolution

**Not Yet Integrated**
- BEVFormer Base (1600x900 resolution)


## Folder Structure

Primary directories/files 

```bash
├─ conversion_steps.py # MUNC-based conversion steps
├─ bevformer_lib/ # primarily third-party code to implement/define BEVFormer
│  ├─ projects/
│  │  ├─ configs/ # MMCV-based BEVFormer configs (defines network structure)
│  │  └─ mmdet3d_plugin/ 
│  │     └─ bevformer/ # primary BEVFormer module implementations 
│  └─ custom_utils/ # support code added for inference and visualization
└─ bevformer_inference.py # generalized BEVFormer inference script (works with .pth, .onnx, and TorchNet)
```


## Main Deltas from Original Implementation

- Dependency modernization
  - Python: `3.8` -> `>=3.10`
  - `mmcv-full`: `1.4.0` -> [modified 1.7.2](https://github.com/Mythic-Public/mmcv/tree/v1.7.2-bevformer)
  - `mmdet3d`: `0.17.1` -> [modified 1.0.0rc6](https://github.com/Mythic-Public/mmdetection3d/tree/v1.0.0rc6-bevformer)
  - `torch`: `1.9.1` -> `>=2`
  - `numpy`: `1.19.5` -> `>=2`
  - CUDA: 11 -> 12
  - other minor dependencies
- ONNX Export support
  - Most of the logic comes from https://github.com/AXERA-TECH/bevformer.axera, with some modifications (some of which are informed by https://github.com/DerryHub/BEVFormer_tensorrt)


## Conversion Overview

**Steps**
- Download and set up dataset (*not managed via model zoo*)
- `generate_nuscenes_annotations` (*only needs to be done once*)
- `train_torch_fp` (regular fp32-based BEVFormer training)
- `to_onnx` (export BEVFormer PyTorch model to ONNX)
- `to_structural` (convert fp32 ONNX model to intermediate "structural" graph)
- `to_training` (convert structural graph to quantized analog-aware graph suitable for retraining)
- `train` (analog-aware retraining) 
- `to_acm` (converts the retrained graph to an evaluation “acm” ONNX graph)
- `create_artifact` (creates an artifact for Mythic's compiler)

*Note: each step takes the subsequent step's output as input*

----


## Setup

### Set up CUDA Dependencies

*Note: some of BEVFormer's dependencies require building from source, hence the extra CUDA libraries* 

```bash
apt-get install -y cuda-nvcc-12-9 cuda-libraries-dev-12-9 
export CUDA_HOME=/usr/local/cuda-12.9 # or other install location as applicable
```

### Dependencies Setup

```bash
uv venv --python=3.12 venv
export UV_PROJECT_ENVIRONMENT=venv
uv sync --extra 'bevformer,acm-m2000,infer,wandb'
# 'wandb' extra can be omitted if not using Weights and Biases
```

### Model Selection

Activate virtual environment and configure environment variables for model variant
```bash
source ./scripts/bevformer/bevformer-tiny-1600x900.env
# - or -
source ./scripts/bevformer/bevformer-tiny-800x450.env
```

**Configuration**

See the following for the original MMCV-based configs for the models
- `mythic/model_zoo/bevformer/bevformer_lib/projects/configs/bevformer/bevformer_tiny.py`
- `mythic/model_zoo/bevformer/bevformer_lib/projects/configs/bevformer/bevformer_tiny-1600x900.py`

See files in `configs/bevformer` for Mythic-specific configuration of BEVFormer conversion and retraining


### Dataset Setup

Follow setup instructions from https://www.nuscenes.org/nuscenes / https://github.com/nutonomy/nuscenes-devkit to download and set up the dataset folder structures, including the can bus extension.

**Generate Dataset Annotation Files**

Configure dataset and annotations paths in the BEVFormer config

```yaml
# configs/bevformer/bevformer_tiny.yaml
nuscenes_root: /path/to/downloaded/nuscenes
nuscenes_data_annotation_root: /path/to/generate/nuscenes/annotations/to
pytorch_checkpoint_root: /path/to/pytorch/fp32/pth/files
```

Where:
- `nuscenes_root`: The path to dataset downloaded above
  - Overrides the `data_root` path in `mythic/model_zoo/bevformer/bevformer_lib/projects/configs/bevformer` configs
- `nuscenes_data_annotation_root`: Location to generate .pkl files that serve as preprocessed nuScenes indexes/caches for BEVFormer. They flatten the nuScenes JSON tables plus CAN bus extension into the exact structure BEVFormer expects
  - Overrides the `data_annotation_root` path in `mythic/model_zoo/bevformer/bevformer_lib/projects/configs/bevformer` configs
  - *Note: if you have other nuScenes-based projects, the annotations generated are not necessarily transferable between projects. As such, you'll likely want to set the `nuscenes_data_annotation_root` to a BEVFormer-specific directory*
- `pytorch_checkpoint_root`: Any downloaded PyTorch fp32 checkpoints (which can be used at the starting weights when exporting to ONNX)

Run generation script:
```bash
python3 scripts/common/convert_model.py steps=generate_nuscenes_annotations
```
*Note: This can take a while (1 hr+)*

## Training FP32

If you aren't starting conversion from a pretrained `.pth` checkpoint, you'll likely want to train in fp32 first.

*Note: although you can skip the initial fp32 training if you want to use initial weights (for BEVFormer Tiny, the ResNet backbone is initialized from the pretrained `torchvision://resnet50`, and the transformer head is untrained).*

```bash
# 1 GPU
python3 scripts/common/convert_model.py steps=train_torch_fp
# - or -
# Multiple GPUs (change the --nproc-per-node value accordingly)
torchrun --standalone --nproc-per-node=4 scripts/common/convert_model.py steps=train_torch_fp
```


## Export FP32 PyTorch Model to ONNX

Exports the PyTorch model to an ONNX model. 

```bash
python3 scripts/common/convert_model.py steps=to_onnx
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::to_onnx`

**Output**: `data/bevformer/fp32-<resolution>.onnx`


## Structural Modifications

- On chip vs off chip
  - The ResNet image backbone is placed on chip
  - The transformer head is kept digital
- Light constant folding is performed to simplify the model graph without folding subgraphs that should remain present and trainable

```bash
python3 scripts/common/convert_model.py steps=to_structural
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::to_structural`

**Output**: `data/bevformer/structural-<resolution>.onnx`


## Conversion To Mythic

Converts the structural graph to a quantized analog-aware graph suitable for retraining.

```bash
python3 scripts/common/convert_model.py steps=to_training
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::to_training`

**Output**: `data/bevformer/mythic-<resolution>.onnx`


## Mythic Retraining

Analog-aware retraining.

```bash
python3 scripts/common/convert_model.py steps=train
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::train`

**800x450 Training Configuration**: `configs/bevformer/training_config/tiny_800x450.yaml`

**1600x900 Training Configuration**: `configs/bevformer/training_config/tiny_1600x900.yaml`

**Outputs**: 
- `data/bevformer/trained-<resolution>.onnx` (trained onnx model at the end of training)
- `data/bevformer/bevformer_training/<resolution>/<training-run-name>/epoch_*.pth` (per-epoch snapshots)


## Bake Snapshot Weights to ONNX Model (optional)

While the above Mythic retraining will save the trained weights back to `data/bevformer/trained-<resolution>.onnx` at the end of training,
you can use this step to save mid-training checkpoints back to an onnx model.

```bash
python3 scripts/common/convert_model.py steps=bake_snapshot_weights_to_onnx
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::bake_snapshot_weights_to_onnx`

**Output**: `data/bevformer/trained-<resolution>-<custom-label>.onnx`


## Post Processing

Removes training-oriented graph nodes/structures (e.g. dropout). Optionally performs constant folding to further simplify the model and optimize for TorchNet inference/eval speed.

```bash
python3 scripts/common/convert_model.py steps=post_retraining_simplification
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::post_retraining_simplification`

**Output**: `data/bevformer/post-training-processed-<resolution>.onnx`


## Eval Analog-Aware ONNX Model

This runs the eval on an analog-aware Mythic model (by default the `post-training-processed-${model_setup.resolution}.onnx` model, will also work with other BEVFormer ONNX models). Use this to collect mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE (as well as the per-class level metrics).

```bash
python3 scripts/common/convert_model.py steps=eval_mythic_model
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::eval_mythic_model`

**Output**: Eval metrics


## Compiler Artifacts

Creates an artifact for Mythic's compiler, going through an ACM intermediate step

```bash
python3 scripts/common/convert_model.py steps=to_acm
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::to_acm`

**Output**: `data/bevformer/acm-<resolution>.onnx`


```bash
python3 scripts/common/convert_model.py steps=create_artifact
```

**Step Configuration**: `configs/bevformer/bevformer_tiny.yaml::create_artifact`

**Output**: `data/bevformer/compiler_ready_artifact-800x450.tar.gz`

----


## Generating Inference Videos (fp32, onnx, analog-aware)

For full options see `mythic/model_zoo/bevformer/bevformer_inference.py --help`

Examples below:

**FP32**
```bash
./bevformer_inference.py pytorch \
  '/path/to/checkpoint/epoch_<#>.pth' \
  bevformer_lib/projects/configs/bevformer/bevformer_tiny-1600x900.py \
  --end-scene=5 --map-bev --data-type=sweeps
```

Supports:
- `.pth` checkpoints

**ONNX**
```bash
./bevformer_inference.py onnx \
  '/path/to/mythic-<resolution>.onnx' \
  bevformer_lib/projects/configs/bevformer/bevformer_tiny.py \
  --end-scene=5 --map-bev --data-type=sweeps
```

Supports:
- `fp32-<resolution>.onnx`
- `structural-<resolution>.onnx`

**Analog Aware**

```bash
# From checkpoint
./bevformer_inference.py torchnet \
  '/path/to/mythic-<resolution>.onnx' \
  bevformer_lib/projects/configs/bevformer/bevformer_tiny-1600x900.py \
  --checkpoint '/path/to/checkpoint/epoch_<number>.pth' \
  --end-scene=5 --map-bev --data-type=sweeps

# Trained
./bevformer_inference.py torchnet \
  '/path/to/trained-<resolution>.onnx' \
  bevformer_lib/projects/configs/bevformer/bevformer_tiny-1600x900.py \
  --end-scene=5 --map-bev --data-type=sweeps
```

Supports:
- `mythic-<resolution>.onnx`
- `trained-<resolution>.onnx`
- `post-training-processed-<resolution>.onnx`

**Ground Truth Reference**
```bash
./bevformer_inference.py ground-truth \
  bevformer_lib/projects/configs/bevformer/bevformer_tiny.py \
  --end-scene=5
```

----

## ONNX Model Inputs and Outputs

Where:
- `B`: batch size
- `W`: image input width in pixels
  - e.g. `1600` or `800`
- `H`: image input height in pixels
  - e.g. `900` or `450`
- `C`: image channels
  - `3`
- `N`: number of cameras
  - `6`
- `bev_w`: BEV space columns
  - Tiny: `50`
  - Base: `200`
- `bev_h`: BEV space rows
  - Tiny: `50`
  - Base: `200`
- `embed_dims`: embedding dimension
  - `256`
- `D`: number of decoder layers
  - `6`
- `Q`: number of object queries
  - e.g. `900`
- `num_classes`: number of object classes
  - `10`
- `code_size`

**Inputs**

| Input          | Shape                                   | Description |
|----------------|-----------------------------------------|-------------|
| `img`          | [`B`, `N`, `C`, `H`, `W`]               | The 6 camera inputs |
| `can_bus`      | [`B`, `18`]                             | 18 data fields from the can bus (e.g. position, orientation, velocity, acceleration) |
| `lidar2img`    | [`B`, `N`, `4`, `4`]                    | Despite the "lidar" this is a per-camera projection matrix to convert from the common 3D coordinate space (canonically lidar-centered on nuScenes) to each camera's image (pixel) coordinates |
| `prev_bev`     | [`B`, `bev_h` * `bev_w`,  `embed_dims`] | BEV feature state carried forward between frames |
| `use_prev_bev` | [`1`]                                   | Used as a boolean flag of whether or not to use the prev_bev (namely at the start of a new scene and in retraining at the start of constructing historical BEV from a sequence of samples) |

**Outputs**

| Output          | Shape                                  | Description |
|-----------------|----------------------------------------|-------------|
| outputs_coords  | [`B`, `D`, `Q`, `code_size`]           | Per decoder layer regression for each object query as 3D bounding boxes (see `projects/mmdet3d_plugin/core/bbox/util.py::denormalize_bbox`, channels are `(cx, cy, log w, log l, cz, log h, sin θ, cos θ, vx, vy)`). Only the last decoder layer is used for inference. |
| bev_embed       | [`B`, `bev_h` * `bev_w`, `embed_dims`] | Encoder output: the BEV feature grid, used both as the decoder's K/V and as the prev_bev pass to the next frame. |
| outputs_classes | [`B`, `D`, `Q`, `num_classes`]         | Per decoder layer class logits per query. Only the last decoder layer is used at inference.|
