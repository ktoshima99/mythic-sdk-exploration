# 将来課題: BEVFormer 推論の実機実行（FP32 vs アナログ動画比較）

状態: **未実行（将来課題）**。実行要件の調査のみ完了。

## 目的

BEVFormer の推論を 2 通りで実行し、結果動画を比較する:
- **FP32 相当**（アナログノイズなし）
- **アナログaware**（Mythic の量子化＋アナログノイズ込み、doc 03 Part A の `munc.TorchNet` 経路）

さらに（当初指示の発展として）**再学習(train) → 精度シミュレーション(eval_trained)** のワークフロー実行も対象。

処理の詳細は [03_accuracy_simulation.md](03_accuracy_simulation.md) の A.13（推論・可視化パイプライン）および A.1〜A.12（精度シミュレーション本体）を参照。

---

## 実行環境の調査結果（確認済み）

| 要素 | 状態 | 備考 |
|---|---|---|
| GPU | ✅ 利用可 | ホストに NVIDIA L40S (46GB)。コンテナは `--gpus all` で起動が必要（`gpu_run_mythic_sdk_container.sh` を使用） |
| Python 環境 | ✅ 導入済み | SDK コンテナ `mythic-sdk-ubuntu-24.04:m2000-v26.05.0` の venv に munc / mmdet3d / mmcv / torch 2.11.0+cu129 が導入済み・import 動作確認済み |
| 学習済み ONNX | ✅ ホストにあり | `training-models-installer-m2000-v26.05.0/archive/models/training/bevformer/`：<br>・`bevformer-tiny-fp32-1600x900.onnx`（FP32、5871 ノード）<br>・`bevformer-tiny-1600x900-trained.onnx`（アナログ学習済み、2126 ノード、Mythic 変換済み）<br>両者とも標準 ONNX オペのみ（`domain=''`）、opset 20、入出力ノードは同一（img/can_bus/lidar2img/prev_bev/use_prev_bev → bev_embed/outputs_classes/outputs_coords） |
| nuScenes mini 本体 | ✅ 用意あり | `s3://s3-srdm-iit-intern-2026/internship-2026/v1.0-mini/`（`maps/ samples/ sweeps/ v1.0-mini/` と LICENSE）。メタデータ JSON 一式（attribute/calibrated_sensor/ego_pose/sample/sample_annotation/sample_data 等）を確認済み |
| **CAN bus 拡張** | ❌ **未確認/不足** | 上記 S3 パスに `can_bus/` が見当たらない。BEVFormer は `use_can_bus=True`（config `bevformer_tiny-1600x900.py`）で CAN bus（自車の位置・向き・速度・加速度など 18 フィールド）を必須とする。**これが最大のブロッカー** |
| **annotation (.pkl)** | ❌ 未生成 | `generate_nuscenes_annotations` ステップで `nuscenes_infos_temporal_{train,val}.pkl` を生成する必要がある。生成には CAN bus 拡張が要る見込み |

---

## 実行に必要なステップ（想定手順）

### 前提: データ配置とパス上書き
Mythic 側 config `mythic-model-zoo/configs/bevformer/bevformer_tiny.yaml` の以下がデータパスを規定（mmcv 側 config の `data_root`/`data_annotation_root` を上書き）:
```yaml
nuscenes_root: /data/shared/global/datasets/nuscenes/nuscenes/
nuscenes_data_annotation_root: /data/shared/global/datasets/nuscenes/nuscenes/annotations/bevformer/mythic/
pytorch_checkpoint_root: /data/shared/global/datasets/bevformer/
```
→ nuScenes mini（+ CAN bus）をこの構造に配置するか、config を上書きする。README（`bevformer_inference_support/README.md`）§Dataset Setup に対応。

### 手順
1. **CAN bus 拡張の入手**（未解決）: nuScenes CAN bus expansion を入手し `nuscenes_root` 直下に `can_bus/` として配置。
2. **annotation 生成**（1 回のみ、1時間以上かかる場合あり）:
   ```bash
   python3 scripts/common/convert_model.py steps=generate_nuscenes_annotations
   ```
   → `nuscenes_infos_temporal_{train,val}.pkl` を生成。
3. **（再学習ルートの場合）train → eval_trained**:
   ```bash
   source ./scripts/bevformer/bevformer-tiny-1600x900.env   # 解像度選択
   python3 scripts/common/convert_model.py steps=train           # アナログaware再学習（GPU必須・時間がかかる）
   python3 scripts/common/convert_model.py steps=eval_trained    # 実データセットで精度評価（mAP/NDS等）
   ```
4. **（動画比較ルートの場合）bevformer_inference.py**:
   ```bash
   cd mythic/model_zoo/bevformer
   # FP32相当（ONNX Runtime、ノイズなし）
   ./bevformer_inference.py onnx \
     <path>/bevformer-tiny-fp32-1600x900.onnx \
     bevformer_lib/projects/configs/bevformer/bevformer_tiny-1600x900.py \
     --end-scene=5 --map-bev --data-type=sweeps
   # アナログaware（TorchNet、量子化＋ノイズ込み）
   ./bevformer_inference.py torchnet \
     <path>/bevformer-tiny-1600x900-trained.onnx \
     bevformer_lib/projects/configs/bevformer/bevformer_tiny-1600x900.py \
     --end-scene=5 --map-bev --data-type=sweeps
   ```
   → 各シーンの `scene.mp4` を FP32 版とアナログ版で見比べる。

> 注意: 当初要望の「`pytorch` サブコマンド(FP32)」は `.pth` チェックポイントが必要だが手元は `.onnx` のみ。FP32 相当は上記の通り `onnx` サブコマンド + `bevformer-tiny-fp32-1600x900.onnx` で代替する。

---

## ブロッカーと未解決事項

1. **CAN bus 拡張データが未確認**（最優先）。これが無いと annotation 生成・推論のどちらも動かない見込み。
   - 対応案: (a) CAN bus 拡張を別途入手・配置する、(b) 生成済み `.pkl` があればそれを直接使う、(c) config で `use_can_bus=False` にできるか要調査（BEVFormer の時系列 attention が CAN bus 依存のため精度低下・エラーの恐れ）。
2. **再学習(train) の所要時間**: BEVFormer tiny でも GPU で相応の時間。mini（10 シーン）での短時間実行が現実的かは要検証。
3. **解像度選択**: 手元の学習済み ONNX は 1600x900。より軽い 800x450 で回すには対応する ONNX/チェックポイントが別途必要。
4. **コンテナ GPU 起動**: 解析では `docker exec`（CPU）で環境確認したが、実行時は `--gpus all` 付きで起動する必要がある（`gpu_run_mythic_sdk_container.sh` 参照。ただし同スクリプトのデフォルトのデータマウントパスは Collaboration Chamber 向けなので上書きが必要）。

---

## 再開時のチェックリスト

- [ ] CAN bus 拡張データの所在を確定（または `.pkl` の入手）
- [ ] nuScenes mini + CAN bus を `nuscenes_root` 構造に配置（S3 → ローカル）
- [ ] SDK コンテナを `--gpus all` で起動、venv 有効化
- [ ] `steps=generate_nuscenes_annotations` で `.pkl` 生成
- [ ] 動画比較: `bevformer_inference.py onnx`（FP32）と `torchnet`（アナログ）を実行し `scene.mp4` を比較
- [ ] （任意）`steps=train` → `steps=eval_trained` で再学習〜精度評価を実行
