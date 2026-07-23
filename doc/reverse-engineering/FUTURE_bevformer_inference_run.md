# 将来課題: BEVFormer 推論の実機実行（FP32 vs アナログ動画比較）

> **動画を作る作業だけが目的なら、経緯を読まずに [HOWTO_bevformer_carla_video_generation.md](HOWTO_bevformer_carla_video_generation.md) の手順書を直接使うこと。** 本ファイルは調査の経緯・発見した不具合の詳細記録。

状態: **CARLA データでの推論実行 + 検出ゼロの原因特定 + 修正版データでの検出復活確認まで完了**。CAN bus 拡張の代替として CARLA シミュレーション生成データセットを前処理し、`generate_nuscenes_annotations` → `bevformer_inference.py`（FP32 / TorchNet, GPU）まで実行した（§「CARLA データセットの前処理と annotation 生成（完了）」および §「CARLA データでの推論実行（FP32 / TorchNet）と結果検証（完了）」参照）。
- **配布 ONNX は学習済みと確定**（sampling_offsets weight の非ゼロ性で立証）。
- 旧 CARLA データで**検出がほぼ得られない**（生スコア最大 ≈0.118 < 既定閾値0.3）原因を **`ground-truth` 可視化 + 投影行列の数値検証で特定**（§「検出ゼロの根本原因特定」参照）。**原因はドメインギャップではなく、CARLA データセット側のカメラ外部パラメータのバグ**——`calibrated_sensor.rotation` に光学座標系→車体座標系の変換（≈90°軸入れ替え）が含まれておらず、`lidar2img` 投影で GT box の 40 中 38 個がカメラ背後に落ちる。BEVFormer の SpatialCrossAttention が画像特徴を参照できないため検出が出ない。
- **データセット側への修正要望を [DATASET_ISSUE_carla_camera_extrinsics.md](DATASET_ISSUE_carla_camera_extrinsics.md) にまとめた**（CARLA 生成チーム向け）。
- **修正版データセット（`/mnt/nvme_scratch/nuscenes_carla`、3シーン60サンプル）で再検証し、検出復活を確認**（§「修正版データセットでの再検証」参照）。FP32/TorchNet ともに閾値0.3で検出が出るようになった（FP32: 4件、TorchNet: 9件）。原因特定が正しかったことが実証された。
- 残っていた「検出漏れ・低確信度」を生スコアで切り分け（§「残る見逃しの原因調査」参照）——**こちらはドメインギャップ**（座標変換ではなく分類ヘッドの確信度不足）。
- **交通参加者を大幅に増やした第3版データセット（4シーン215サンプル、instance 189件、annotation 9086件）で再検証**（§「交通参加者を増やした大規模データセットでの検証」参照）。**FP32 626件・TorchNet 611件の検出**（スコア最大0.91）まで大幅改善。ドメインギャップは主要因ではなく、単純にオブジェクト密度が低かったことが検出の少なさの主因だったと判明。
- GT 動画目視で「後方3カメラ(CAM_BACK/BACK_LEFT/BACK_RIGHT)に box が出ない」現象を発見・原因特定 → **CARLA の LiDAR が後方180°を一切スキャンしていない**ことが原因（`num_lidar_pts`フィルタで後方物体が全除外）。データセット側への修正要望を別プロジェクト（`/home/ubuntu/carla_project/requests/DATASET_ISSUE_lidar_rear_coverage.md`）にまとめた。
- **LiDAR 360°化 + カメラ解像度1600x900統一済みの第4版データセットで再検証**（§「LiDAR 360°化データセットでの検証」参照）。LiDAR点群の前後カバレッジがほぼ均等（前48.8%/後51.2%）になり、後方カメラにも GT box が正しく描画されるようになった。FP32 38件・TorchNet 41件検出（4サンプルのみ、最大スコア0.90）。
- 動画が短い（val split が4サンプルのみ）という指摘を受け、**第5版データセット（3シーン33サンプル、LiDAR 360°維持）で train split の最長シーン(22サンプル)を使い動画を生成**（§「第5版データセットでの検証と長尺動画の生成」参照）。FP32 234件・TorchNet 241件検出。**副次的に発見した SDK 側の未解明挙動**（データローダーがシーン順を並べ替え、1本の動画が複数の断片に分割される）も記録した。
- **第6版データセット（2シーン171サンプル、1シーンあたり85-86サンプルの大規模）で最長尺の動画を生成**（§「第6版データセットでの検証（86サンプル、単一シーン動画）」参照）。単一シーン86サンプル全部が1本の動画として途切れず生成できた（前セクションの分割問題は再発せず）。**FP32 821件・TorchNet 856件検出**（最大スコア0.92）と最良の結果。
- nuScenes 実データ（mini/trainval）での推論、および CARLA データでの `steps=train` 再学習は依然未着手。

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

1. **CAN bus 拡張データが未確認**（nuScenes 実データに関しては依然未解決）。これが無いと annotation 生成・推論のどちらも動かない見込み。
   - 対応案: (a) CAN bus 拡張を別途入手・配置する、(b) 生成済み `.pkl` があればそれを直接使う、(c) config で `use_can_bus=False` にできるか要調査（BEVFormer の時系列 attention が CAN bus 依存のため精度低下・エラーの恐れ）。
   - **[進展]** CARLA シミュレーション生成データセットには CAN bus 相当のデータ（`can_bus/*_pose.json`）が最初から含まれており、これを使った annotation 生成は完了済み（下記セクション参照）。nuScenes 実データ（mini/trainval）自体の CAN bus 拡張入手は依然未解決だが、パイプライン自体の動作確認という当初の目的の一部は CARLA データで代替達成できた。
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

---

## 補足: BEVFormer-Tiny の analog/digital 演算振り分け（実 artifact で確認済み）

上記の実機推論実行（未着手）とは別に、**学習済み artifact を静的に解析することで、どのモデル部分がアナログ(BCM/ACE)割り当てで、どこがデジタル割り当てかは既に確認済み**。以下は `training-models-installer-m2000-v26.05.0/archive/models/training/bevformer/bevformer-tiny-1600x900-trained.tar.gz` を実際に展開し、コンパイラ入力の3分割サブグラフ（`compiler_ready_artifact_{off_chip_0,on_chip_1_bcm,off_chip_2}.onnx`）の `op_type` をノード単位で集計した結果（doc 03 Part A/doc 01 の判定基準と整合）。

### ノードレベルの振り分け

| モデル部分 | 割り当て | 根拠 |
|---|---|---|
| ResNetバックボーン（画像特徴抽出のConv） | アナログ（BCM/ACE） | `on_chip_1_bcm.onnx` の `BCMConv2d`×74 |
| FPN(img_neck) lateral/fpn convs | アナログ | 同上 |
| BEVFormerEncoder（TemporalSelfAttention・SpatialCrossAttention・GridSample） | デジタル | `off_chip_2.onnx` |
| DetrTransformerDecoder（MultiheadAttention） | デジタル | 同上 |
| 検出ヘッド（reg/cls branches, MatMul/Gemm） | デジタル | 同上 |
| 入力前処理（正規化） | デジタル（軽量） | `off_chip_0.onnx` |

補足（ノード数・op_type 集計）: `off_chip_0`(5ノード)=`Mul/Reshape/Add/Clip`、`on_chip_1_bcm`(148ノード)=`BCMConv2d`×74,`BCMAdd`×16,`BCMSum`×13,`BCMMul`×12,`MaxPool`×1,`Slice`×32（重み512超のタイル分割用）、`off_chip_2`(1955ノード)=`MatMul`×155,`Gemm`×6,`LayerNormalization`×40,`Softmax`×18,`GridSample`×13,`Sigmoid`×8等。

ノード名から特定したモデル構造との対応:
- **アナログ側(`on_chip_1_bcm`)**: `n_ResNet_conv1_Conv`〜`n_ResNet_layer4_2_Bottleneck_conv3_Conv`（ResNet-50 バックボーン）、`n_img_neck_lateral_convs_*`/`n_img_neck_fpn_convs_*`（FPN）。**画像特徴抽出用CNN(ResNet+FPN)のConv層のみ**。
- **デジタル側(`off_chip_2`)**: `n_BEVFormerHead_PerceptionTransformer_BEVFormerEncoder_*_TemporalSelfAttention_*`、`...SpatialCrossAttention...`（`GridSample`含む）、`...DetectionTransformerDecoder_*_MultiheadAttention_Gemm`、`reg_branches`/`cls_branches`（検出ヘッド）。**Transformer エンコーダ/デコーダ・検出ヘッドは全てデジタル**。

これは doc 01 の判定基準（`Conv`/`Dense`/`MmaDot` はアナログ実行可、Attention/LayerNorm 等はデジタル）が BEVFormer-Tiny では「CNNバックボーンのみアナログ、Transformer 部分は全てデジタル」という形で具現化されていることの実例。

### MAC数で見た比率（`Compiler Optimization Report - BEVFormer-Tiny.pdf` の数値の正しい解釈）

PPA レポート（p.4-5）記載の MAC 数は、**アナログ側=1カメラ入力換算、デジタル側=6カメラ入力換算**という異なる前提で計算されている（レポートに明記: "assuming 1-camera input for the Analog NPU part... assuming 6-camera input for the Digital NPU part"。バックボーンはカメラ画像を1枚ずつ順次処理するが、Transformer部分は6カメラ分をまとめて1回処理するため）。単純に2値を比較すると誤り。6カメラ推論1回あたりに揃えると:

```
Analog MACs (6カメラ換算) = 159,170,560,000 × 6 = 955,023,360,000
Digital MACs (6カメラ換算) = 16,529,000,000（レポートの値がそのまま6カメラ基準）
合計 = 971,552,360,000

Analog比率 = 98.3%
Digital比率 = 1.7%
```

**MAC数で見るとアナログ側が約98.3%を占め、ほぼ全ての演算がResNet+FPNバックボーンに集中する。** 一方、処理時間で見るとアナログ27.30ms・デジタル4.63ms（デジタル比14.5%、PPAレポート "Combined Analog + Digital NPU Latency"）で、MAC数の偏りほどには時間の偏りは大きくない——デジタル側はMAC単価が低いがレイテンシに効く演算（MatMul/LayerNorm/Softmax/GridSample等のシーケンシャル処理・メモリアクセス）が多いためと考えられる[推測]。

### 参照
- artifact: `training-models-installer-m2000-v26.05.0/archive/models/training/bevformer/bevformer-tiny-1600x900-trained.tar.gz`（展開先 `compiler_ready_artifact/reference/compiler_ready_artifact_{off_chip_0,on_chip_1_bcm,off_chip_2}.onnx`）
- PPA レポート: `doc/reports/Compiler Optimization Report - BEVFormer-Tiny.pdf`
- 分割・BCM変換のロジック: [03_accuracy_simulation.md](03_accuracy_simulation.md) A.6/A.12、振り分け判定基準: [01_compilation.md](01_compilation.md) §3.3.3

---

## CARLA データセットの前処理と annotation 生成（完了）

状態: **完了**。CARLA シミュレーションで生成された nuScenes テーブル形式データセット（`s3://s3-srdm/nuscenes_carla/mythic_sdk_test/`）を Mythic SDK の BEVFormer-tiny パイプラインが読める形に変換し、`generate_nuscenes_annotations` の実行（`nuscenes_infos_temporal_{train,val}.pkl` 生成）まで通した。

### 背景

nuScenes 実データ（mini/trainval）は CAN bus 拡張・メタデータ一式が未入手のままブロックされていた（上記参照）。一方、CARLA シミュレーションで独自生成されたデータセットが用意され、nuScenes のテーブルスキーマ（`category`/`attribute`/`sample_annotation`/`calibrated_sensor`等）に準拠していたため、これを Mythic SDK 向けに前処理して使えるようにした。

### 発見した不整合と対処

CARLA データはスキーマとしては互換だが、`nuscenes-devkit` と BEVFormer 変換コード（`nuscenes_converter.py`）が要求する**厳密な文字列・形状一致**を満たしていなかった:

| 不整合 | 内容 | 対処 |
|---|---|---|
| シーン名形式 | `full_sensor_test_20260720_...` のような自由形式。`NuScenesCanBus.get_messages()` が `^scene-\d\d\d\d$` を assert するため使用不可 | `scene-9001`/`scene-9002`/`scene-9003` にリネーム |
| カメラチャンネル名 | `CAM_FRONT_WIDE`(FOV~110°)/`CAM_FRONT_NARROW`(FOV~23°) の2種のみで、標準の `CAM_FRONT` が存在しない。`sample['data'][cam]` の完全一致文字列参照で使えない | `CAM_FRONT_NARROW`(実nuScenesのCAM_FRONT(~65°)に近い)を`CAM_FRONT`にリネーム。`CAM_FRONT_WIDE`は不使用として除外 |
| カメラ解像度混在 | `CAM_FRONT_NARROW` 3840×2160、他5カメラ 1280×720。`LoadMultiViewImageFromFiles`が`np.stack`で6カメラ同一shapeを要求するため失敗 | 全カメラを1600×900にリサイズ（既存学習済みartifact `bevformer-tiny-1600x900-trained.tar.gz` と同解像度）、`calibrated_sensor.camera_intrinsic` を width/height比で再計算 |
| train/val分割 | `create_nuscenes_infos` は実在nuScenesのシーン名固定リスト（`nuscenes.utils.splits.train`(700件)/`.val`(150件)）との一致で振り分ける。CARLAのシーン名はどちらにも一致せず、何もしなければ全件valに入りtrainが空になる | `sitecustomize.py` による非破壊パッチで、リネーム後のシーン名（`scene-9001`,`scene-9002`→train、`scene-9003`→val）を該当リストに追加登録 |
| `instance.json`欠損（データ自体の不具合、初回提供分のみ） | 初回データでは`instance.json`が46件しかなく、3シーン中1シーンのみに対応。他2シーンの`sample_annotation`（6509件中4423件、68%）が参照する`instance_token`が存在せず孤立 | CARLA側チームに報告。再アップロードされた修正版データでは全3シーン分（148件instance、孤立0件）に解消していることを確認済み |

### 前処理スクリプト

`preprocess_carla_to_bevformer.py`（ホスト: `tools/preprocess_carla_to_bevformer.py`）。SDKコンテナのvenv Python（PIL/cv2/numpy既存）で実行するスタンドアロンスクリプト。S3の元データは読み取り専用で扱い、変更しない。

処理段階: ①S3からのダウンロード確認 → ②JSONテーブル読み込み → ③シーンリネーム → ④センサーフィルタ（`CAM_FRONT_WIDE`除外、`CAM_FRONT_NARROW`→`CAM_FRONT`） → ⑤`calibrated_sensor`のintrinsics再計算 → ⑥`sample_data`のフィルタ+パス書き換え → ⑦画像リサイズ（cv2、約2700枚） → ⑧マップマスクPNG生成（`carla-town10.json`の`canvas_edge`から実寸算出） → ⑨CAN busファイルのシーン名リネームコピー → ⑩テーブル書き出し → ⑪サマリ出力。

実行例:
```bash
docker exec mythic_sdk_impl /root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 \
  /workspace/preprocess_carla_to_bevformer.py \
  --scratch-dir /workspace/carla_scratch \
  --output-root /workspace/carla_output \
  --force --workers 8
```

出力: `sample_data`が3527→3067件（CAM_FRONT_WIDE分460件除外）、155サンプル、2760枚を1600×900にリサイズ。

### train/val分割パッチ（`sitecustomize.py`）

```python
# 非破壊パッチ。削除すれば完全に元に戻る（他ファイルは一切変更しない）。
from nuscenes.utils import splits as _nusc_splits
_CARLA_TRAIN = ['scene-9001', 'scene-9002']
_CARLA_VAL = ['scene-9003']
for _name in _CARLA_TRAIN:
    if _name not in _nusc_splits.train:
        _nusc_splits.train.append(_name)
for _name in _CARLA_VAL:
    if _name not in _nusc_splits.val:
        _nusc_splits.val.append(_name)
```
`PYTHONPATH`環境変数でこのファイルを指すディレクトリを指定すると、Python起動時に自動実行される（`sitecustomize`は特別なモジュール名）。SDKコンテナは`--rm`の使い捨てのため、venv内`site-packages`への`docker cp`ではなく、ホスト側永続領域（`$MYTHIC_WORKSPACE`配下等）に置いて`PYTHONPATH`で読ませる方式を採用——コンテナ再作成に対して耐性がある。

### `generate_nuscenes_annotations`の実行

`mythic.model_zoo.bevformer.conversion_steps.generate_nuscenes_annotations`は内部で`v1.0-trainval`と`v1.0-test`の両方に対し`nuscenes_data_prep`を呼ぶ。CARLAデータには`v1.0-test`テーブルが無いため、**`v1.0-trainval`ディレクトリをコピーして`v1.0-test`としてスタブする**必要があった（test split自体は使わないため実害なし、正式運用では別途本物のtestシーンを用意すべき）。

`can_bus_root_path`引数は`nuscenes_root`そのもの（`NuScenesCanBus`が内部で`can_bus`サブディレクトリを付加するため、`.../can_bus`まで含めると二重になりエラーになる）という実装上の注意点があった。

実行結果（`PYTHONPATH`にパッチを設定した状態で`nuscenes_data_prep`を`version='v1.0-trainval'`で直接呼び出し）:
```
total scene num: 3
exist scene num: 3
train scene: 2, val scene: 1
train sample: 103, val sample: 52
```
生成物: `nuscenes_infos_temporal_{train,val}.pkl`, `nuscenes_infos_temporal_{train,val}_mono3d.coco.json`, `nuscenes_devkit_v1.0-trainval.pkl`（devkitキャッシュ）。pkl内容を検証し、各サンプルに6カメラ全て(`CAM_FRONT`含む)のパス・`can_bus`18次元ベクトル・`gt_names`（カテゴリマッピング結果）が正しく記録されていることを確認済み。

### 未対応・次のステップ

- 生成した`.pkl`を使った実際の`train`/`to_onnx`/`eval_trained`ステップの実行は未着手
- `v1.0-test`は`v1.0-trainval`のコピーでスタブしたのみで、本物のtestシーンは用意していない
- nuScenes実データ（mini/trainval）でのCAN bus拡張入手は依然未解決（本セクションはCARLAデータでの代替達成のみ）
- 3シーンという小規模データでのtrain(2)/val(1)分割はスモークテスト用の判断であり、本格学習時は再検討が必要

### 参照
- 前処理スクリプト: `tools/preprocess_carla_to_bevformer.py`
- パッチファイル: `tools/sitecustomize.py`
- 元データ: `s3://s3-srdm/nuscenes_carla/mythic_sdk_test/`
- 設計計画: `/home/ubuntu/.claude/plans/sdk-wild-snowflake.md`
- 関連コード: `nuscenes/nuscenes.py`（`NuScenes.__init__`）, `nuscenes/utils/map_mask.py`, `nuscenes/can_bus/can_bus_api.py`, `nuscenes/utils/splits.py`, `mythic/model_zoo/bevformer/bevformer_lib/tools/data_converter/nuscenes_converter.py`, `mythic/model_zoo/bevformer/bevformer_lib/tools/create_data.py`, `mythic/model_zoo/bevformer/conversion_steps.py::generate_nuscenes_annotations`

---

## CARLA データでの推論実行（FP32 / TorchNet）と結果検証（完了）

状態: **完了**。前処理済み CARLA データセット + 生成済み `.pkl` を使い、`bevformer_inference.py` の `onnx`（FP32）と `torchnet`（アナログaware）両サブコマンドで実際に推論を実行し、動画を生成した。ただし**検出結果は実質的に得られず**、その原因調査までを行った。

### 実行に必要だった環境設定

1. **推論用 mmcv config の複製**: `bevformer_inference.py` は `data_root`/`data_annotation_root` を CLI では受け取らず、mmcv config (`bevformer_tiny-1600x900.py`) 内のハードコード値を使う。CARLA データを指す複製 config `bevformer_tiny_carla.py` を作成し、`data_root='/workspace/carla_output/'`, `data_annotation_root='/workspace/annotations/'` に書き換えて同ディレクトリ（`_base_` の相対パス解決のため）に配置。ホスト側にも `tools/bevformer_tiny_carla.py` として保存。
2. **DataLoader の共有メモリ問題**: 既定 `workers_per_gpu=4` のマルチプロセス DataLoader が、コンテナ既定の `/dev/shm`(64MB) 不足で `RuntimeError: unable to allocate shared memory` になる。対処として (a) コンテナを `--shm-size 512m` で起動、かつ (b) 複製 config の `workers_per_gpu=0`（単一プロセス化）とした。両方必要だった。
3. **GPU 有効化**: 当初 `--gpus all` なしで起動していたため CPU 実行になっていた。ホストに NVIDIA L40S (46GB) + Docker `nvidia` ランタイムがあるので、`--gpus all --shm-size 512m` で再作成。`torch.cuda.is_available()==True`、TorchNet の `--device` 既定が `cuda:0` になり自動で GPU 実行される。
   - **効果**: TorchNet 推論(52フレーム)が CPU 約 **1時間** → GPU 約 **2分30秒**（約24倍高速）。
   - **注意**: SDK 同梱の `onnxruntime` は **CPU版のみ**（`onnxruntime-gpu` 未導入、`get_available_providers()` に CUDA 系が無い）。そのため `onnx` サブコマンド（ONNXRuntime 経路）は GPU 化できない。ただし FP32 側は元々数分で完走するため実害は小さい。

### 実行コマンド（GPU コンテナ内、`PYTHONPATH=/workspace/patches` でsplitパッチ有効化）

```bash
# FP32 (ONNX Runtime, CPU)
python3 .../bevformer_inference.py onnx \
  /workspace/bevformer_models/bevformer-tiny-fp32-1600x900.onnx \
  .../configs/bevformer/bevformer_tiny_carla.py \
  --output-dir /workspace/inference_out --end-scene=1 --data-type=samples

# TorchNet (アナログaware, GPU cuda:0)
python3 .../bevformer_inference.py torchnet \
  /workspace/bevformer_models/bevformer-tiny-1600x900-trained.onnx \
  .../configs/bevformer/bevformer_tiny_carla.py \
  --output-dir /workspace/inference_out_torchnet --end-scene=1 --data-type=samples
```

### 結果: 検出がほぼ得られない（スコア閾値に届かない）

- **既定スコア閾値 0.3**: FP32・TorchNet とも検出 **0 件**。両動画は**画素単位で完全一致**（全フレーム MD5 一致）——検出box が描画されないため、同じカメラ画像＋空 BEV＋自車マーカーとなり見た目が同一になるのは筋が通る。以前見えた「赤い矩形」は検出結果ではなく自車位置マーカー(`_layer_ego`, 色 `(30,30,180)`)だった。
- **生スコア分布の直接確認**（FP32 ONNX を dataloader 1 サンプルに適用）: `sigmoid(cls_scores)` の **最大が約 0.118**。0.3 超は 0 件、0.1 超 51 件、0.05 超 624 件。
- **スコア閾値 0.08 に下げて再実行**: 大量の紫box（bicycle/motorcycle クラス相当）が BEV に散乱するが、カメラ画像に対応物体がほぼ無い**ノイズ的な誤検出**。この閾値では FP32 と TorchNet の動画 MD5 は**異なる**（BCM アナログノイズが出力に差を生んでいることは確認できた）が、意味のある検出比較にはならない。

生成動画（ホスト側）:
- `bevformer_carla_fp32_scene0.mp4`（閾値0.3, 検出0件）
- `bevformer_carla_torchnet_scene0.mp4`（閾値0.3, FP32と同一）
- `bevformer_carla_fp32_lowthr_scene0.mp4`（閾値0.08）
- `bevformer_carla_torchnet_lowthr_gpu_scene0.mp4`（閾値0.08, GPU実行）
> mp4 は cv2 `mp4v` コーデックのため Ubuntu 標準プレイヤーで再生できないことがある。VLC (`sudo apt install vlc`) で再生可。

### モデルの学習状態の検証（重要 — 初回誤判定 → 訂正）

検出が出ない原因が「モデル未学習」なのか「データのドメインギャップ」なのかを切り分けるため、配布 ONNX の重みを検証した。

**最終結論: 両 ONNX（`bevformer-tiny-fp32-1600x900.onnx` と `bevformer-tiny-1600x900-trained.onnx`）は実際に学習済みモデルである。** ユーザーガイドの「再学習済みモデル」という記述は正しい。

- **決定的証拠 — `sampling_offsets.weight`**: BEVFormer/Deformable-DETR は Deformable Attention の `sampling_offsets` weight を `constant_init(..., 0.)` で**厳密に 0** 初期化する仕様（`mmcv/ops/multi_scale_deform_attn.py:249`）。実測は全層で `mean_abs ≈ 0.006〜0.015`, `std ≈ 0.008〜0.015` と明確に非ゼロ。数値誤差(1e-6〜1e-7)より3〜4桁大きく、勾配更新（学習）以外では説明不可能。
- **補強証拠 — BatchNorm running_var**: 53 層すべてで running_var が 1.0 付近になく（未学習なら≈1.0）、フォワード統計が更新済み。ただし ImageNet 事前学習だけでも起こるため単独では弱い（BEVFormer 特有の sampling_offsets の方が本質的）。
- 破損チェック: NaN/Inf・全ゼロテンソルなし。fp32版と trained(量子化)版は対応重みに強い相関があり同一学習重み由来。

> **初回の誤り記録（教訓）**: 当初 `cls_branches.*.6.bias` が `bias_init_with_prob(0.01)=-4.595` に近い値（実測 -4.55）であることだけを見て「未学習」と誤判定した。FocalLoss の分類バイアスはクラス不均衡対策で**学習後もほとんど動かない**設計であり、初期値に近いこと自体は「未学習の証拠」にならない。判定力の高い層（0初期化される sampling_offsets）を見るべきだった。

### 未解明: なぜ検出が出ないのか（→ 特定済み、下記セクション参照）

当初は「ドメインギャップ」か「前処理・座標系の不整合」の 2 候補で未確定だったが、次セッション（本セッション）で **`ground-truth` 可視化 + 投影行列の数値検証** により**後者（CARLA データセット側のカメラ外部パラメータのバグ）**と特定した。詳細は次セクション。

### 参照（本セクション追加分）
- 推論用 config: `tools/bevformer_tiny_carla.py`（`data_root`/`workers_per_gpu=0` を変更）
- 推論 CLI: `mythic/model_zoo/bevformer/bevformer_inference.py`（`onnx`/`torchnet`/`pytorch`/`ground-truth` サブコマンド）
- 描画: `bevformer_lib/custom_utils/visualization.py`（`_layer_ego` 自車マーカー, `CLASS_COLORS`）
- 後処理・スコア閾値: `bevformer_lib/custom_utils/processing.py::post_process`
- 配布 ONNX（学習済み、S3 と同一 MD5）: `training-models-installer-m2000-v26.05.0/archive/models/training/bevformer/{bevformer-tiny-fp32-1600x900.onnx, bevformer-tiny-1600x900-trained.onnx}`、S3: `s3://s3-srdm/model_onnx/training/bevformer/`
- 初期化仕様: `mmcv/ops/multi_scale_deform_attn.py::init_weights`（sampling_offsets を 0 初期化）, `mmcv.cnn.bias_init_with_prob`

---

## 検出ゼロの根本原因特定（完了）

状態: **完了**。「検出が出ないのはドメインギャップか前処理バグか」を切り分けるため、`ground-truth` サブコマンドで CARLA データの GT アノテーションを可視化し、投影行列を数値検証した。CAN bus 拡張も再学習も不要で、既存の前処理済みデータのみで実施できる安価な診断。

### 結論

**原因はドメインギャップではなく、CARLA データセット側のカメラ外部パラメータ（`calibrated_sensor.rotation`）のバグ。** カメラ回転が「光学座標系（x=右, y=下, z=前方）→ 車体座標系（x=前, y=左, z=上）」の変換（≈90° の軸入れ替え）を含んでおらず、CARLA の車体座標系のまま格納されている。このため `lidar2img` 投影で LiDAR の z 軸（高さ、常に≈-1.5m）がカメラの奥行き軸に化け、GT box の大半がカメラ背後（負の奥行き）に落ちて画像特徴と対応づかない。

### 診断の経緯と証拠

1. **`ground-truth` 可視化**（`bevformer_inference.py ground-truth ... --data-type=samples`）:
   - **BEV インセットには GT box が正しく描画される**（緑=car, シアン=pedestrian, 赤=自車マーカー中心）→ ego/LiDAR 座標系のアノテーション自体は健全。
   - **カメラ画像には 3D box が 1 つも投影されない**——frame 0 では赤いバン・白い車が明確に写っているのに box ワイヤフレームが皆無。`visualization.py::_draw_boxes_on_image`（L90-）は GT を `lidar2img` でカメラに投影する設計なので、これは投影行列の異常を示す。

2. **投影行列の直接検証**（`nuscenes_infos_temporal_val.pkl` の info[0]、40 GT box を全カメラに投影）:
   - **全 6 カメラの `sensor2lidar_rotation` が同一かつ単位行列**。実 nuScenes では 6 カメラは全て異なる向き（光学系変換込み）を持つはずで、これ自体が異常。
   - **投影奥行きが全 box で ≈ -0.72〜0.62m**（本来は数十 m）。40 box 中 38 個が奥行き負＝カメラ背後判定でクリップ。`nuscenes_dataset.py:130-141` の `lidar2img = viewpad @ lidar2cam_rt.T` 構成に沿って再現。

3. **CARLA 元データ（`calibrated_sensor.json`）のカメラ回転を直接確認**:
   - 全 6 カメラの回転行列で 3 行 3 列目が `[0,0,1]`（z 軸=高さがそのまま保存）→ 光学座標系変換が欠落。
   - ヨー（水平回転）は正しい: CAM_FRONT_WIDE/NARROW=0°, FRONT_LEFT/RIGHT=±50°, BACK_LEFT/RIGHT=±130°, BACK=180°。**CAM_FRONT の回転が単位行列なのはヨー=0（前方向き）が正しいだけで、これ自体はバグではない**（当初 CAM_FRONT のヨー欠落を疑ったが誤り。問題は全カメラ一律の光学系変換欠落のみ）。

4. **修正の実証**: 各カメラ回転に光学座標系変換 `R_opt = [[0,0,1],[-1,0,0],[0,-1,0]]` を右から合成（`sensor2lidar_rotation @ R_opt`）して再投影すると、「カメラ前方」の box が全カメラで 2→12〜26 個に激増し、画像内に収まる box も各カメラ 3〜16 個生じる（6 カメラ合計で 40 box を網羅）。→ 修正方針が正しいことを数値で確認済み。

### 対応

**CARLA 生成データ側の修正が本筋**（前処理での後付け補正も可能だが、正しい extrinsics をデータに持たせるべき）。データセットチーム向けの修正要望を **[DATASET_ISSUE_carla_camera_extrinsics.md](DATASET_ISSUE_carla_camera_extrinsics.md)** にまとめた。

### 修正版データセットでの再検証（完了 — 検出復活を確認）

データセットチームが修正版データを `/mnt/nvme_scratch/nuscenes_carla` に提供。`calibrated_sensor.json` を直接検証したところ、全カメラの回転行列に光学座標系変換が正しく合成されていた（CAM_FRONT の回転行列が本ドキュメントで提案した `R_opt = [[0,0,1],[-1,0,0],[0,-1,0]]` と一致）。

このデータで前処理（`preprocess_carla_to_bevformer.py`）→ `nuscenes_data_prep`（train 37 / val 23 サンプル）→ `ground-truth` 可視化 → FP32(ONNX)/TorchNet 推論のフルパイプラインを再実行:

| 項目 | 修正前（旧データ） | 修正後（新データ） |
|---|---|---|
| GT のカメラ投影 | 0/40 box が画像内 | 正常（例: frame 10 で配送トラックに GT box が正確に重なる） |
| FP32 検出数（閾値0.3） | 0 件 | **4 件**（スコア 0.32〜0.40） |
| TorchNet 検出数（閾値0.3） | 0 件（FP32と画素完全一致） | **9 件**（スコア 0.30〜0.40、FP32と異なる＝アナログノイズの影響を確認） |

→ **検出ゼロの原因が CARLA データセット側のカメラ外部パラメータバグであったことが完全に実証された。** ドメインギャップは主要因ではない。

新データセット用の作業ファイル（コンテナ内、ホスト未永続化——次回は `/mnt/nvme_scratch/nuscenes_carla` から再生成可能）:
- 前処理出力: `/workspace/carla_output_v2`、annotation: `/workspace/annotations_v2`
- 推論用 config: `bevformer_lib/projects/configs/bevformer/bevformer_tiny_carla_v2.py`（`bevformer_tiny_carla.py` の `data_root`/`data_annotation_root` を `_v2` パスに書き換えたコピー）
- 検出数は少数（サンプル数23と小規模データセットのため）。より多くのシーン/サンプルでの評価、および `steps=train` 再学習は依然未着手。

### 残る見逃し（正面トラック等）の原因調査 — こちらはドメインギャップ（座標変換ではない）

修正版データでも、GT 動画で明確に写っている物体（frame 10 の正面トラック、token `1095d644`）が FP32/TorchNet の検出結果（閾値0.3）に出てこないケースがある。これがまだ座標変換の問題なのか、モデル側の弱さなのかを、生スコア（post_process 前）を直接見て切り分けた。

**手法**: `bevformer_inference_impl.py` の `onnx_run_frame`/`post_process` を単体で呼び出し、frame 10 の全 900 query の sigmoid スコアと denormalize 済み box 座標を検証。

**結果**: GT トラック中心（x=26.17, y=0.76）からわずか **0.24m** の位置に実際にモデルの query が存在した（query 503, xyz=[26.23, 1.00, -1.33]）。**位置は合っている。** しかしこの query のクラス別スコアは全クラス 0.06 未満（truck=0.0052, car=0.0099, pedestrian=0.0582 が最大）——「そこに何かある」弱い信号はあるが、どのクラスにも確信を持てていない。一方、モデルが最も自信を持った予測（score=0.29、これでも閾値0.3未達）は座標 (21.62, 0.74) の "car" 判定で、GT のどの物体とも対応しない（偽陽性寄りの信号）。

**結論**: 座標変換・投影行列は正常に機能している（query の空間位置は GT と一致）。残っている問題は**分類ヘッドの確信度がドメインギャップにより全体的に低い**こと——これは前回のカメラ外部パラメータバグとは異なる、モデルの汎化性能側の問題。座標系バグの修正で「検出ゼロ→数件」への復活は説明できたが、「検出漏れ・低確信度」の残存分はドメインギャップに起因すると判断できる。

参照: 再現スクリプトは `bevformer_inference_impl.py::run_onnx_runtime_command` の呼び出し列（`build_dataloader_from_mmcv_config` → `onnx_run_frame` → `processing.py::post_process` 相当の手動デコード）に基づく。次の一手は `steps=train` での CARLA 再学習、または実 nuScenes での同一 ONNX 推論による比較。

### 参照（本セクション追加分）
- GT 可視化コマンド: `bevformer_inference.py ground-truth <config> --output-dir <dir> --end-scene=1 --data-type=samples --map-bev`
- 投影行列構成: `bevformer_lib/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py:123-145`（`sensor2lidar_rotation` → `lidar2cam` → `lidar2img`）
- カメラ投影描画: `bevformer_lib/custom_utils/visualization.py:90-139`（`_draw_boxes_on_image`）, `:893-959`（GT のカメラ投影）
- CARLA 元データ: `s3://s3-srdm/nuscenes_carla/mythic_sdk_test/` の `v1.0-trainval/calibrated_sensor.json`
- 修正要望書: [DATASET_ISSUE_carla_camera_extrinsics.md](DATASET_ISSUE_carla_camera_extrinsics.md)

---

## 交通参加者を増やした大規模データセットでの検証（完了）

状態: **完了**。データセットチームが交通参加者数を大幅に増やした第3版データセットを同じ場所（`/mnt/nvme_scratch/nuscenes_carla`）に提供。**4シーン・215サンプル・instance 189件・annotation 9086件**（第2版は3シーン・60サンプル・instance 13件・annotation 205件——約44倍のオブジェクト数）。カメラ回転は第2版と同様、光学座標系変換が正しく合成済みであることを確認（全カメラ `R_row3` が非ゼロの一貫したパターン）。

### 実行

前処理は 4 シーン対応のため `--scene-names scene-9001,scene-9002,scene-9003,scene-9004` を指定（デフォルトは3シーン固定）。train/val split パッチ（`sitecustomize.py` 相当）も `scene-9004` を追加して `nuscenes_data_prep` 呼び出し内で直接適用。annotation 生成: train 159 samples, val 56 samples。

推論は val split（scene-9003, 56 サンプル）で FP32(ONNX, CPU) と TorchNet(GPU) を実行。

### 結果: 検出数が大幅に改善

| 項目 | 第2版（3シーン, 60サンプル） | 第3版（4シーン, 215サンプル、val 56サンプルで推論） |
|---|---|---|
| FP32 検出数（閾値0.3） | 4 件（スコア0.32〜0.40） | **626 件**（スコア最大 **0.909**、car/pedestrian 中心） |
| TorchNet 検出数（閾値0.3） | 9 件（スコア0.30〜0.40） | **611 件**（FP32とほぼ同数） |

GT 可視化でもカメラ画像・BEV ともに多数の car/pedestrian/bus が確認できる密度になっている（frame 27 で複数車両・信号機・横断歩道の歩行者が同時に写る）。

→ 前セクションで「残る見逃しはドメインギャップ」と暫定結論したが、**この結果を見ると、単純にオブジェクト密度が低い小規模データセット（第2版は1サンプルあたりGT box数が少ない）だったことが検出数の少なさの主因だった可能性が高い**。ドメインギャップ自体が無いとは言えないが、少なくとも「ほぼ検出できない」と評価するほどの深刻な汎化不足ではなく、十分な密度のシーンでは高スコア（0.9台）の検出が普通に出る。

生成動画（ホスト側 `bevformer_carla_videos/`、`bevformer_carla_v3_*.mp4`）:
- `bevformer_carla_v3_groundtruth_scene0.mp4`（GT、val split scene-9003, 56フレーム）
- `bevformer_carla_v3_fp32_scene0.mp4`（FP32, 検出626件）
- `bevformer_carla_v3_torchnet_scene0.mp4`（TorchNet, 検出611件）

### 未対応・次のステップ
- train split（scene-9001/9002/9004、159サンプル）や他バックエンドでの推論は未実施
- `steps=train` での CARLA データ再学習、実 nuScenes での同一 ONNX 比較は依然未着手
- FP32 と TorchNet の検出差（626 vs 611件）の内訳比較（どの検出がアナログノイズで閾値割れしたか）は未実施

### 参照
- 前処理コマンド: `preprocess_carla_to_bevformer.py --scratch-dir /workspace/nuscenes_carla_v2 --output-root /workspace/carla_output_v2 --scene-names scene-9001,scene-9002,scene-9003,scene-9004 --force --workers 8`
- annotation 生成: `create_data.py::nuscenes_data_prep`（`root_path=/workspace/carla_output_v2`）
- 推論 config: `bevformer_lib/projects/configs/bevformer/bevformer_tiny_carla_v2.py`

### 既知の問題（別件、修正済み — 下の「BEVパネルの座標系不整合を修正」参照）: BEV パネルで自車マーカーの向きと box 位置が90°食い違う

修正版データセットでの `ground-truth` 動画（`bevformer_carla_v2_groundtruth_scene0.mp4`）を目視確認したところ、**正面のトラックの GT box が BEV パネル上では右側に表示される**（カメラ画像側の box は正しい）現象を発見。数値シミュレーションで検証した結果、**データセット側の問題ではなく、SDK の可視化コード `bevformer_lib/custom_utils/visualization.py` 内部の不整合**と判明:

- `_layer_boxes`/`_layer_lidar`/`_layer_radar`/`_layer_map_*`（L214-394 付近）は世界座標 `(x=前方, y=左)` を `ix=f(y)`, `iy=f(x, 前方ほど小)` にマッピングし、最終的な「90°CCW回転 + 左右反転」（`_draw_bev_map` L563-566）後には**前方(+x)が画像上で右方向**になる。
- 一方 `_layer_ego`（L186-209）の進行方向矢印はコード中のコメント「Arrow points canvas-right so after the global 90°CCW+flip it points UP」の意図通りには変換されず、実際には最終画像上で**上方向**を向く。

→ 自車マーカーの矢印（上向き）と実際の物体位置（前方=右）の基準が90°ズレている。**検出結果（スコア・座標JSON）やカメラ画像上の投影には影響しない、BEV パネル描画のみのバグ。** 修正はユーザー判断で見送り（現状のまま運用）。

**追加検証（実コード直接実行による確定）**: 上記は当初シミュレーションコードの再実装による推定だったため、「本当にGTデータの座標自体は正しいのか」を確認するため、SDK の実関数 `viz._layer_boxes` / `viz._layer_ego` / `viz._draw_bev_map` の変換部分をそのまま呼び出す再現テストを実施。世界座標で「自車の正面 x=25m, y=0」に置いたテスト box を実際の `_layer_boxes` で描画し、`_layer_ego` の自車マーカーと同じ最終変換（90°回転+反転）を適用した結果:
- 自車マーカーの矢印: 最終画像で **上** 方向（ego centroid y=102 → arrow centroid y=83、y減少=上）
- 世界座標「正面」の box: 最終画像で自車の **右** 側（box x=139 vs ego x=99、同じ y 帯）

→ 実コードでも同じ食い違いを確認。**GT の 3D 座標・box レイヤーの変換自体は自己整合的に正しく、問題は `_layer_ego` の矢印描画だけ**という結論が確定した（カメラ投影が正しいという目視確認とも整合する）。

参照: `visualization.py::_layer_ego`(L186-209), `::_layer_boxes`(L342-394), `::_draw_bev_map`(L508-566)

---

## 既知の問題（別件、報告済み）: 後方3カメラに GT box が描画されない → LiDAR 後方カバレッジ欠落

GT 動画（第3版データセット）の目視で、CAM_BACK/CAM_BACK_LEFT/CAM_BACK_RIGHT に GT box が一切描画されない現象を発見。カメラ投影行列自体は正常（前セクションで検証済みの光学系変換も正しい）にもかかわらず後方だけ空になるため、`get_ann_info`（nuScenes標準）が使う `valid_flag`（＝`num_lidar_pts>0`）フィルタを疑い、pkl の `num_lidar_pts` と自車ローカル座標での前後判定を集計。

**結果**: 検証データ（4シーン215サンプル、9086件）で `num_lidar_pts>0`（GT描画・学習に使われる）の物体は前方1665件・**後方0件**。LiDAR点群ファイル（`.pcd.bin`）自体を角度ビン集計すると、後方180°の角度範囲に点が1つも存在しない（前方のみ74875点）。→ **CARLA の LiDAR シミュレーションが後方をスキャンしていない、データセット側の不具合**と特定。

CARLA生成側は別プロジェクト（`/home/ubuntu/carla_project/`）で管理されているため、ソースコード調査はそちら側に留め、症状のみを報告書にまとめた: `/home/ubuntu/carla_project/requests/DATASET_ISSUE_lidar_rear_coverage.md`。

---

## LiDAR 360°化データセットでの検証（完了）

状態: **完了**。データセットチームが LiDAR を 360° 視野に修正し、カメラ解像度も 1600x900 に統一した第4版データセットを同じ場所（`/mnt/nvme_scratch/nuscenes_carla`、3シーン22サンプル、instance 135件、annotation 891件）に提供。

### 検証結果

| 項目 | 修正前（第3版、後方0%） | 修正後（第4版） |
|---|---|---|
| カメラ解像度 | 混在（3840x2160 / 1280x720） | **全カメラ1600x900で統一済み**（前処理でのリサイズが実質no-op） |
| LiDAR点群の角度分布（1フレーム） | 前方のみ（後方180°が完全に0点） | **前方48.8%・後方51.2%とほぼ均等**（8方向ビン全てに1135〜1291点） |
| `valid_flag`有効の前後分布（val split） | 前方1665件・後方0件 | 前方56件・後方37件（両方に存在） |
| GT カメラ画像への box 描画 | 後方3カメラ空 | **後方3カメラ全てに box 描画を確認**（frame内で救急車・複数車両に正確に重なる） |
| FP32 検出数（閾値0.3、4サンプルのみ） | — | 38件（最大スコア0.90） |
| TorchNet 検出数 | — | 41件 |

LiDARセンサーの `horizontal_fov` 修正により、後方物体のアノテーションも学習・評価・GT可視化パイプラインに正しく載るようになったことを確認した。

生成動画（ホスト側 `bevformer_carla_videos/`）:
- `bevformer_carla_v4_groundtruth_scene0.mp4`
- `bevformer_carla_v4_fp32_scene0.mp4`
- `bevformer_carla_v4_torchnet_scene0.mp4`

### 未対応・次のステップ
- 今回のデータセットは3シーン22サンプルのみ（第3版の215サンプルより小規模）。密度と規模を両立したデータでの再評価が望ましい
- `steps=train` での CARLA データ再学習、実 nuScenes での同一 ONNX 比較は依然未着手

### 参照
- LiDAR後方カバレッジの不具合報告: `/home/ubuntu/carla_project/requests/DATASET_ISSUE_lidar_rear_coverage.md`
- 前処理・annotation生成: 前セクションと同一手順（`preprocess_carla_to_bevformer.py` → `create_data.py::nuscenes_data_prep`、3シーンなのでデフォルトの `scene-9001/9002/9003` 名でOK）

---

## 第5版データセットでの検証と長尺動画の生成（完了）

状態: **完了**。前セクション（第4版）は動画が2秒（4フレーム）と短すぎたため、データセットチームが再生成した第5版（3シーン・33サンプル、instance 124件、annotation 1221件、LiDAR 360°は維持）で、より長い動画を作成した。

### データセットの確認
- カメラ解像度: 全6カメラ1600x900で統一済み（変化なし）
- カメラ回転: 光学系変換込みで正常（変化なし）
- LiDAR点群の角度カバレッジ: 前方48.2%・後方51.8%、8方向ビン全てに1120〜1286点と均等（**360°化を維持**）
- シーン構成: scene-9001(22サンプル) / scene-9002(7サンプル) / scene-9003(4サンプル、val split)

### 長い動画を作るための工夫
`bevformer_inference.py` はデフォルトで val split（`nuscenes_infos_temporal_val.pkl`）を使うため、そのままだと scene-9003 の4サンプル（2秒動画）しか使えない。**train split（`nuscenes_infos_temporal_train.pkl`、scene-9001+9002 で29サンプル）を指すconfigコピー**（`ann_file` を `_val.pkl`→`_train.pkl` に書き換えた `bevformer_tiny_carla_v2_trainsplit.py`）を作成し、より長いシーン（scene-9001、22サンプル）を含む動画を生成した。

### 副次的に発見: データローダーのシーン順序が pkl の連続順と一致しない（SDK側、未解明）

`ground-truth`/`onnx`/`torchnet` いずれも、29サンプルを1本の動画にせず **4つの断片（6+2+1+20フレーム）に分割**して出力した。原因を調査:
- pkl自体（`nuscenes_infos_temporal_train.pkl`）は正しい——`scene_token` が index 0-21 で scene-9001、22-28 で scene-9002 と連続し、`prev`/`next` チェーンも綺麗に繋がっている。
- しかし実際にデータローダーを直接イテレートして `extract_scene_token` を記録すると、返される順序は `scene-9002 ×6 → scene-9001 ×2 → scene-9002 ×1 → scene-9001 ×20` という、pkl の連続順とは異なる並びだった。
- `bevformer_inference_impl.py` の `SceneFilter`（シーン境界検出ロジック）自体は単純な「直前と違うtokenなら新シーン」の実装で問題なし。原因は `build_dataloader_from_mmcv_config`（`shuffle=False`, `sampler=None`）より上流、データセットクラス側の内部並び替え（`CustomNuScenesDataset` 等の初期化時ソート）である可能性が高いが未確定。
- 実用対応として、**最長の連続区間（20フレーム = 10秒相当、scene-9001の大部分）を動画として使用**した。

この現象は CARLA データセット側の問題ではなく、Mythic SDK の推論パイプライン側の挙動（複数シーンを含む pkl を読む際の順序）である可能性が高い。次にSDK側を深く調査する場合の入り口として記録しておく。

### 結果

| 項目 | 値 |
|---|---|
| 使用区間 | scene-9001 の20フレーム連続区間（フォルダ `003-70a7c9a6...`） |
| FP32 検出数（閾値0.3、全29サンプル中） | 234件（最大スコア0.895） |
| TorchNet 検出数 | 241件（FP32とほぼ同数） |

生成動画（ホスト側 `bevformer_carla_videos/`、いずれも20フレーム=10秒相当）:
- `bevformer_carla_v5_groundtruth_scene1_20frames.mp4`
- `bevformer_carla_v5_fp32_scene1_20frames.mp4`
- `bevformer_carla_v5_torchnet_scene1_20frames.mp4`

### 未対応・次のステップ
- データローダーのシーン順並び替えの根本原因調査（`CustomNuScenesDataset`初期化時のソート処理等）
- `steps=train` での CARLA データ再学習、実 nuScenes での同一 ONNX 比較は依然未着手

### 参照
- train split 指定 config: `bevformer_lib/projects/configs/bevformer/bevformer_tiny_carla_v2_trainsplit.py`（`ann_file` を train.pkl に変更）
- シーン境界検出: `bevformer_lib/custom_utils/inference.py::SceneFilter`, `::extract_scene_token`（`data_loading.py`）

---

## 第6版データセットでの検証（86サンプル、単一シーン動画）（完了）

状態: **完了**。データセットチームがさらに規模を拡大した第6版（**2シーン・171サンプル、1シーンあたり85-86サンプル**、instance 101件、annotation 7442件、LiDAR 360°・カメラ解像度1600x900は維持）を同じ場所に提供。これまでで最大規模かつ、1シーンあたりのサンプル数も最多。

### データセットの確認（すべて健全）
- カメラ解像度: 全6カメラ1600x900（変化なし）
- カメラ回転: 光学系変換込みで正常（変化なし）
- LiDAR点群の角度カバレッジ: 前方49.3%・後方50.7%、8方向ビン全てに1133〜1315点と均等（**360°化を維持**）
- orphan annotation: 0件

### train/val split の割り当て
2シーンのみのため、`scene-9001`(85サンプル)→train, `scene-9002`(86サンプル)→val とした（`sitecustomize.py`のデフォルトは3シーン想定の `scene-9001,9002→train / scene-9003→val` だが、今回は該当パッチをその場で書き換えて対応）。

### 結果: 分割問題が発生せず、単一シーン全86フレームが1本の動画に

前セクション（第5版）で発見した「データローダーがシーン順を並べ替えて動画が断片化する」現象は、**val split が単一シーンのみ（scene-9002、86サンプル）だったため今回は再発しなかった**。GT・FP32・TorchNet いずれも `000-39288299...` の1フォルダに86フレーム全てが収まった。

| 項目 | 第5版（20フレーム、複数シーンpklの断片） | 第6版（86フレーム、単一シーン全体） |
|---|---|---|
| 動画フレーム数 | 20（10秒相当） | **86（43秒相当）** |
| FP32 検出数（閾値0.3） | 234件 | **821件**（最大スコア0.92） |
| TorchNet 検出数 | 241件 | **856件** |

GT可視化を目視確認（frame 40付近）：消防車・トラック・車など複数物体に GT box が正確に重なり、前後カメラともに描画されている。

生成動画（ホスト側 `bevformer_carla_videos/`、86フレーム=43秒相当）:
- `bevformer_carla_v6_groundtruth_scene0_86frames.mp4`
- `bevformer_carla_v6_fp32_scene0_86frames.mp4`
- `bevformer_carla_v6_torchnet_scene0_86frames.mp4`

### 未対応・次のステップ
- train split（scene-9001, 85サンプル）での推論は未実施
- 前セクションで見つかったデータローダーのシーン順並び替え問題は、複数シーンを含む split を使う場合に再発する可能性がある（根本原因は依然未解明）
- `steps=train` での CARLA データ再学習、実 nuScenes での同一 ONNX 比較は依然未着手

### 参照
- 前処理コマンド: `preprocess_carla_to_bevformer.py --scene-names scene-9001,scene-9002`（2シーン対応、デフォルトの3シーン名リストを上書き）

---

## BEVパネルの座標系不整合を修正（完了、2段階）

状態: **完了**。「自車マーカーの矢印は上を向くのに、物体(box)が前進すると右方向に動いて見える」という指摘を受け、SDK 側の可視化コード `bevformer_lib/custom_utils/visualization.py` にパッチを適用した。**1回目のパッチでは前後(forward/backward)は直ったが左右(left/right)が反転したままで、2回目のパッチで左右も修正した。**

### 修正方針（最終版）
`_layer_ego`（自車マーカー）はそのまま「正」とし、`_layer_boxes`/`_layer_lidar`/`_layer_radar` の `ix, iy` 計算式を以下に統一：
```
ix = (x - x_min) / xr * size_px   # 前方(+x) が大きいix → 最終画像で上
iy = (y - y_min) / yr * size_px   # 左(+y) が大きいiy → 最終画像で左
```
`_layer_map_polylines`/`_layer_map_raster`（`_shapely_to_bev_pts`）は別の反転処理（`cv2.flip`）を経由する独立した仕組みのため、今回のスコープからは除外（マップ機能自体は今回のBEV動画では未使用）。

### 1回目のパッチ（前後のみ修正、左右バグが残存）
`ix = (x-x_min)/xr*size_px`, `iy = (y_max-y)/yr*size_px` に変更——前方が上に来ることは実現したが、`iy`の`(y_max - y)`という式のせいで**左右が反転**していた（世界座標で自車の左(+y)にある物体が最終画像で右側に描かれる）ことをユーザーが指摘。数値シミュレーションで検証し、`iy`の式を`(y_max - y)`から`(y - y_min)`に修正することで、前後・左右の両方が同時に正しくなることを確認（4方向テストで forward→UP, backward→DOWN, left→LEFT, right→RIGHT を確認）。

### 変更箇所（`_extracted_sdk/bevformer_lib/custom_utils/visualization.py`、gitで差分追跡）
- `_layer_lidar`: `ix, iy` の式を交換 → `iy`をさらに`(y-y_min)`に修正
- `_layer_radar`: `ix, iy` の式を交換 → `iy`をさらに`(y-y_min)`に修正
- `_layer_boxes`: box中心の `ix, iy` および4隅 `pts_img` の計算式を交換 → `iy`関連をさらに`(y-y_min)`基準に修正

計3関数・最終的に7行の変更（`git diff` で確認可能）。コンテナ内の実ファイルにも同一パッチを適用済み。

### 検証
1. **シミュレーションで事前確認**: 前方移動・後方移動・左移動・右移動の4パターンをテストし、最終BEV画像でそれぞれ上/下/左/右に対応することを数値で確認。
2. **実データで再生成**: 第6版データセット（86サンプル単一シーン）で GT/FP32/TorchNet を再実行。
   - 検出結果自体（件数・スコア）は不変（FP32: 821件、TorchNet: 827件——パッチ前と同水準、可視化のみの変更のため想定通り）
   - GT可視化のBEVパネルで box の配置が自車マーカーの前後・左右と整合することを目視確認

生成動画（ホスト側 `bevformer_carla_videos/`、最終修正版）:
- `bevformer_carla_v6_groundtruth_scene0_86frames_fixed.mp4`
- `bevformer_carla_v6_fp32_scene0_86frames_fixed.mp4`
- `bevformer_carla_v6_torchnet_scene0_86frames_fixed.mp4`

（1回目のパッチ版動画は左右反転が残っていたため削除済み）

### 参照
- パッチ差分: `_extracted_sdk/bevformer_lib/custom_utils/visualization.py`（ホスト側リポジトリで `git diff` 追跡）
- コンテナ内の反映先: `/root/mythic_sdk/v26.05.0/mythic-model-zoo/mythic/model_zoo/bevformer/bevformer_lib/custom_utils/visualization.py`（コンテナ再作成時は再度 `docker cp` が必要、`--rm` の使い捨てコンテナのため）
