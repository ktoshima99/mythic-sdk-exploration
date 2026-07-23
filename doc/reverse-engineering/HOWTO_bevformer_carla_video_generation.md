# 手順書: CARLA データセットから BEVFormer 推論動画（GT / FP32 / TorchNet）を作るまで

本ドキュメントは、`/mnt/nvme_scratch/nuscenes_carla` に置かれた CARLA 生成データセットから、BEVFormer-tiny の Ground-Truth 可視化・FP32(ONNX Runtime)推論・TorchNet(アナログaware)推論の3本の動画を作成するまでの一連の手順をまとめたもの。背景・発見した不具合の詳細は [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) を参照。

前提として `/mnt/nvme_scratch/nuscenes_carla` に置かれるデータは都度更新される（データセットチームが新版を都度同じ場所に上書き配置する運用）。**ディレクトリが再作成された場合、既存の Docker バインドマウントは古い inode を掴んだまま更新内容を反映しない**ため、毎回コンテナを再作成する必要がある。

---

## 0. 前提: 更新されたか確認する

```bash
stat /mnt/nvme_scratch/nuscenes_carla
docker inspect mythic_sdk_impl --format '{{.Created}}'
```
`stat` の `Birth` がコンテナの `Created` より新しければ、ディレクトリが再作成されている。この場合は次のステップでコンテナを作り直す。

---

## 1. コンテナの起動（GPU + 各種マウント）

`docker run --mount src=...` は絶対パスが必要なため、以下は**リポジトリルートで実行し、`$(pwd)` でリポジトリルートの絶対パスに展開する**:

```bash
cd <repo-root>
docker stop mythic_sdk_impl 2>/dev/null
docker rm mythic_sdk_impl 2>/dev/null

docker run -d --name mythic_sdk_impl \
  --gpus all --shm-size 512m \
  --mount type=bind,src=/tmp/carla_scratch_v2,dst=/workspace/carla_scratch \
  --mount type=bind,src=/tmp/carla_output_v2,dst=/workspace/carla_output \
  --mount type=bind,src="$(pwd)/mythic_sdk/training-models-installer-m2000-v26.05.0/archive/models/training/bevformer",dst=/workspace/bevformer_models \
  --mount type=bind,src=/mnt/nvme_scratch/nuscenes_carla,dst=/workspace/nuscenes_carla_v2,readonly \
  --mount type=bind,src="$(pwd)/tools/sitecustomize.py",dst=/workspace/patches/sitecustomize.py \
  --mount type=bind,src="$(pwd)/tools/preprocess_carla_to_bevformer.py",dst=/workspace/preprocess_carla_to_bevformer.py \
  --mount type=bind,src="$(pwd)/tools/bevformer_tiny_carla.py",dst=/workspace/bevformer_tiny_carla.py \
  gcr.io/mythic-devops/mythic-sdk-ubuntu-24.04:m2000-v26.05.0 \
  sleep 14400
```

`--rm` は付けず `sleep 14400`（4時間）で自動終了するようにしている。**4時間経過すると自動的に停止するため、長時間作業する場合は事前に気づいて再作成すること**（`docker ps -a` で `Exited` を確認）。

GPU とマウント内容の確認:
```bash
docker exec mythic_sdk_impl bash -c '
  /root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 -c "import torch; print(torch.cuda.is_available())"
  ls /workspace/nuscenes_carla_v2/v1.0-trainval/
'
```

コンテナ内の BEVFormer config ディレクトリに `bevformer_tiny_carla.py` を配置し直す（コンテナ再作成で消えるため毎回必要）:
```bash
docker exec mythic_sdk_impl bash -c '
  cp /workspace/bevformer_tiny_carla.py \
     /root/mythic_sdk/v26.05.0/mythic-model-zoo/mythic/model_zoo/bevformer/bevformer_lib/projects/configs/bevformer/bevformer_tiny_carla.py
'
```

**可視化コードのバグ修正パッチも毎回コピーが必要**（後述「6. 可視化バグの既知の修正」参照、`--rm`ではなくとも `docker run` で作った新コンテナには反映されていないため）。

---

## 2. データセットの事前チェック（推奨）

前処理前に、シーン数・カメラ解像度・カメラ回転・LiDAR カバレッジを確認しておくと、既知の不具合（カメラ外部パラメータ・LiDAR後方カバレッジ欠落）が再発していないか早期に判断できる。

```bash
docker exec mythic_sdk_impl bash -c "
/root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 -c \"
import json, numpy as np
base = '/workspace/nuscenes_carla_v2/v1.0-trainval'
scenes = json.load(open(base+'/scene.json'))
print('scenes:', len(scenes), [(s['name'], s['nbr_samples']) for s in scenes])

sd = json.load(open(base+'/sample_data.json'))
sensors = json.load(open(base+'/sensor.json'))
cs = json.load(open(base+'/calibrated_sensor.json'))
sen = {s['token']: s for s in sensors}
cs_by_tok = {c['token']: c for c in cs}
seen = {}
for rec in sd:
    ch = sen[cs_by_tok[rec['calibrated_sensor_token']]['sensor_token']]['channel']
    if 'CAM' in ch and ch not in seen:
        seen[ch] = 1
        print(ch, rec['width'], 'x', rec['height'])

lidar_sd = [x for x in sd if 'LIDAR' in x['filename'] and x.get('is_key_frame')]
pc = np.fromfile('/workspace/nuscenes_carla_v2/'+lidar_sd[0]['filename'], dtype=np.float32).reshape(-1,5)
x = pc[:,0]
print('LiDAR frac x>0 (front):', round((x>0).mean(),3), ' frac x<0 (back):', round((x<0).mean(),3))
\"
"
```
- LiDAR の前後比が概ね50/50であれば360°カバレッジは健全。片側が0に近ければ [DATASET_ISSUE_lidar_rear_coverage.md 相当の不具合](FUTURE_bevformer_inference_run.md) が再発している。
- カメラ回転の光学系変換チェックは [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) の該当セクション参照（`quat_to_R` で回転行列を出し `R_row3` を確認）。

---

## 3. 前処理（CARLA形式 → Mythic SDK 向け nuScenes 形式）

シーン数がデフォルトの3つでない場合は `--scene-names` で明示する（例: 2シーンなら `scene-9001,scene-9002`、4シーンなら `scene-9001,scene-9002,scene-9003,scene-9004`）。

```bash
docker exec mythic_sdk_impl bash -c '
  rm -rf /workspace/carla_output_v2
  mkdir -p /workspace/carla_output_v2
  /root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 \
    /workspace/preprocess_carla_to_bevformer.py \
    --scratch-dir /workspace/nuscenes_carla_v2 \
    --output-root /workspace/carla_output_v2 \
    --force --workers 8
'
```
これは `CAM_FRONT_NARROW`→`CAM_FRONT` のリネーム、`CAM_FRONT_WIDE` の除外、カメラ画像の1600x900統一リサイズ、intrinsics再計算、マップマスク生成、CAN busファイルのコピーを行う（詳細は [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) の「CARLA データセットの前処理と annotation 生成」参照）。既に1600x900の場合、リサイズは実質 no-op になる。

---

## 4. train/val split の割り当てと annotation 生成

`v1.0-test` は生成コードが要求するのでスタブとして `v1.0-trainval` をコピーする。scene 数に応じて train/val に割り振るシーン名を決める（1シーンだけを val にすると、動画を作るシーンの選択がシンプルになる）。

```bash
docker exec mythic_sdk_impl bash -c '
  cp -r /workspace/carla_output_v2/v1.0-trainval /workspace/carla_output_v2/v1.0-test
  cd /root/mythic_sdk/v26.05.0/mythic-model-zoo
  /root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 - <<PY
import sys
sys.path.insert(0, "mythic/model_zoo/bevformer")
from nuscenes.utils import splits as _nusc_splits

# シーン数に応じて調整（例: 2シーンなら片方をval, もう片方をtrainに）
for name in ["scene-9001"]:
    if name not in _nusc_splits.train:
        _nusc_splits.train.append(name)
for name in ["scene-9002"]:
    if name not in _nusc_splits.val:
        _nusc_splits.val.append(name)

from bevformer_lib.tools.create_data import nuscenes_data_prep

nuscenes_data_prep(
    root_path="/workspace/carla_output_v2",
    can_bus_root_path="/workspace/carla_output_v2",
    info_prefix="nuscenes",
    version="v1.0-trainval",
    dataset_name="NuScenesDataset",
    out_dir="/workspace/annotations_v2",
    max_sweeps=10,
)
PY
'
```
出力: `/workspace/annotations_v2/nuscenes_infos_temporal_{train,val}.pkl`。

**注意（既知の制約）**: データローダーは pkl 内のシーン順を必ずしも維持しない（複数シーンを含む split を読むと、シーン単位で断片化されることがある。原因未解明、[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) 参照）。**val split を単一シーンにしておくと、この問題を回避でき、そのシーンの全サンプルが1本の動画に確実に収まる。**

---

## 5. 推論用 config の準備

`bevformer_tiny_carla.py` はデフォルトで `/workspace/carla_output/`（旧パス）を指しているため、`_v2` パスを指す config コピーを作る。

```bash
docker exec mythic_sdk_impl bash -c '
CFGDIR=/root/mythic_sdk/v26.05.0/mythic-model-zoo/mythic/model_zoo/bevformer/bevformer_lib/projects/configs/bevformer
sed -e "s#/workspace/carla_output/#/workspace/carla_output_v2/#" \
    -e "s#/workspace/annotations/#/workspace/annotations_v2/#" \
    $CFGDIR/bevformer_tiny_carla.py > $CFGDIR/bevformer_tiny_carla_v2.py
'
```

---

## 6. 可視化バグの既知の修正（毎回のコンテナ再作成後に必要）

`bevformer_lib/custom_utils/visualization.py` の BEV パネル描画には、自車マーカー(`_layer_ego`)と物体位置(`_layer_boxes`/`_layer_lidar`/`_layer_radar`)の座標変換規則が食い違うバグがあった（前方が右に見える、さらに左右も反転する）。**修正パッチはホスト側リポジトリの `mythic_sdk/_extracted_sdk/bevformer_lib/custom_utils/visualization.py` で git 管理されている。**コンテナを再作成すると SDK 側のファイルは元の状態に戻るため、毎回このパッチを再適用する必要がある。

```bash
cd <repo-root>
docker cp mythic_sdk/_extracted_sdk/bevformer_lib/custom_utils/visualization.py \
  mythic_sdk_impl:/root/mythic_sdk/v26.05.0/mythic-model-zoo/mythic/model_zoo/bevformer/bevformer_lib/custom_utils/visualization.py
```

修正内容の詳細（`ix,iy` の座標変換式を `_layer_ego` の規約に統一）は [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) の「BEVパネルの座標系不整合を修正」を参照。`git diff mythic_sdk/_extracted_sdk/bevformer_lib/custom_utils/visualization.py` で変更点を確認できる。

---

## 7. GT / FP32 / TorchNet の実行

3本とも `--data-type=samples`（2Hzキーフレーム）で、val split（単一シーンにしておいた方）を対象に実行する。86サンプル程度ならCPU推論（FP32/GT）で1〜2分、TorchNet(GPU)で2〜3分程度。

### Ground Truth 可視化
```bash
docker exec mythic_sdk_impl bash -c '
cd /root/mythic_sdk/v26.05.0/mythic-model-zoo/mythic/model_zoo/bevformer
export PYTHONPATH=/workspace/patches
CFG=bevformer_lib/projects/configs/bevformer/bevformer_tiny_carla_v2.py
rm -rf /workspace/gt_out
/root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 bevformer_inference.py ground-truth \
  "$CFG" --output-dir /workspace/gt_out --data-type=samples
'
```

### FP32（ONNX Runtime, CPU）
```bash
docker exec mythic_sdk_impl bash -c '
cd /root/mythic_sdk/v26.05.0/mythic-model-zoo/mythic/model_zoo/bevformer
export PYTHONPATH=/workspace/patches
CFG=bevformer_lib/projects/configs/bevformer/bevformer_tiny_carla_v2.py
rm -rf /workspace/inference_out_fp32
/root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 bevformer_inference.py onnx \
  /workspace/bevformer_models/bevformer-tiny-fp32-1600x900.onnx \
  "$CFG" --output-dir /workspace/inference_out_fp32 --data-type=samples --save-json
'
```
サンプル数が数十〜100程度だとバックグラウンド実行（`docker exec -d`）＋ログファイルへのリダイレクトを推奨（フォアグラウンドのタイムアウトに引っかかりやすい）:
```bash
docker exec -d mythic_sdk_impl bash -c '... > /workspace/fp32.log 2>&1; echo DONE >> /workspace/fp32.log'
# 完了はポーリングで確認: grep -c DONE /workspace/fp32.log
```

### TorchNet（アナログaware, GPU）
```bash
docker exec mythic_sdk_impl bash -c '
cd /root/mythic_sdk/v26.05.0/mythic-model-zoo/mythic/model_zoo/bevformer
export PYTHONPATH=/workspace/patches
CFG=bevformer_lib/projects/configs/bevformer/bevformer_tiny_carla_v2.py
rm -rf /workspace/inference_out_torchnet
/root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 bevformer_inference.py torchnet \
  /workspace/bevformer_models/bevformer-tiny-1600x900-trained.onnx \
  "$CFG" --output-dir /workspace/inference_out_torchnet --data-type=samples --save-json
'
```

`onnxruntime` は CPU版のみ同梱（GPU化不可）なので FP32 は常にCPU実行、TorchNet は `--device` 既定が `cuda:0` になり自動でGPU実行される。

---

## 8. 検出結果の確認（任意）

```bash
docker exec mythic_sdk_impl bash -c '
/root/mythic_sdk/v26.05.0/mythic-model-zoo/venv/bin/python3 -c "
import json
d = json.load(open(\"/workspace/inference_out_fp32/results.json\"))
r = d.get(\"results\", d)
print(\"detections:\", sum(len(v) for v in r.values()), \"across\", len(r), \"samples\")
"
'
```

---

## 9. 動画をホストへコピー

出力ディレクトリ名は `<index>-<scene_token>` の形式（例: `000-39288299cd7148999bfad146cde512cb`）。単一シーンの val split なら通常1フォルダに全フレームが収まる。

```bash
mkdir -p output/bevformer_carla_videos
FOLDER=$(docker exec mythic_sdk_impl bash -c 'ls /workspace/gt_out/' | head -1)

docker cp mythic_sdk_impl:/workspace/gt_out/$FOLDER/scene.mp4 \
  output/bevformer_carla_videos/bevformer_carla_groundtruth.mp4
docker cp mythic_sdk_impl:/workspace/inference_out_fp32/$FOLDER/scene.mp4 \
  output/bevformer_carla_videos/bevformer_carla_fp32.mp4
docker cp mythic_sdk_impl:/workspace/inference_out_torchnet/$FOLDER/scene.mp4 \
  output/bevformer_carla_videos/bevformer_carla_torchnet.mp4
```

動画は `mp4v` コーデックのため Ubuntu 標準プレイヤーで再生できないことがある。VLC (`sudo apt install vlc`) を推奨。

---

## チェックリスト（毎回の再実行時）

- [ ] `/mnt/nvme_scratch/nuscenes_carla` の birth time がコンテナ作成時刻より新しいか確認
- [ ] コンテナを `--gpus all --shm-size 512m` で再作成し、全マウントを再指定
- [ ] `bevformer_tiny_carla.py` をコンテナ内 configs ディレクトリにコピー
- [ ] **可視化バグ修正パッチ（`visualization.py`）を `docker cp` で再適用**
- [ ] データセットのシーン数を確認し、`--scene-names` とtrain/val割り当てを調整
- [ ] 前処理 → annotation生成 → config作成 → GT/FP32/TorchNet実行 → 結果確認 → 動画コピー

## 参照
- 背景・発見した不具合の詳細な経緯: [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)
- CARLA データセット側への修正要望書: [DATASET_ISSUE_carla_camera_extrinsics.md](DATASET_ISSUE_carla_camera_extrinsics.md)、`/home/ubuntu/carla_project/requests/DATASET_ISSUE_lidar_rear_coverage.md`
- 可視化パッチのソース: `_extracted_sdk/bevformer_lib/custom_utils/visualization.py`（git管理）
