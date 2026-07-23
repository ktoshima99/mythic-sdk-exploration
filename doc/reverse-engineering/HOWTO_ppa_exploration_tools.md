# 手順書: BEVFormer PPA(面積-電力トレードオフ)探索に使うツール群

**[訂正版]** 前版は内部実装(Python API)を深く追いすぎて、同じ製品ツールの内部呼び出しを別々のツールとして数えてしまっていた(6項目)。`doc/user-guides/GEN2 User Guide.pdf`(SDK v26.05.0付属、公式ユーザー向けドキュメント)を確認した結果、ユーザーが実際に叩くエントリーポイントは**3つ**であることが確定した。本版はこの3ツールを基準に構成する。

| # | ツール | コマンド | 前版での対応(誤って分割していた項目) |
|---|---|---|---|
| 1 | **Compiler**(`mythic-compiler`) | `mythic-compiler --input-artifact <artifact> --compiler-config <yaml> --output-artifact <out>.tar.gz` | 旧§1(`QuantizationConfig`)・§2(`explore_model`)・§3(`run_compilation`/`compile.py`)は全部この1つのCLIの内部実装 |
| 2 | **PPA Estimator**(`mythic-ppa-estimators`) | `mythic-ppa-estimators --estimate-performance --estimate-power --power-inference-rate <fps> <model_artifact>.tar.gz` | 旧§4(`perf_analysis.py`)・§5(`power_estimator.py`)はこの1つのCLIに統合されている |
| 3 | **Accuracy Simulation**(`convert_model.py steps=eval_trained`) | `DATASET=<dir> python3 scripts/common/convert_model.py steps=eval_trained trained_model=<...>.onnx` | 旧§6`bevformer_inference.py`は動画生成用の補助ツールで、正式なaccuracy simulation経路ではない(旧§1.2で見つけていた`eval_trained`が正式) |

計画([PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md))のStage対応: Stage 0/A=ツール1(コンパイル)+ツール3(精度)、Stage B/C=ツール1(num_aces/n_mps違いでの再コンパイル)+ツール2(PPA計測)。

**環境について**: GEN2 User Guideの§4-8はCadence Collaboration Chamber(SLURM経由)を前提に書かれているが、**本探索はこのリポジトリのあるAWS環境上でdockerを直接操作する**(Collaboration Chamberは使わない)。§0でこの差分を整理する。

---

## 0. 実行環境の前提(このAWS環境固有 — GEN2 User Guideの記載からの差分)

**[本探索の前提]** GEN2 User Guideの§4-8はCadence Collaboration Chamber(SLURM経由でのノード確保、`/projects/tonbomythic3`配下のアクセス権管理)を前提に書かれているが、**本探索はCollaboration Chamberを使わず、このAWS環境上のdockerを直接操作する**。したがって以下は本探索では**不要**:
- Cadence Collaboration Chamberへのログイン、`tonbomythic3`/`docker`グループ確認
- SLURMセッション起動(`sbatch -p aw71-interactive`/`aw71-gpu-g6e`)——AWS上でdockerはSLURM無しで直接使える
- `/projects/tonbomythic3`配下のパス。付属`.sh`スクリプトの`DATASET_DIR`等の既定値がこのパスを指しているが、AWS環境では環境変数で上書きする(§0.2参照)

**この環境で既に確認済みの実態**(§0.1):

### 0.1 導入済みの状態(確認済み)

- SDK配布物一式(installer zip/展開ディレクトリ、起動スクリプト、ベンダー提供ドキュメント)は`mythic_sdk/`フォルダにまとめてある(リポジトリ独自の分析物と区別するための構成——詳細はリポジトリルートの`README.md`参照)。起動スクリプトは`mythic_sdk/run_mythic_sdk_container.sh`・`mythic_sdk/gpu_run_mythic_sdk_container.sh`・`mythic_sdk/load_and_tag_docker_images.sh`。SDK配布zipの展開先も`mythic_sdk/`直下(`mythic_sdk/archive/`、`mythic_sdk/training-models-installer-m2000-v26.05.0/`等)。
- Dockerイメージは`docker images`で確認済み、両方とも導入されている:
  ```
  gcr.io/mythic-devops/mythic-sdk-ubuntu-24.04:m2000-v26.05.0
  gcr.io/mythic-devops/compilerd-bin:1.5.2
  ```
  未導入の環境を新たに用意する場合は、`mythic_sdk/archive/`ディレクトリから`./install_compiler.sh`・`./install_sdk_docker_image.sh`を実行する(またはヘルパー`mythic_sdk/load_and_tag_docker_images.sh`)。

### 0.2 SDKコンテナの起動(AWS版の実際のパス上書き)

**GEN2 User Guide/付属スクリプトの既定値はCollaboration Chamber向け**(`DATASET_DIR`の既定は`/projects/tonbomythic3/datasets`、`TRAINING_MODELS_HOST_DIR`の既定はそのCollaboration Chamber固有パス)なので、**このAWS環境では環境変数で上書きして起動する**。`run_mythic_sdk_container.sh`/`gpu_run_mythic_sdk_container.sh`のスクリプト本体を確認した結果、上書き可能な変数は:

| 変数 | 既定(Collaboration Chamber向け) | 本環境での指定例 |
|---|---|---|
| `MYTHIC_WORKSPACE` | `$HOME` | コンパイル結果・PPA推定結果の永続化先。既定の`$HOME`のままでも動く |
| `DATASET_DIR` | `/projects/tonbomythic3/datasets` | 本物のnuScenesデータセット配置先(例: `/mnt/nvme_scratch/nuscenes` 等、実際の配置場所に置き換える) |
| `TRAINING_MODELS_HOST_DIR` | Collaboration Chamber固有パス | `mythic_sdk/training-models-installer-m2000-v26.05.0/archive/models/training`(既存のzip展開先) |

スクリプトは`mythic_sdk/`ディレクトリの中にあるため、`mythic_sdk/`に`cd`してから実行する(相対パスの`archive/SDK-VERSION`解決に必要)。

Compiler/PPA Estimator用(GPU不要):
```bash
cd <repo-root>/mythic_sdk
DATASET_DIR=<本物のnuScenesを置いたホストパス> \
TRAINING_MODELS_HOST_DIR=training-models-installer-m2000-v26.05.0/archive/models/training \
./run_mythic_sdk_container.sh
root@<container-id>:~/mythic_sdk/v26.05.0#
```

Accuracy Simulation用(GPU必須、nvidiaランタイム経由):
```bash
cd <repo-root>/mythic_sdk
DATASET_DIR=<本物のnuScenesを置いたホストパス> \
TRAINING_MODELS_HOST_DIR=training-models-installer-m2000-v26.05.0/archive/models/training \
./gpu_run_mythic_sdk_container.sh
root@<container-id>:~/mythic_sdk/v26.05.0#
```
(`gpu_run_mythic_sdk_container.sh`は`docker run --gpus all`相当を内部で実行する。[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)で確認済みの通り、ホストにNVIDIA GPU + Docker `nvidia`ランタイムが必要。)

**どちらのコンテナでも起動直後にPythonの仮想環境を有効化する**(`MYTHIC_SDK_ROOT`はイメージにビルド時設定済みの環境変数、確認済み: `/root/mythic_sdk/v26.05.0`):
```bash
source $MYTHIC_SDK_ROOT/mythic-model-zoo/venv/bin/activate
```
以降、本ドキュメントの`mythic-compiler`/`mythic-ppa-estimators`/`convert_model.py`の各コマンドは、**このvenv有効化後のコンテナシェル内で実行する**ことを前提とする。

### 0.3 本物のnuScenesデータセット + CAN busデータ + annotationの配置(本探索の前提)

**[本探索の前提]** Collaboration Chamber固有のannotation生成権限問題(GEN2 User Guide §8.3.5に明記)は本探索では回避する方針——**本物のnuScenesデータセット・CAN bus拡張データ・annotationを別途用意し、このAWS環境からアクセスできる場所に配置する**。配置先はコンテナ起動時に`DATASET_DIR`環境変数(§0.2)で指定し、コンテナ内`$MYTHIC_SDK_ROOT/mythic-model-zoo/datasets`にマウントされる。

annotation生成(`nuscenes_infos_temporal_{train,val}.pkl`)自体は、この環境のroot権限下で行うディレクトリ権限の制約を受けないと想定されるが、**実際に`steps=generate_nuscenes_annotations`(GEN2 User Guideには明示されていないが、[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)で確認済みのステップ名)を1回通してannotationが生成できることを確認するのが、Accuracy Simulation着手前の最初の検証項目**。

---

## 1. Compiler: `mythic-compiler`(Stage 0 / Stage A / Stage B)

**実行環境**: §0.2の`run_mythic_sdk_container.sh`で起動したコンテナ内(venv有効化後)。GPUは不要。

コンパイル済みfirmware artifactを生成する。**量子化ポリシー・タイル分割・マルチコア割当・アナログ/デジタル振り分けは全てここで一度に決まる**(前版で個別のPython API`QuantizationConfig`/`explore_model`/`run_compilation`として分けていたものは、この1コマンドの内部処理)。

```bash
mythic-compiler --input-artifact <compiler_ready_artifact> \
    --compiler-config <model-compiler-configuration>.yaml \
    --output-artifact /rscratch/tonbomythic3/<your_username>/<model_name>.tar.gz
```

### BEVFormer-tinyの実例(GEN2 User Guide §7.3)

| 項目 | パス |
|---|---|
| コンパイル対象artifact | `$MYTHIC_SDK_ROOT/models/training/bevformer/bevformer-tiny-1600x900-trained.tar.gz` |
| Compiler config(default) | `$MYTHIC_SDK_ROOT/mythic-model-zoo/configs/bevformer/compiler/bevformer_tiny_backbone_6x928x1600x3_m2072_default_optimization.yaml` |
| Compiler config(high) | `$MYTHIC_SDK_ROOT/mythic-model-zoo/configs/bevformer/compiler/bevformer_tiny_backbone_6x928x1600x3_m2072_high_optimization.yaml` |

```bash
mythic-compiler --input-artifact $MYTHIC_SDK_ROOT/models/training/bevformer/bevformer-tiny-1600x900-trained.tar.gz \
    --compiler-config $MYTHIC_SDK_ROOT/mythic-model-zoo/configs/bevformer/compiler/bevformer_tiny_backbone_6x928x1600x3_m2072_high_optimization.yaml \
    --output-artifact /rscratch/tonbomythic3/<your_username>/bevformer_tiny_compiled.tar.gz
```

出力artifact(tar.gz)には`artifacts/firmware/weight_utilization.txt`・`sram_utilization.txt`(重み/SRAM使用率レポート)、`artifacts/firmware/perf_trace_dump.h5`、`artifacts/firmware/vnn/Mythic<Model>`(digital NPU JSON)などが含まれる(§2でこのartifactをそのまま使う)。

### Stage A: 量子化ポリシーを振る

計画の量子化ポリシー(§1のQuantizationConfig.tensor_n_bits、8/16bitのみ許可)を試すには、**`--compiler-config`に渡すYAMLを複製し、量子化関連の設定項目を書き換える**。既定/高最適化の2種類が用意されているモデルもある(`_default_optimization.yaml` / `_high_optimization.yaml`)ことから、YAML内には最適化レベルの選択肢が存在することが分かる——**具体的にどのYAMLキーが`tensor_n_bits`相当の量子化ビット幅指定に対応するかは、実際に`bevformer_tiny_backbone_..._default_optimization.yaml`を開いて確認する必要がある[未検証]**(このリポジトリの抽出済みファイルには含まれておらず、実機のSDKコンテナ内`$MYTHIC_SDK_ROOT/mythic-model-zoo/configs/`で確認する)。

### Stage B: `num_aces`/`n_mps`を振る

GEN2 User Guideの例では、YAML名に`_default_optimization`/`_high_optimization`という最適化レベルの区別はあるが、`num_aces`/`n_mps`を直接変える例は示されていない。前回の調査で分かった`--relative-objective-target`等のフラグ、および`num_aces`(何タイル使うか)の指定方法は、**このYAML内、または`--compiler-config`と併用する別フラグとして存在すると推定される[未検証]**。実装開始時に、実際に`--compiler-config`のYAMLファイルを開いて中身を確認するのが最優先の調査項目。

---

## 2. PPA Estimator: `mythic-ppa-estimators`(Stage B / Stage C)

**実行環境**: §1と同じコンテナ(`run_mythic_sdk_container.sh`)内。コンテナを維持していれば§1の直後にそのまま続けて実行できる。

**§1でコンパイルしたartifact(tar.gz)をそのまま渡すだけ**でlatency/power両方を推定する。前版で別々のCLIとして扱っていた`perf_analysis.py`(`--hdf5-path`)と`power_estimator.py`(`--l0-pb-path`)は、実際には**このコマンドの内部で自動的に両方読まれる**(artifact内の`perf_trace_dump.h5`と、firmware内のL0 protobufを両方開いている、実行ログで確認済み)。

```
mythic-ppa-estimators --help
usage: mythic-ppa-estimators [-h] [--estimate-performance] [--estimate-power]
                              [--power-inference-rate POWER_INF_RATE] model_artifact_path

positional arguments:
  model_artifact_path   Path to Mythic model artifact archive (tar.gz)

options:
  -h, --help            show this help message and exit
  --estimate-performance
                        Estimate performance for model artifact
  --estimate-power      Estimate power for model artifact
  --power-inference-rate POWER_INF_RATE
                        Set a target inference rate in frames/second (fps) for power estimation
```

実行例(ResNet-50、最大推論速度時、GEN2 User Guide §7.4):
```bash
mythic-ppa-estimators --estimate-performance --estimate-power resnet50_compiled.tar.gz
```

### 出力(実際のログ、要点)

```
Analog NPU Total Estimated Processing Time for Compiled ONNX File: 0.44 ms
Analog NPU Estimated Frame Rate for Compiled ONNX File: 2,274.40 fps
...
Total Executed ACE Operations: 47,236
Total Executed ACE MACs: 5,324,275,712
Maximum Theoretical ACE Execution Time (With No Parallelization Using 1 ACE): 7.56 ms
Minimum Theoretical ACE Execution Time (With Even Parallelization Across 24 ACEs): 0.31 ms
ACE Utilization (Minimum Theoretical Time/Estimated Processing Time): 71.62%
Total SRAM Bytes Read Across Chip: 140,795,352 bytes
Total SRAM Bytes Written Across Chip: 96,647,544 bytes
Estimated Die Area to Achieve Estimated Processing Time: 253.09 mm^2
...
Digital NPU Performance Metrics (from separate ONNX graph processing)
  Total Estimated MACs Targeting Digital NPU: 2,000,000
  Digital NPU MAC Utilization: 30.00%
  ...
Combined Analog + Digital NPU Latency: 0.55 ms
Combined Analog + Digital NPU Total Estimated Frame Rate: 1,830.36 fps (frames per second)
...
Process Nodes: Analog 28nm, Digital 5nm
Number of ACEs: 24
Number of Digital NPU Cores: 1
Inference Rate: 1830 frames/second (fps)

Analog NPU Estimated Power: 1.104 W for target ONNX file
    Functional Unit Power: 0.821 W
    Interconnect Power:    0.283 W
Digital NPU Estimated Power: 0.009 W for digital target ONNX file
Total Combined (Analog + Digital NPU) Power: 1.113 W
```
末尾に`INFO:mythic.ppa_estimators.interface:Creating <model>_ppa_<timestamp>.tar.gz with saved estimation log data.` — 推定ログ付きの結果artifactが自動生成される。

### BEVFormer特有の注意点(GEN2 User Guide §2「Changes in SDK 26.05」より重要)

> Due to limitations in Mythic's functional simulator, performance estimates must be run with a single camera feed. **The final latency must be estimated by multiplying the single-camera latency by six**(6カメラ)。

これは既に[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)で確認済みのMAC数換算(アナログ側1カメラ入力 vs デジタル側6カメラ入力)と整合する事実であり、**Stage B/CでBEVFormerのlatencyを33ms制約と比較する際、`mythic-ppa-estimators`の出力(アナログ側のみ1カメラ相当)を単純にそのまま使ってはならず、アナログ側の値を6倍してからデジタル側と合算する**必要がある。

また `Number of ACEs: 24` はartifactに埋め込まれた**コンパイル時に決まった固定値**を表示しているように見える([未検証] — `mythic-ppa-estimators`自体に`--num-aces`のようなオーバーライドフラグがヘルプ上に見当たらない)。**したがってStage Bで`num_aces`を振るには、`mythic-ppa-estimators`ではなく§1の`mythic-compiler`側でコンパイル条件を変えて、artifactそのものを複数バリエーション用意する必要がある**(前版で想定していた「PPA estimator実行時に`--num-aces`を指定して再計算する」という発想は誤りだった可能性が高い)。

「NOTE: Estimation for leakage, clock tree, chip I/O power will be added in future versions」——[02_ppa_estimation.md](02_ppa_estimation.md)で確認済みのLeakage/ClockTree未算入という限界は、PDFの公式ドキュメントでも明記されており確定。

---

## 3. Accuracy Simulation: `convert_model.py steps=eval_trained`(Stage 0 / Stage A)

**実行環境**: §0.2の`gpu_run_mythic_sdk_container.sh`で起動したコンテナ内(venv有効化後)。§1/§2とは別のコンテナ(GPU付き)になる——ホストにNVIDIA GPU + Docker `nvidia`ランタイムが必要([FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)で確認済みの`--gpus all`起動と同じ要件)。

`$MYTHIC_SDK_ROOT/mythic-model-zoo`ディレクトリで、モデルごとの環境スクリプト(`.env`)を`source`してから`convert_model.py`を呼ぶ。

### 環境セットアップ(モデル別)

| モデル | 環境スクリプト |
|---|---|
| BEVFormer-tiny | `scripts/bevformer/bevformer-tiny-1600x900.env` |

### BEVFormerの実行例(GEN2 User Guide §8.3.5)

```bash
cd $MYTHIC_SDK_ROOT/mythic-model-zoo
source scripts/bevformer/bevformer-tiny-1600x900.env   # mythic-model-zoo直下から実行必須

DATASET=datasets/nuScene/ python3 scripts/common/convert_model.py steps=eval_trained \
    trained_model=../models/training/bevformer/bevformer-1600x900-trained.onnx
```

**[本探索の前提]** 本検証は本物のnuScenesデータセット(CAN bus拡張データ・annotationを含む)を用意し、このAWS環境からアクセスできる場所(§0.3)に配置する想定。GEN2 User Guideには「nuScenesはBEVFormer training guideの手順でannotationを生成する必要があるが、**Collaboration Chamber内では**ディレクトリ権限の問題でMythicがannotation生成できていない」という既知のブロッカーが明記されているが、これはCollaboration Chamber環境固有の制約であり、**本探索ではCollaboration Chamberを使わないため直接は関係しない**([FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)で判明したCAN bus拡張データの欠如の話とも別)。annotation生成自体(`generate_nuscenes_annotations`ステップ)は、実際にこのAWS環境上で1回通して問題なく完了することを確認するのが最初の検証項目(§0.3で既に言及)。

### 他モデルの実行例(パターン比較用)

```bash
# ResNet-50
cd $MYTHIC_SDK_ROOT/mythic-model-zoo
source scripts/huggingface_classifiers/resnet50-m2000.env
DATASET=datasets/imagenet_huggingface/ python3 scripts/common/convert_model.py \
    steps=eval_trained trained_model=../models/training/huggingface_classifiers/resnet50_imagenet_trained.onnx

# YOLOv8 Object Detection(データセット既定パスのためDATASET変数不要、代わりにバッチ/ワーカー数を1に制限)
cd $MYTHIC_SDK_ROOT/mythic-model-zoo
source scripts/yolov8/detect-m2000.env
python3 scripts/common/convert_model.py steps=eval_trained \
    trained_model=../models/training/yolov8/yolov8s_trained.onnx \
    eval_config.batch=1 +eval_config.workers=1
```
`eval_config.batch=1`/`+eval_config.workers=1`はGEN2 User Guideの例ではCollaboration Chamberのメモリ制約回避のためのオーバーライドとして紹介されているが、メモリに余裕があるこの環境では必須ではない(モデルによって`eval_config.*`か`training_config.WORKERS`かが変わる——GEN2 User Guide §8.3の注記通り、フレームワーク依存)。GPUメモリが不足する場合の調整用として残しておく。

`trained_model=`に渡すONNXを、§1(Compiler)でStage Aの量子化ポリシーを変えてコンパイルした結果に切り替えることで、各ポリシーの精度を評価できる(コンパイル済みartifactではなく、コンパイル**前**のtrainedONNXを指定する点に注意——`eval_trained`はコンパイル前の量子化aware評価であり、コンパイル後のartifactを評価するものではない)。

---

## 4. 実装時に最初に検証すべき事項(このドキュメントの[未検証]まとめ)

1. **[Compiler]** `--compiler-config`のYAML内の、量子化ビット幅(Stage A)・`num_aces`/並列化(Stage B)に対応する実際のキー名(§1)——実機で`bevformer_tiny_backbone_..._default_optimization.yaml`を開いて確認するのが最優先
2. **[PPA Estimator]** `mythic-ppa-estimators`に`num_aces`をオーバーライドするフラグが本当に存在しないか(`--help`の全文を確認。無い場合、Stage Bの`num_aces`探索は§1のコンパイル条件を変える形でのみ実施可能)(§2)
3. **[PPA Estimator]** BEVFormerのアナログ側1カメラ→6カメラ換算を、Stage B/Cの33ms制約判定に正しく組み込む具体的な計算方法(§2)
4. **[Accuracy Simulation]** このAWS環境上で、本物のnuScenesデータセット+CAN bus拡張データを使ったannotation生成(`nuscenes_infos_temporal_{train,val}.pkl`)が問題なく通るか(§0.3/§3)。Collaboration Chamber固有の権限問題は本探索では発生しない想定だが、実際に1回通して確認する。
5. **[環境]** `DATASET_DIR`/`TRAINING_MODELS_HOST_DIR`環境変数によるパス上書きが、`run_mythic_sdk_container.sh`/`gpu_run_mythic_sdk_container.sh`の実際のマウント処理と噛み合うか(§0.2)——スクリプト本体は静的に読んだのみで、実行未検証。

これらは[PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) §5のリスク項の更新が必要なことを示す。

---

## 参照

- [PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) — 本ツール群を使う探索計画そのもの
- `doc/user-guides/GEN2 User Guide.pdf` — 本ドキュメントの一次情報源(SDK v26.05.0公式ユーザーガイド、Compiler/PPA Estimator/Accuracy Simulationの3ツールの正式な使い方)
- [01_compilation.md](01_compilation.md) — Compiler内部実装の詳細解析(量子化/マッピング/コード生成の各フェーズ、Python APIレベル)
- [02_ppa_estimation.md](02_ppa_estimation.md) — PPA Estimator内部実装(`perf_analysis.py`/`power_estimator.py`)の式レベル解析——`mythic-ppa-estimators`が内部で呼んでいると推定される処理
- [03_accuracy_simulation.md](03_accuracy_simulation.md) — Accuracy Simulator(munc/BCM)内部の仕組み
- [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) — BEVFormer推論の環境構築・nuScenes/CARLAデータセットの扱い
