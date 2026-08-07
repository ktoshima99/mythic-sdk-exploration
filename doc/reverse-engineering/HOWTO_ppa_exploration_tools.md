# 手順書: BEVFormer PPA(面積-電力トレードオフ)探索に使うツール群

ユーザーが実際に叩くエントリーポイントは**3つ**(`doc/user-guides/GEN2 User Guide.pdf`で確認)。本ドキュメントはこの3ツールを基準に構成する。

| # | ツール | コマンド |
|---|---|---|
| 1 | **Compiler**(`mythic-compiler`) | `mythic-compiler --input-artifact <artifact> --compiler-config <yaml> --output-artifact <out>.tar.gz` |
| 2 | **PPA Estimator**(`mythic-ppa-estimators`) | `mythic-ppa-estimators --estimate-performance --estimate-power --power-inference-rate <fps> <model_artifact>.tar.gz` |
| 3 | **Accuracy Simulation**(`convert_model.py steps=eval_trained`) | `DATASET=<dir> python3 scripts/common/convert_model.py steps=eval_trained trained_model=<...>.onnx` |

計画([PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md))のStage対応: Stage 0/A=ツール1(コンパイル)+ツール3(精度)、Stage B/C=ツール1(num_aces違いでの再コンパイル)+ツール2(PPA計測)。

**環境について**: GEN2 User Guideの§4-8はCadence Collaboration Chamber(SLURM経由)を前提に書かれているが、**本探索はこのリポジトリのあるAWS環境上でdockerを直接操作する**(Collaboration Chamberは使わない)。§0でこの差分を整理する。

**SDKバージョン**: 本探索は **SDK v26.05.2**(Honda Phase 3向けパッチ)で実施。BEVFormer compiler config が6カメラ入力にネイティブ対応し、estimator ツールが structured output 化されている(§2)。

---

## 0. 実行環境の前提(このAWS環境固有 — GEN2 User Guideの記載からの差分)

**[本探索の前提]** GEN2 User Guideの§4-8はCadence Collaboration Chamber(SLURM経由でのノード確保、`/projects/tonbomythic3`配下のアクセス権管理)を前提に書かれているが、**本探索はCollaboration Chamberを使わず、このAWS環境上のdockerを直接操作する**。したがって以下は本探索では**不要**:
- Cadence Collaboration Chamberへのログイン、`tonbomythic3`/`docker`グループ確認
- SLURMセッション起動(`sbatch -p aw71-interactive`/`aw71-gpu-g6e`)——AWS上でdockerはSLURM無しで直接使える
- `/projects/tonbomythic3`配下のパス。付属`.sh`スクリプトの`DATASET_DIR`等の既定値がこのパスを指しているが、AWS環境では環境変数で上書きする(§0.2参照)

### 0.1 導入済みの状態(確認済み)

- SDK配布物一式は`mythic_sdk/v26.05.2/`にまとめてある(リポジトリルート`/home/ubuntu/mythic_sdk/26.05/`配下の`mythic_sdk/`に、バージョン別サブディレクトリ`v26.05.0/`・`v26.05.2/`として集約。この`mythic_sdk/`配下はgit管理外、S3運用)。installer zip3本はS3(`s3://mythic-sdk/26.05.2/`)から取得→`archive/`にマージ展開→zip削除済み(再取得はS3から)。起動スクリプトは`run_mythic_sdk_container.sh`・`gpu_run_mythic_sdk_container.sh`・`load_and_tag_docker_images.sh`。
- Dockerイメージは`docker images`で確認済み:
  ```
  gcr.io/mythic-devops/mythic-sdk-ubuntu-24.04:m2000-v26.05.2
  gcr.io/mythic-devops/compilerd-bin:v26.05.2   (install scriptが :develop / 1.5.4 も自動タグ付け)
  ```
  未導入の環境を新たに用意する場合は、`archive/`ディレクトリから`./install_compiler.sh`・`./install_sdk_docker_image.sh`を実行する(またはヘルパー`load_and_tag_docker_images.sh`)。

### 0.2 SDKコンテナの起動(AWS版の実際のパス上書き)

**GEN2 User Guide/付属スクリプトの既定値はCollaboration Chamber向け**(`DATASET_DIR`の既定は`/projects/tonbomythic3/datasets`)なので、**このAWS環境では環境変数で上書きして起動する**。上書き可能な変数:

| 変数 | 既定(Collaboration Chamber向け) | 本環境での指定例 |
|---|---|---|
| `MYTHIC_WORKSPACE` | `$HOME` | コンパイル結果・PPA推定結果の永続化先。既定の`$HOME`のままでも動く |
| `DATASET_DIR` | `/projects/tonbomythic3/datasets` | 本物のnuScenesデータセット配置先(実際の配置場所に置き換える) |
| `TRAINING_MODELS_HOST_DIR` | Collaboration Chamber固有パス | `archive/models/training`(既存のzip展開先) |

`MYTHIC_SDK_ROOT`はイメージにビルド時設定済みの環境変数(確認済み: `/root/mythic_sdk/v26.05.2`)。venvは**uv管理の隠しディレクトリ**になっている点に注意:
```bash
source $MYTHIC_SDK_ROOT/mythic-model-zoo/.venv/bin/activate
```
以降、本ドキュメントの`mythic-compiler`/`mythic-ppa-estimators`/`convert_model.py`の各コマンドは、**このvenv有効化後のコンテナシェル内で実行する**ことを前提とする。

Accuracy Simulation用にはGPUが必要で、`gpu_run_mythic_sdk_container.sh`(内部で`docker run --gpus all`相当を実行、SHM 2g割当・nuScenes bind mount付き)を使う。ホストにNVIDIA GPU + Docker `nvidia`ランタイムが必要([FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md))。Compiler/PPA Estimator(Stage B/C)はGPU不要。

### 0.2.1 Compiler/PPA Estimator実行に必須の追加環境変数(実機で確認済み)

起動スクリプトで立てただけでは`mythic-compiler`/`mythic-ppa-estimators`は動かない。以下2つの環境変数を追加で設定する必要がある(いずれも起動スクリプトには含まれておらず、実機で発見した):

| 変数 | 値の例 | 理由 |
|---|---|---|
| `MYTHIC_COMPILER_DOCKER_TAG` | `v26.05.2`(導入済みイメージのタグに合わせる、§0.1参照) | `mythic-compiler`/`mythic-ppa-estimators`は内部でDocker-out-of-Docker(`docker.sock`経由)により`gcr.io/mythic-devops/compilerd-bin`コンテナを子プロセスとして起動する。既定では`:develop`タグを要求する。このAWS環境には`:develop`・`:v26.05.2`・`:1.5.4`が導入済み。 |
| `MYTHIC_ROOT` | ホスト上の実在するディレクトリへの絶対パス(例: `/tmp/ppa_mythic_root_2605_2`) | SDKコンテナとコンパイラ子コンテナ間でファイルを共有するための一時ディレクトリ。**この変数の値は、SDKコンテナ内から見えるパスであり、かつホストのDockerデーモンに対して子コンテナをマウントする際にそのまま使われるパスでもある**(Docker-out-of-Dockerの制約)。したがって、SDKコンテナ起動時に`--mount type=bind,src=<同じ絶対パス>,dst=<同じ絶対パス>`として、ホストの実パスとコンテナ内パスを一致させる必要がある。未設定のまま実行すると`/data/local`が既定で使われ、ホスト側に同名のマウントが無ければ「ONNXファイルが無効です」という誤解しやすいエラーになる(コンパイラ自体は正常に起動し`--amp-arch`等は受理された上で、入力ファイルパスの解決に失敗する)。 |

起動スクリプトを使わず、直接`docker run`する場合の最小構成例(GPU不要、Stage B/C専用):
```bash
mkdir -p /tmp/ppa_workspace_2605_2/out /tmp/ppa_workspace_2605_2/compiler_configs \
         /tmp/ppa_mythic_root_2605_2 /tmp/ppa_empty_datasets_2605_2
TRAIN=/home/ubuntu/mythic_sdk/26.05/mythic_sdk/v26.05.2/archive/models/training
docker run -d --name mythic_ppa_explore_2605_2b \
  --shm-size 2g \
  --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  --mount type=bind,src=/tmp/ppa_workspace_2605_2,dst=/tmp/ppa_workspace_2605_2 \
  --mount type=bind,src=/tmp/ppa_mythic_root_2605_2,dst=/tmp/ppa_mythic_root_2605_2 \
  --mount type=bind,src=/tmp/ppa_empty_datasets_2605_2,dst=/root/mythic_sdk/v26.05.2/mythic-model-zoo/datasets \
  --mount type=bind,src=$TRAIN,dst=/root/mythic_sdk/v26.05.2/models/training \
  -e MYTHIC_ROOT=/tmp/ppa_mythic_root_2605_2 \
  -e MYTHIC_COMPILER_DOCKER_TAG=v26.05.2 \
  gcr.io/mythic-devops/mythic-sdk-ubuntu-24.04:m2000-v26.05.2 \
  sleep infinity
```
Stage B/Cはnuscenesデータセットに依存しないため(冒頭のStage対応表参照)、`datasets`マウントは空ディレクトリで構わない。**funcsim推定は1回~4時間かかるため、`sleep infinity`(期限なし)で起動しておくと長時間ジョブ中にコンテナが落ちない。**

### 0.3 本物のnuScenesデータセット + CAN busデータ + annotationの配置(Accuracy Simulationの前提)

**[本探索の前提]** Collaboration Chamber固有のannotation生成権限問題(GEN2 User Guide §8.3.5に明記)は本探索では回避する方針——**本物のnuScenesデータセット・CAN bus拡張データ・annotationを別途用意し、このAWS環境からアクセスできる場所に配置する**。配置先はコンテナ起動時に`DATASET_DIR`環境変数(§0.2)で指定し、コンテナ内`$MYTHIC_SDK_ROOT/mythic-model-zoo/datasets`にマウントされる。

annotation生成(`nuscenes_infos_temporal_{train,val}.pkl`)自体は、この環境のroot権限下で行うためディレクトリ権限の制約を受けないと想定されるが、**実際に`steps=generate_nuscenes_annotations`を1回通してannotationが生成できることを確認するのが、Accuracy Simulation着手前の最初の検証項目**。なおStage B/C(PPA探索)はnuScenesに依存しないため、この項目はaccuracy simulationを行う場合のみ関係する。

---

## 1. Compiler: `mythic-compiler`(Stage 0 / Stage A / Stage B)

**実行環境**: §0.2/§0.2.1の設定でSDKコンテナ内(venv有効化後)。コンパイル自体はコンテナ内のCPU(+内部で起動する`compilerd-bin`子コンテナ)で行われ、GPUは不要。

コンパイル済みfirmware artifactを生成する。**量子化ポリシー・タイル分割・マルチコア割当・アナログ/デジタル振り分けは全てここで一度に決まる**。

```bash
mythic-compiler --input-artifact <compiler_ready_artifact> \
    --compiler-config <model-compiler-configuration>.yaml \
    --output-artifact <workspace>/<model_name>.tar.gz
```

**所要時間(実測、m2072/high_optimization構成、6カメラ)**: 約108分。`--optimization-effort 300`のイベントドリブンソルバーがボトルネックで、GPU化はされていない(`funcsim`同様、CPU上の逐次処理。§2参照)。複数`num_aces`候補を試す場合、この時間×候補数が素直な直列実行コストになる(並列実行は可能、§4)。

### BEVFormer-tinyの実例(GEN2 User Guide §7.3、実ファイル名で確認済み)

| 項目 | パス |
|---|---|
| コンパイル対象artifact | `$MYTHIC_SDK_ROOT/models/training/bevformer/bevformer-tiny-1600x900-trained.tar.gz` |
| Compiler config(default) | `$MYTHIC_SDK_ROOT/mythic-model-zoo/configs/bevformer/compiler/bevformer_tiny_6x928x1600x3_m2072_default_optimization.yaml` |
| Compiler config(high) | `$MYTHIC_SDK_ROOT/mythic-model-zoo/configs/bevformer/compiler/bevformer_tiny_6x928x1600x3_m2072_high_optimization.yaml` |

```bash
mythic-compiler --input-artifact $MYTHIC_SDK_ROOT/models/training/bevformer/bevformer-tiny-1600x900-trained.tar.gz \
    --compiler-config $MYTHIC_SDK_ROOT/mythic-model-zoo/configs/bevformer/compiler/bevformer_tiny_6x928x1600x3_m2072_high_optimization.yaml \
    --output-artifact <workspace>/bevformer_tiny_m2072_high.tar.gz
```

出力artifact(tar.gz)には`artifacts/firmware/weight_utilization.txt`・`sram_utilization.txt`(重み/SRAM使用率レポート)、`artifacts/firmware/vnn/BevformerTiny_*`(digital NPU JSON)などが含まれる(§2でこのartifactをそのまま使う。実際の`perf_trace_dump.h5`は`--estimate-performance`実行時に別途`artifacts/ppa/`配下へ生成される)。

### Stage A: 量子化ポリシーを振る

計画の量子化ポリシー(§1のQuantizationConfig.tensor_n_bits、8/16bitのみ許可)を試すには、**`--compiler-config`に渡すYAMLを複製し、量子化関連の設定項目を書き換える**。既定/高最適化の2種類が用意されている(`_default_optimization.yaml` / `_high_optimization.yaml`)。**具体的にどのYAMLキーが`tensor_n_bits`相当の量子化ビット幅指定に対応するかは未検証**(本探索では既定8bitで精度制約を満たすため、量子化ポリシー探索は不要——PLAN §2)。

### Stage B: `num_aces`を振る(実機で確認済み)

**`num_aces`は`--compiler-config`のYAML内の`COMPILER_OPTIONS`リストにある`--amp-arch`フラグで指定する**。BEVFormer-tinyの既定YAML(`bevformer_tiny_6x928x1600x3_m2072_*.yaml`)には以下が書かれている:

```yaml
COMPILER_OPTIONS: [
  "--amp-arch m2072",  # Gen 2 with 72 Analog Compute Engines
  "--acm",
  "--input-dims 6 928 1600 3",   # 6カメラ入力(v26.05.2ネイティブ)
  "--stats terse",
  "--optimization-effort 300",
]
```

`--amp-arch`に渡せる値は**このSDKに同梱の`funcsim`バイナリが対応する4パターン**(`funcsim --help`の`--hardware_topology`許容値と対応)だが、**実際にコンパイルが通るのは`m2048`と`m2072`の2つのみ**:

| `--amp-arch`値 | ACE数 | 対応する`funcsim --hardware_topology` | BEVFormer-tinyでのコンパイル結果 |
|---|---|---|---|
| `m2024` | 24 | `denali_24ace_6tile` | **失敗**(CP-SAT `status: UNKNOWN`、122分後に L0 optimization failed。原因は下記) |
| `m2032` | 32 | `denali_32ace_8tile` | **利用不可**(バイナリにターゲット定義なし。原因は下記) |
| `m2048` | 48 | `denali_48ace_12tile` | **成功**(~94分、exit 0) |
| `m2072` | 72 | `denali_72ace_18tile` | **成功**(既定、~108分) |

新しいarch用のYAMLは、既定の`bevformer_tiny_6x928x1600x3_m2072_high_optimization.yaml`をコピーして`--amp-arch`だけ書き換えれば動く(`--input-dims`等の他パラメータはm2072のまま流用可)。

#### `m2032`が利用不可な根本原因(バイナリ調査+実機での抜け道テストで確定)

`m2032`のエラー(`ABORT: Unable to build AMP architecture "m2032" without compatibility enabled!`)を`dnn_compiler`バイナリ自体の文字列(`strings`)・逆アセンブルで調査した結果、**このコンパイラビルドには`m2032`のターゲット定義が一切存在しない**(バイナリ中の全AMPアーキ文字列は`m2024`/`m2048`/`m2048_ace16mb`/`m2072`のみ)。`funcsim --hardware_topology`に`denali_32ace_8tile`があるのは、機能シミュレータが将来対応予定のトポロジを先行して持っているためで、コンパイル経路では使用不可。

エラー文言の「*without* compatibility enabled」が「compatibilityを*有効化する*手段が存在する」ことを示唆していたため、逆アセンブルで中断箇所を追ったところ、中断は`mythic::hw::TargetSpecFromAmpArch(std::string, bool)`内で発生しており、この**第2引数の`bool`が「compatibility mode」のトグル**で、`--boreas-a-compatible`フラグがこれを立てることを特定した(バイナリ内文字列`boreas_a_compatible`・`Generated Target will use compatibility mode.`と対応)。

**実機で`--boreas-a-compatible`を付けて`m2032`をテストした結果**:フラグは効き、ログが`Generating Target for AMP arch m2032.` → `Generated Target will use compatibility mode.`に変化したが、**依然として中断**した——エラー文言が`without`から`with`に変わっただけ:
```
ABORT: Unable to build AMP architecture "m2032" with compatibility enabled!
```
つまりcompatibilityモードの有無によらず、このコンパイラビルドには`m2032`のターゲット定義そのものが無い。**唯一考えられる抜け道(compatibilityトグル)を実機で塞いだ上で、32 ACEsはこのSDKバージョンでは利用不可と確定した。** なお`--boreas-a-compatible`フラグ自体は実在し機能する(m2032特有の問題であって、フラグが無いわけではない)。

#### `m2024`が失敗する根本原因——高充填によるCP-SATタイムアウト(INFEASIBLE証明ではない)

m2024失敗時のログでは、コンパイラの`CpSatClustering`パス(Google OR-Tools CP-SATソルバーによる、モデル演算をACEタイルに配置する制約充足問題)は**「解なし」と確定したわけではなく、`status: UNKNOWN`(探索予算を使い切ったための打ち切り)で終了する**:
```
status: UNKNOWN
walltime: 7319
...
schedule.cpp:915 FATL| CHECK FAILED: success L0 optimization failed for Crate "main_graph_main_graph"!
```
探索予算引き上げ(`--ace-effort 1000`)・充填密度レバー(`--multiple-weights-per-bank`)を試しても同じ`status: UNKNOWN`で失敗する(3手法とも90分超)。

**容量計算**(1タイル=4 ACE、重み容量は固定`4,980,736`/タイル。成功構成の`weight_utilization.txt`で検証済み):

| 構成 | タイル数 | 重み容量 | 実配置(=Weight Blocks数) |
|---|---|---|---|
| m2072 | 18 | 89,653,248 | 35,756,800 / 39.9%(235ブロック)|
| m2048 | 12 | 59,768,832 | 31,253,248 / 52.3%(193ブロック)|
| **m2024** | **6** | **29,884,416** | 推定 ~25M(156ブロック)|

Weight Blocks数はACE数とともに増える(コンパイラはACEが多いほど重みを複製して並列化するため)。逆にm2024の156ブロックは既に並列化を最小限に抑えた状態。ブロック数比からの概算で、m2024が必要とする重みは約25M——容量29.88Mに対して**約84%充填**。

**したがって24 ACEsの失敗は「物理的に絶対入らない(INFEASIBLE)」ではなく、bin-packingが最も難しくなる高充填(~84%)領域でCP-SATが実用時間内に解を見つけられずタイムアウトしている**。ただし実務上は:(1)探索予算3.3倍・充填密度レバーの両方を試して90分超かけても解けない、(2)仮に無理やり通しても約84%充填で並列化の余地がほぼ無く、並列化を削ればlatencyが悪化するため33ms制約(72 ACEですらマージン4.3%)を満たす見込みがほぼゼロ——という2点から、**24 ACEsはSKU候補として利用不可**(「本質的に不可能」ではなく「実用時間内にコンパイルが通らず、通ってもlatency制約を満たせない」)。

コンパイル所要時間の実測(すべて`--optimization-effort 300`・high_optimization設定、6カメラ):

| `num_aces` | 所要時間(壁時計) | 結果 |
|---|---|---|
| 24 | 約122分 | 失敗(`status: UNKNOWN`) |
| 32 | 数秒 | 即時失敗(ターゲット定義なし。`--boreas-a-compatible`でも中断、上記) |
| 48 | 約94分 | 成功 |
| 72 | 約108分 | 成功 |

**結論: BEVFormer-TinyでStage Bの探索対象になる`num_aces`は48と72の2点に確定**(24は3手法、32は抜け道フラグを試したがいずれも利用不可と確認)。ACE数が少ないほど速いとは限らない(24が最も長くかかった上に失敗している)——配置制約の難しさとコンパイル時間は単純な比例関係にない。

`n_mps`に対応するYAMLキーは、BEVFormer既定YAML内には見当たらず未確認(§4参照)。

---

## 2. PPA Estimator: `mythic-ppa-estimators`(Stage B / Stage C)

**実行環境**: §1と同じコンテナ内。コンテナを維持していれば§1の直後にそのまま続けて実行できる。

**§1でコンパイルしたartifact(tar.gz)をそのまま渡すだけ**でlatency/power両方を推定する。`perf_analysis.py`(HDF5トレース)と`power_estimator.py`(L0 protobuf)は**このコマンドの内部で自動的に両方読まれる**。

```
mythic-ppa-estimators --help
usage: mythic-ppa-estimators [-h] [--estimate-performance] [--estimate-power]
                              [--power-inference-rate POWER_INF_RATE]
                              [--allow-fps-over-max] [--functional-simulation]
                              model_artifact_path

positional arguments:
  model_artifact_path   Path to Mythic model artifact archive (tar.gz)

options:
  -h, --help            show this help message and exit
  --estimate-performance  Estimate performance for model artifact
  --estimate-power        Estimate power for model artifact
  --power-inference-rate POWER_INF_RATE
                        Set a target inference rate in frames/second (fps) for power estimation
  --allow-fps-over-max  要求fpsが達成可能fpsを超えてもクランプせず推定する
  --functional-simulation  推定なしでfuncsim単独実行
```

BEVFormer実行例(6カメラネイティブ):
```bash
mythic-ppa-estimators --estimate-performance --estimate-power --power-inference-rate 30 \
    <workspace>/bevformer_tiny_m2072_high.tar.gz
```

### 出力(実際のログ、要点)

structured output 化されており、6カメラの最終値を直読できる。m2072/high_optimization の実測:
```
Analog NPU Processing Time: 26.95 ms          (6カメラ込)
Digital Estimated Frame Processing: 4.63 ms
Combined Analog + Digital NPU Latency: 31.58 ms   (6カメラの最終latency、33ms制約と直接比較可)
Combined Analog + Digital NPU Total Estimated Frame Rate: 31.67 fps
ACE Utilization: 69.43%
Number of ACEs: 72                            (正表示)
...
Total Combined (Analog + Digital NPU) Power: 4.505 W   (@30fps: analog 3.287 / digital 1.218)
```
末尾に`Creating <model>_ppa_<timestamp>.tar.gz with saved estimation log data.` — 推定ログ付きの結果artifactが自動生成される。

### 【最重要】6カメラはネイティブ、手動×6は不要

v26.05.2 では BEVFormer compiler config が `--input-dims 6 928 1600 3`(6カメラ)にネイティブ変更されている(§1)。したがって:
- `Combined Analog + Digital NPU Latency`(および`Analog NPU Processing Time`)は**既に6カメラの最終値**。この値をそのまま33ms制約と比較する。
- **アナログ側を手動で×6してデジタル側と合算してはならない**(6倍の二重計上になる)。
- ppa-estimatorは`--power-inference-rate 30`で**1回だけ**実行すればよい(target fps用・6倍fps用の2回実行は不要)。

**GEN2 User Guide本文には旧「single-camera latencyを6倍せよ」の記述が残っている**が、これは旧アーティファクト時代の記述の残存で、新config(`--input-dims 6 ...`)+ Compiler Optimization Report v1.2(Combined 31.58msを直接出力)が正。

### 所要時間・GPU化不可(実機で確認済み)

`--estimate-performance`は内部で`funcsim`(機能シミュレータ、CPU上のイベントドリブン逐次シミュレータ)を子コンテナで実行する。**`funcsim --help`で確認した限り、GPU関連オプションは存在しない**(`-j, --threads`のマルチスレッド化のみ)。BEVFormer-tiny(m2072/high_optimization、6カメラ)では、`--estimate-performance --estimate-power`1回の実行に**約4時間**かかった(実測)。`num_aces`候補ごとにこの時間がかかる点を作業時間見積りに反映する。総ウォールクロック短縮のためには複数SKUを並列実行する(§4)。

### fpsクランプ挙動(実機で確認、latency判定には影響しない)

`--power-inference-rate 30`を指定しても、モデルがそのfpsを達成できない場合、ツールは達成可能fpsへ**自動クランプ**する:
```
INFO:mythic.ppa_estimators.estimate:Power inference rate of '30' exceeds estimated. Using estimated rate of '23' instead.
```
m2048はcombined frame rate 23.00fps(=1/43.47ms)しか出せないため、30fps要求が23fpsにクランプされる。**クランプ後のpower(m2048: 3.428W)は要求fpsで動いていないため、30fps基準のm2072(4.505W)とはfps基準が異なり直接比較できない。** `--allow-fps-over-max`でクランプを抑止して30fps強制推定も可能。

**ただしlatency判定はpower推定と独立**(combined latencyはfpsクランプの影響を受けない)なので、クランプはStage Bの主判定(33ms制約の可否)には影響しない。

### `Number of ACEs`表示について

`Number of ACEs`はartifactに埋め込まれた**コンパイル時に決まった固定値**を表示する(`mythic-ppa-estimators`自体に`--num-aces`のようなオーバーライドフラグはヘルプ上に存在しない)。v26.05.2 では m2048=48、m2072=72 と**正しく表示される**(structured output 化で修正済み)。**Stage Bで`num_aces`を振るには、`mythic-ppa-estimators`ではなく§1の`mythic-compiler`側でコンパイル条件(`--amp-arch`)を変えて、artifactそのものを複数バリエーション用意する必要がある**(§1参照)。

### die area(面積)について

**v26.05.2 の `mythic-ppa-estimators` は die area 値を出力しない**(旧版が出していた `Estimated Die Area to Achieve Estimated Processing Time` 行は撤去された)。**SKUの面積比較には、データシート由来の物理傾き 5.278 mm²/ACE(=380/72)を使う**: 24 ACE=158mm² / 48 ACE=253mm² / 72 ACE=380mm²。これでarea単調増加が正しく成立する。詳細は[[ppa-die-area-not-physical]] / [PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) §4.1参照。

「NOTE: Estimation for leakage, clock tree, chip I/O power will be added in future versions」——[02_ppa_estimation.md](02_ppa_estimation.md)で確認済みのLeakage/ClockTree未算入という限界は、PDFの公式ドキュメントでも明記されており確定。PPA推定値は絶対値のサインオフではなく相対比較専用として扱う。

### レイテンシの律速要素を分解する(funcsim再実行不要)

公表レイテンシは `max(ACE, SRAM, SIMD)` なので、**そのモデルがACE律速なのかSRAM律速なのか**は既定出力の`Critical Path ACE Latency`と`Maximum SRAM Read/Write Time`を見比べれば判定できる(§2の出力例では両方とも出ている)。ACE律速ならACE増設でレイテンシが縮むが、SRAM律速なら縮まない——Stage B/Cのnum_aces選定で直接効く判断材料になる。

**`_ppa_*.tar.gz`に`artifacts/ppa/perf_trace_dump.h5`が保存されているので、約4時間のfuncsimを回し直さずに再解析できる**(解析自体は約9秒)。バイト法の項やタイル別内訳まで含めた全項の取得は:

```bash
tar xzf <model>_ppa_<ts>.tar.gz artifacts/ppa/perf_trace_dump.h5
tools/perf_breakdown/perf_breakdown.sh artifacts/ppa/perf_trace_dump.h5 48 out_dir   # 48=m2048, 72=m2072
```

手順の原理・実測値・SIMD項が常に0である件は[02_ppa_estimation.md](02_ppa_estimation.md) §3.9を参照。

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

**[本探索の前提]** 本検証は本物のnuScenesデータセット(CAN bus拡張データ・annotationを含む)を用意し、このAWS環境からアクセスできる場所(§0.3)に配置する想定。GEN2 User Guideの「Collaboration Chamber内ではディレクトリ権限の問題でannotation生成できない」という既知ブロッカーはCollaboration Chamber環境固有の制約であり、本探索ではCollaboration Chamberを使わないため直接は関係しない。annotation生成自体(`generate_nuscenes_annotations`ステップ)がこのAWS環境上で1回通ることを確認するのが最初の検証項目(§0.3)。

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
`eval_config.batch=1`/`+eval_config.workers=1`はGPUメモリが不足する場合の調整用(モデルによって`eval_config.*`か`training_config.WORKERS`かが変わる——GEN2 User Guide §8.3の注記通りフレームワーク依存)。

`trained_model=`に渡すONNXを、Stage Aの量子化ポリシーを変えたものに切り替えることで各ポリシーの精度を評価できる(コンパイル済みartifactではなく、コンパイル**前**のtrained ONNXを指定する点に注意——`eval_trained`はコンパイル前の量子化aware評価)。

---

## 4. 未検証事項

1. **[Compiler]** `n_mps`に対応するYAMLキーがBEVFormer既定YAML内に見当たらない(§1)。`--amp-arch`以外の`COMPILER_OPTIONS`項目(`--acm`/`--input-dims`/`--stats`/`--optimization-effort`)に対応が無く、別のフラグまたは`n_mps`はBEVFormerでは固定なのか要確認。
2. **[Compiler]** 量子化ビット幅(`tensor_n_bits`相当)に対応するYAMLキーが未特定(§1 Stage A)。本探索では既定8bitで精度制約を満たすため探索不要。
3. **[並列実行]** `mythic-compiler`/`mythic-ppa-estimators`はDocker-out-of-Docker(`docker.sock`共有)で子コンテナを起動するが、`MYTHIC_ROOT`がコンパイルごとにユニークなサブディレクトリを作るため、複数SKUの同時コンパイル/推定で衝突しないことを実機で確認済み(本探索でも並列実行して総時間を短縮した)。

---

## 参照

- [PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) — 本ツール群を使う探索計画そのもの
- `doc/user-guides/GEN2 User Guide.pdf` — 本ドキュメントの一次情報源(Compiler/PPA Estimator/Accuracy Simulationの3ツールの正式な使い方)
- `doc/reports/Compiler Optimization Report - BEVFormer-Tiny.pdf`(v1.2) — m2072構成のPPA推定コマンドと詳細レポート、照合基準値の出典
- [01_compilation.md](01_compilation.md) — Compiler内部実装の詳細解析(量子化/マッピング/コード生成の各フェーズ、Python APIレベル)
- [02_ppa_estimation.md](02_ppa_estimation.md) — PPA Estimator内部実装(`perf_analysis.py`/`power_estimator.py`)の式レベル解析
- [03_accuracy_simulation.md](03_accuracy_simulation.md) — Accuracy Simulator(munc/BCM)内部の仕組み
- [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) — BEVFormer推論の環境構築・nuScenes/CARLAデータセットの扱い
