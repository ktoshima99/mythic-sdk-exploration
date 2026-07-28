# Mythic M2000 SDK 処理ロジック解析ドキュメント（総合概要・索引）

対象 SDK: **Mythic M2000 SDK v26.05.0**（内部呼称 `vnnsdk 26.05`）
解析対象展開先: `mythic_sdk/v26.05.0/_extracted_compiler/`（コンパイラコンテナ由来）、`mythic_sdk/v26.05.0/_extracted_sdk/`（SDK コンテナ由来）

---

## 0. このドキュメント群について

Mythic M2000（フラッシュ／NVM メモリセルをアナログ乗算器として用いる **アナログ compute-in-memory 型 AI アクセラレータ**）SDK の、処理ロジックをソースコードから解析した記録である。

| # | ドキュメント | 対象ステップ | 解析状況 |
|---|---|---|---|
| 01 | [01_compilation.md](01_compilation.md) | **コンパイル** | 解析済（Compiler コンテナ側） |
| 02 | [02_ppa_estimation.md](02_ppa_estimation.md) | **PPA 推定**（性能・電力・面積） | 解析済（Compiler コンテナ側） |
| 03 | [03_accuracy_simulation.md](03_accuracy_simulation.md) | **精度シミュレーション** | 解析済（Part A: SDK コンテナ本体 / Part B: Compiler コンテナ部品） |
| 04 | （未作成） | **再学習** | 未解析（SDK コンテナ側） |

### 解析手法
- **コンパイラコンテナ**: `compiler_m2000.tar`（OCI イメージ）を `docker load` し、`compilerd-bin:1.5.2` から生 Python ソース約 27,000 行を `_extracted_compiler/` に抽出。doc 01/02、および doc 03 Part B はこれに基づく。
- **SDK コンテナ**: `sdk.tar` を `docker load` し `mythic-sdk-ubuntu-24.04:m2000-v26.05.0` を起動、`munc` パッケージの核心（約 8,843 行: ノイズモデル・BCM モデル階層・モンテカルロ・評価ワークフロー）を `_extracted_sdk/` に抽出して解析。doc 03 Part A はこれに基づく。再学習ロジック（`train.py` 群）は存在を確認したのみで本文解析は未実施。
- コンパイル済みバイナリ（`dnn_compiler` 23MB, `vnnmap`, `vnncodegen`, `funcsim`）は `strings` で補助的に解析。
- 各ドキュメントは実コードの **ファイルパス:行番号** を根拠として引用し、確定できない箇所は **[推測]** と明記している。

---

## 1. SDK コンポーネントと 2 つの Docker コンテナ

| コンテナ | イメージ | 役割 |
|---|---|---|
| **Compiler** | `gcr.io/mythic-devops/compilerd-bin:1.5.2` | コンパイル・PPA 推定の**バックエンドエンジン**。ユーザーが直接入る想定ではなく、SDK コンテナから docker.sock 経由で呼ばれる |
| **Mythic SDK** | `mythic-sdk-ubuntu-24.04:m2000-v26.05.0` | ユーザーが作業する**メイン環境**。再学習・精度シミュレーション・コンパイラフロントエンドを含む。`mythic-model-zoo` を格納 |

### ステップ × コンテナ 対応表（実物確認済み）

| ステップ | Compiler コンテナ | SDK コンテナ | 本ドキュメント群の解析状況 |
|---|---|---|---|
| **再学習** | ✗ | ✅ `train.py` 群（pythia/yolopx/huggingface/bevformer）, `distillation_trainer.py` | ⛔ 未解析 |
| **精度シミュレーション** | △ 部品（`vnnort`: データセットローダ/評価メトリクス/量子化） | ✅ **本体**（`convert_model.py`→`munc`→`conversion_steps.py`, ノイズモデル `_pytorch/noise.py`, BCM モデル階層 `bcm/bcm_models/`, モンテカルロ `_monte_carlo/`） | ✅ 解析済（doc 03 Part A=本体 + Part B=部品） |
| **コンパイル** | ✅ バックエンド（`dnn_compiler`, `vnnmap`, `vnncodegen`） | フロントエンド（`mythic-compiler` が Compiler コンテナを呼ぶ） | ✅ バックエンド解析済（doc 01） |
| **PPA 推定** | ✅ （`perf_analysis.py`, `power_estimator.py`） | フロントエンド（`mythic-ppa-estimators`） | ✅ 解析済（doc 02） |

> doc 01/02 の解析は Compiler コンテナ側で完結しており妥当。doc 03（精度シミュレーション）は両コンテナに跨る全体像を解析済み——Part A（SDK コンテナ側の駆動本体・確率的ノイズモデル・BCM）と Part B（Compiler コンテナ側の QDQ 決定論的量子化・推論エンジン・評価メトリクス）の関係は doc 03 Part C を参照。

### 2 コンテナの連携
SDK コンテナは `run_mythic_sdk_container.sh` で `/var/run/docker.sock` をマウントして起動する。ユーザーは SDK コンテナ内で作業し、コンパイル時にはフロントエンド（`mythic-compiler`）が docker 経由で Compiler コンテナを起動する（docker-in-docker 的連携）。

### 内部呼称についての注意
SDK 内部では旧社名 **videantis** 由来の `vid*` プレフィックス、`v-NN Mapper`(vnnmap) という呼称が一貫して使われる。ハードウェア（M2000）の物理仕様は `mythic.*` protobuf 側に定義される。

---

## 2. 4 ステップを貫くデータフロー

凡例: 【S】=SDKコンテナで実行  【C】=Compilerコンテナで実行

**重要**: 精度シミュレーションとコンパイルは直列ではなく、**再学習が生成する学習済み ONNX（`trained.onnx`）から並行して分岐する**枝である。精度シミュ（`eval_trained`）はコンパイルの前提ではなく、学習結果を評価する末端の枝。コンパイル（`to_acm`→`create_artifact`）も同じ `trained.onnx` を独立に入力する。

```
┌─ 再学習 【S】(mythic-model-zoo/*/train.py, 未解析) ───────────┐
│  データセット + FP32 ONNX                                     │
│   → to_structural → to_training → train(アナログaware再学習)  │
│  出力: 学習済 ONNX (data/trained.onnx)                        │
└──────────────────────────────────┬───────────────────────┘
                                    │  学習済 trained.onnx
                    ┌───────────────┴────────────────┐
                    │ (分岐: 同じ trained.onnx を独立に入力)  │
                    ▼                                 ▼
┌─ 【枝A】精度シミュ 【S】本体+【C】部品 [doc03] ┐  ┌─ 【枝B】コンパイル [doc01] 【S→C】───────────┐
│ eval_trained(=eval_onnx_step)                │  │ to_acm → create_artifact → compile          │
│  → make_torch_net() で BCM/ACE アナログ       │  │  → 最適化: 標準ONNXオペ→com.videantis vidConv │
│    モデル(6階層)を forward に注入              │  │    (MatMul/Gemm/Attention も Conv=行列積に統一)│
│ ノイズ: 重みプログラミング誤差+温度ドリフト     │  │  → 量子化: 8bit対称 power-of-two 固定小数点     │
│   +ADC熱雑音 (noise.py, bcm/bcm_models/)     │  │    (max_exp / 重みper-ch / bias16bit)         │
│ 実データセットで HF Trainer.evaluate()        │  │  → .vidir → dnn_compiler(auto_partition):     │
│ (モンテカルロ時) NIST片側許容区間で保証精度    │  │    アナログ/デジタル振り分け(Denali/Digital)   │
│                                              │  │  → vnnmap: ACEタイル配置/BitSpreading/NVM書込 │
│ 出力: mAP / NDS / accuracy 等の精度メトリクス  │  │  → vnncodegen/vnnrtgen → ランタイムバイナリ    │
│                                              │  │ 出力: compiler_ready_artifact/               │
│ ※コンパイルの前提ではない（評価専用の末端）   │  │  off_chip_0→on_chip_1_bcm(ACE)→off_chip_2    │
└──────────────────────────────────────────────┘  └──────────────────────────┬──────────────────┘
                                                                              │ compiled artifact
                                                   ┌──────────────────────────▼──────────────────┐
                                                   │─ PPA推定 【C】[doc02] ────────────────────────│
                                                   │ 入力: perf_trace_dump.h5 + vnn JSON           │
                                                   │ 性能: ボトルネックmax(ACEクリティカルパス,SRAM,SIMD)│
                                                   │ 電力: 1推論エネルギー×fps / 面積: タイル数×42mm² │
                                                   │ 出力: latency / fps / power / area            │
                                                   └───────────────────────────────────────────────┘
```

補足:
- **PPA 推定はコンパイルの下流**（コンパイル成果物 `perf_trace_dump.h5` / vnn JSON を入力とする）であり、これは直列関係で正しい。
- `mythic-model-zoo` 内では全ステップが 1 つのステップ列 `to_structural→to_training→train→eval_trained→…→to_acm→create_artifact→compile` に定義されているが、`eval_trained`（枝A）と `to_acm`/`create_artifact`（枝B）は**互いに依存せず、どちらも `train` の出力 `trained.onnx` を入力とする独立した枝**である（doc 03 Part A.3 / A.12 参照）。順番に並んでいるのは列挙上の都合で、データ依存ではない。
- `eval_trained` は step_type=`eval_onnx`（精度シミュ内の1ステップ名）。

---

## 3. 3 ステップ横断で共通する「アナログ量子化」の核心

3 ドキュメントに共通する最重要概念。M2000 のアナログ MAC は固定小数点前提であり、コンパイル・PPA・精度のすべてがこれを軸に動く。

- **power-of-two（2 の冪）スケール固定小数点**。非対称ゼロ点なし。「`max_exponent`（2 の指数）」表現。
  - 定義: `quantizer/quant_utils.py`（`round_up_to_power_of_two`, `power_of_two_scaling_only=True`）
- **量子化式**: `q = clip(round(x * 2^(fraction - max_exp)), -2^(n-1), 2^(n-1)-1)`
- ビット幅: 入出力 8bit（小数 7bit）、**バイアス 16bit**（小数 14bit）、最終層は per-tensor 強制
- **レイヤ間の指数整合**: Conv/Relu/Pool 間で入力・重み・出力の指数を強制的に桁合わせ（`layer_handlers.py`）——これがアナログ固定小数点ハードウェアの必須制約
- **精度シミュレーション**では、この量子化を `QDQLayer`（`quantizer/qdq_layer.py`）として ONNX グラフに fake-quant ノードとして埋め込み、ONNXRuntime で決定論的に数値実行することでアナログ挙動を模擬する

---

## 3.5 モデルの状態遷移と中核となる中間表現

学習済みモデルは、SDK コンテナ（`munc`）→ Compiler コンテナ（`vnnort`）へと進む過程で**複数の中間表現（モデル状態）**を辿る。各状態は特定のクラス／グラフ形式で表現され、`BCM` はそのうちの 1 つである。BCM だけが中核なのではなく、以下が並ぶ「状態の連鎖」として理解するのが正確。

### SDK コンテナ側のモデル状態（`munc._constants.MODELType`, `_session.py` の状態遷移）

`munc` は `MODELType`（`_constants.py:327`）でモデルの状態を管理し、`_session.py` の `get_*_conversion_ops()` が状態を進める:

```
ORIGINAL ──to_structural──▶ (structural) ──to_training──▶ MYTHIC ──to_acm──▶ BCM ──create_artifact──▶ COMPILER
OriginalModel               中間graph        MythicModel            BCMModel               CompilerModel
```

| モデル状態 / 中間表現 | 実体・クラス | 役割 | 使われるステップ |
|---|---|---|---|
| **ORIGINAL** (`OriginalModel`) | 素の FP32 ONNX | 学習/エクスポート直後の入力モデル | `to_onnx` の出力 |
| **structural** | 中間 ONNX グラフ | on/off-chip マーキング・定数畳み込み等の構造整理済み | `to_structural` |
| **MYTHIC** (`MythicModel`) | `MythicConv2d`/`MythicLinear` 等の Mythic ノード（`_constants.py:89-94`）を持つ ONNX | **アナログaware 再学習可能**な量子化グラフ（FSR 分解・DSF 学習可能化済み） | `to_training` の出力 → `train` の対象 |
| **BCM** (`BCMModel`) | `BCMConv2d`/`BCMLinear`（`bcm_layers.py`）= **アナログ MAC(`mma_class`) + デジタルデータパス**（doc 03 A.6.1） | 学習済みモデルを**ハードウェア忠実に数値再現**する中間表現。別名 **ACM (Analog Compute Model)** | `to_acm` で生成 → `eval_acm`・`create_artifact` の入力 |
| **COMPILER** (`CompilerModel`) | コンパイラ入力用に整えた ONNX（`compiler_ready_artifact.tar.gz`） | Compiler コンテナへ渡す最終成果物 | `create_artifact` の出力 |
| **TorchNet** | `torch.nn.Module`（`munc._torchnet.TorchNet`, doc 03 A.13） | ONNX（Mythic/BCM ノード含む）を**実行可能な PyTorch モデル**に変換。`make_torch_net()` で構築 | `eval_trained`・推論動画生成の実行基盤 |

> `RETRAIN`/`PTM` も `MODELType` に定義されているが本解析では未確認。

### Compiler コンテナ側の中間表現（doc 01）

Compiler コンテナに `COMPILER` モデルが渡ると、さらに別系統の中間表現に変換される:

| 中間表現 | 実体 | 役割 |
|---|---|---|
| **vidConv 等の `com.videantis` カスタムオペ** | 最適化後 ONNX | 標準オペを Mythic 演算に書き換えた形（MatMul/Attention も Conv に統一） |
| **QDQLayer** | fake-quant ONNX 関数（`qdq_layer.py`） | 決定論的 power-of-two 量子化を ONNXRuntime 上で数値模擬（doc 03 Part B / Part C） |
| **`.vidir`** (CapnProto Network) | `VNNMapExporter` 出力 | 量子化情報（max_exponents 等）を埋めた中間ネットワーク |
| **`.vci` / L0 IR** | `vnnmap`/`dnn_compiler` 出力 | ACE タイル配置・パーティション済みの低レベル表現 |

### 中核モデルの対応関係（要点）

- **BCM (= ACM)**: SDK コンテナ側で、学習済みモデルをハードウェア忠実に**数値再現**する中間表現。**精度評価とコンパイラ artifact 生成の両方**で使われる（精度シミュ専用ではない。doc 03 A.5/A.12）。artifact 生成時は `SwitchBCM(munc_digital)` でノイズなしデジタル忠実モデルに固定される。
- **TorchNet**: BCM/Mythic ノードを含む ONNX を**実行**するための PyTorch ラッパ。精度評価・推論動画の実行エンジン。
- **QDQLayer**: Compiler コンテナ側の決定論的量子化模擬。BCM の確率的ノイズモデルの「ゼロノイズ極限」に相当する下部構造（doc 03 Part C）。
- **vidConv / .vidir / .vci**: Compiler コンテナ側のコンパイル用中間表現。

つまり「中核となるモデル」は BCM 単独ではなく、**`ORIGINAL → structural → MYTHIC → BCM(ACM) → COMPILER` という状態連鎖**であり、実行時にはそれぞれ **TorchNet**（SDK 側実行）や **QDQLayer/.vidir/.vci**（Compiler 側）として具現化される。

---

## 4. ハードウェア仕様（protobuf・電力モデルから復元）

| 項目 | 値 | 出典 |
|---|---|---|
| ACE（アナログ計算エンジン）1 タイルの行列サイズ | **1280 入力 × 272 出力** | `power_estimator.py`（MAX_ACE_INPUTS/OUTPUTS） |
| ACE 演算時間 | **160 ns 固定**（8bit ADC 出力） | `perf_analysis.py`（ACE_DURATION_NS） |
| チップあたり ACE 数（既定） | **24**（= 6 タイル × 4 ACE） | `perf_analysis.py`（num_aces） |
| アナログプロセスノード | 28nm 固定 | `power_estimator.py` |
| デジタル NPU プロセスノード | 5/12/28nm 可変（既定 5nm） | `power_estimator.py` |
| 1 ACE タイル面積 | 約 42.16 mm²（shrink 0.9 適用後） | `perf_analysis.py` |
| 重み配置 | MmaWeightArea（バンク矩形）へタイリング + BitSpreading + NvmRampVoltage | protobuf L0 IR |

---

## 5. 解析の重要な限界（3 ドキュメント共通の注意）

各ドキュメント本文でも詳述しているが、横断的に押さえるべき限界:

1. **コンパイラの実マッピングは Python 外の C++ バイナリにある**
   「アナログ/デジタル振り分け」「off_chip/on_chip 分割」「ACE タイル配置」の実装は Python ソースではなくコンパイル済みバイナリ `dnn_compiler`（23MB）/ `vnnmap`（3MB）内にある。`strings` 解析により振り分けの所在・単位・判定基準は判明している:
   - 振り分け本体 = `dnn_compiler` の `mythic/optimizer/high/passes/auto_partition.cpp`
   - 単位 = **IPU パーティション**（`Denali`=アナログ IPU / `Digital`）。判定関数 `IsDenali()` / `IsDigital()`
   - 基準 = ①演算のアナログ実行可否（Conv/Dense/MmaDot はアナログ、DepthwiseConv 等はデジタル）②物理 SRAM 容量（超過で分割）
   - 分割境界 = infeed/outfeed 接続 → artifact の `off_chip_0 → on_chip_1_bcm → off_chip_2`

   未確定なのは分割点の探索アルゴリズムとコスト関数の内部実装のみ（C++ 逆アセンブルが必要）。詳細は [01_compilation.md](01_compilation.md) の 3.3 節。なお "BCM" という語は Compiler コンテナの Python ソース・バイナリ双方に存在しない（"BCM" は SDK コンテナ側 `munc` の Boreas Compute Model を指す。§3.5 参照。artifact ステージ名 `on_chip_1_bcm` の "bcm" の由来はこの Boreas Compute Model と考えられる[推測]）。

2. **PPA の電力モデルに未算入の項目**
   leakage / clock tree / PCIe / D2D / NOC 電力は定義済みだが `total` に未算入（コード上 "future versions" とコメント）。レイテンシは近似（作者コメントに "this probably isn't right" の注記あり）。

---

## 6. 抽出ソースの所在

### Compiler コンテナ由来（`_extracted_compiler/`）

| パス | 内容 | 規模 |
|---|---|---|
| `vnnort/` | コンパイラ中核（optimizer/quantizer/inference/models） | 79 ファイル / 17,282 行 |
| `perf_analysis.py` | PPA 性能推定 | 1,329 行 |
| `mythic_pkg/m2000_power_estimator/power_estimator.py` | 電力推定 | 649 行 |
| `vnnsdk_scripts/` | 各モデル後処理・評価スクリプト | 18 ファイル / 4,501 行 |
| `vnnmap/`, `vnncodegen/` | マッピング・コード生成の Python ラッパ | 1,151 行 |
| `mythic_pkg/irs/l0/*_pb2.py`, `target_spec/*_pb2.py` | ハードウェア仕様 protobuf | — |

### SDK コンテナ由来（`_extracted_sdk/`）

| パス | 内容 |
|---|---|
| `munc_pytorch/noise.py` | 確率的アナログノイズモデル（学習用、STE付きautograd） |
| `munc_bcm/bcm_models/*.py` | BCM(Boreas Compute Model) 6階層の忠実度モデル |
| `munc_bcm/{registry,bcm_layers,bcm_utils}.py` | BCM 基盤（Conv→BCMConv2d 変換等） |
| `munc_monte_carlo/{chip_instance_generator,tolerance}.py`, `munc_cli/monte_carlo.py` | モンテカルロ駆動・NIST 片側許容区間 |
| `_ace_model.py`, `_denali_ace_*_model.py`, `_boreas_ace_model.py` | ACE(アナログ計算エンジン) の nn.Module モデル |
| `conversion_steps.py`, `munc_cli/helpers.py`, `_session.py` | 精度評価の駆動ワークフロー本体 |
| `hw_specs.py` | ハードウェア仕様定数(Boreas/Denali) |

合計約 8,843 行。`mythic-model-zoo/configs/*.yaml` 等の Hydra 設定はコンテナ内で確認したのみで未抽出。

---

## 7. 次のステップ候補

優先度順の候補:

1. **再学習ロジックの解析（最優先候補）**: `mythic-model-zoo/*/train.py` 群（pythia/yolopx/huggingface/bevformer, QAT・蒸留・ACM 変換 `convert_training_to_acm_step`）を調査し `04_retraining.md` を新規作成。SDK コンテナ（`mythic-sdk-ubuntu-24.04:m2000-v26.05.0`）の `munc` / `mythic-model-zoo` が対象。
2. **精度シミュレーションの未解明点の解消**: doc 03 §D に列挙した項目（`hw_model.randomize()` の実パラメータ名、Hydra config の実 YAML、`munc_acm_signoff` バージョン差異の背景等）。`mythic.acm.denali.*` 等の外部参照パッケージの追加解析が必要。
3. **コンパイラバイナリのさらなる解析**: `vnnmap` / `dnn_compiler` の逆アセンブル・動的トレースによる分割アルゴリズムの推定。
4. **実行トレースの取得**: 実際にモデルをコンパイル・PPA・精度評価まで実行し、生成される `.vidir` / `.vci` / `perf_trace_dump.h5` / `metrics.json` の実データで各ドキュメントの記述を検証。
5. **BEVFormer 推論の実機実行（FP32 vs アナログ動画比較 / 再学習→eval_trained）**: 実行環境の調査は完了済み。CAN bus 拡張データが未確認という最大のブロッカーがある。要件・手順・再開チェックリストは [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) を参照。
