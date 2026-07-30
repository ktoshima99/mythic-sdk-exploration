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

### 2.1 ステップ列（`step_order`）と ①②③ の担当区間

`mythic-model-zoo` の全ステップは、1 本の順序付きリスト `step_order` に定義されている（`conversion_steps.py`）。ユーザーの 4 エントリーポイント（§2.5）のうち ①再学習・②精度シミュ・③compiler は、この列の**連続した区間**を担当する:

**(a) M2000 標準フロー（`m2000.yaml`）** — 精度評価は `to_acm` の**前**の `eval_trained` だけ:

```
 to_structural ─▶ to_training ─▶ train │ eval_trained ─▶ summarize_metrics │ to_acm ─▶ create_artifact ─▶ compile
└──────────── ① 再学習 ─────────────┘ └──── ② 精度シミュレーション ────┘ └──────────── ③ compiler ────────────┘
        （出力: trained.onnx）              （出力: metrics.json）              （出力: firmware artifact）
```

**(b) generic フロー** — 精度評価は `to_acm` の**後**の `eval_acm`。`to_acm` が作る BCM(ACM) 表現を評価する:

```
… train ─▶ to_acm ─▶ eval_acm ─▶ create_artifact ─▶ compile
              │         └─ acm.onnx を 6 階層忠実度で評価（出力: metrics.json）… ② 精度シミュ
              └─ trained.onnx(MYTHIC) → acm.onnx(BCM) へ変換 … ③ compiler の前段
```

**重要 —「列に並んでいる＝直列にデータ依存」ではない**。②精度シミュと③compiler は、いずれも同じ元モデルから**分岐する並行枝**であり、②は③の前提ではない（②は評価して metrics を出す末端）。

- **`eval_trained`（フロー a）と `eval_acm`（フロー b）は別ステップ・別位置**。前者は `to_acm` の**前**で `trained.onnx`(MYTHIC) を、後者は `to_acm` の**後**で `acm.onnx`(BCM) を評価する。両者とも出力は metrics.json（＝②精度シミュ）で、firmware は作らない（§3.5 の状態遷移図・doc 03 §3.2/§4.3）。
- `mc_eval_trained`（`eval_trained` のモンテカルロ版）は `trained.onnx` を入力とし、フロー a と同じ位置。`step_order` の必須項ではなく明示指定時に走る。
- ④PPA 推定は `step_order` に含まれない独立 CLI（コンパイル成果物の下流。§2.5）。

### 2.2 データ依存の分岐図

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
- この分岐図は §2.1 の一列 `step_order` を**データ依存の観点で描き直したもの**。`eval_trained`（枝A）と `to_acm`/`create_artifact`（枝B）は互いに依存せず、どちらも `train` の出力 `trained.onnx` を入力とする独立した枝である（doc 03 §3.2 参照）。
- `eval_trained` は step_type=`eval_onnx`（精度シミュ内の1ステップ名）。

このステップ列（`step_order`）を誰がどう起動するか、ユーザーが叩く CLI との関係は次の §2.5 で整理する。

---

## 2.5 オーケストレーション層とユーザー向けツールの関係

§2 のステップ列は `convert_model.py`（正確には `munc.cli.helpers.run_conversion_steps`）という**オーケストレーション層**が回す。一方、GEN2 ユーザーが実際に叩く CLI コマンドは 3 つに分かれる。この節では「ステップ列」と「3 つの CLI」の対応、および各エントリーポイントがオーケストレーション層とどう関係するかを整理する。

### ユーザーが叩く 3 つのエントリーポイント

GEN2 ユーザーが直接実行する CLI は 3 つ（`doc/user-guides/GEN2 User Guide.pdf`, `HOWTO_ppa_exploration_tools.md` §1/§3）:

| # | エントリーポイント（CLI） | 担当ステップ / 機能 | 入力 → 出力 | 実装本体 | doc |
|---|---|---|---|---|---|
| ① | 再学習（`convert_model.py steps=train` 等） | `to_structural`/`to_training`/`train` | データセット + FP32 ONNX → `trained.onnx` | `mythic-model-zoo/*/train.py`（未解析） | 04(未) |
| ② | 精度シミュレーション（`convert_model.py steps=eval_trained`） | `eval_trained`（=`eval_onnx_step`） | `trained.onnx` + 実データ → `metrics.json` | `munc`（SDK コンテナ内で完結） | 03 |
| ③ | コンパイル（`mythic-compiler`） | コンパイル本体 | compiler-ready artifact + `--compiler-config` → firmware artifact | `mythic.model_deployment.rmcr.compile:main()`（→ Compiler コンテナ起動） | 01 |
| ④ | PPA 推定（`mythic-ppa-estimators`） | 性能・電力・面積推定 | コンパイル成果物 → latency/fps/power/area | `perf_analysis.py` / `power_estimator.py`（Compiler コンテナ側） | 02 |

### オーケストレーション層（`convert_model.py`）との関係

**重要な非対称性**: `convert_model.py`（= `run_conversion_steps`）は `step_order` の**任意のステップ**を `steps=...` で選んで実行できる汎用ドライバである。ただし各エントリーポイントとの関係は一様ではない:

- **② 精度シミュレーションは `convert_model.py` からのみ実行できる**。`eval_trained`/`eval_acm`/`mc_eval_trained` は `conversion_steps.py` のステップとしてのみ存在し、専用 CLI は無い。実行は SDK コンテナ内で完結し、Compiler コンテナを呼ばない（doc 03 §1, §9）。
- **③ コンパイルは 2 通りの起動経路がある**——どちらも最終的に**同一のコンパイル本体**（`rmcr/compile.py:main()` → `compile_artifact()`）に収束する:
  1. **`mythic-compiler` を直接叩く**（GEN2 標準の推奨経路）。`--input-artifact`/`--compiler-config`/`--output-artifact` を取る（`rmcr/compile.py` の `rewrite_argv` が `--input-artifact`→`src`, `--output-artifact`→`dest` にマップ, `compile.py:337-342`）。
  2. **`convert_model.py steps=compile` 経由**。この `compile` ステップの実体 `compile_munc_artifact`（`munc_cli/helpers.py:631-651`）は、内部で **`subprocess.run(["mythic-compiler", ...])`** を呼ぶ。つまり `convert_model.py` の compile ステップは `mythic-compiler` の**薄い呼び出しラッパ**であり、両者は等価な結果を出す。
  - 違いは前段の扱いのみ: `mythic-compiler` は**完成済みの compiler-ready artifact** を入力に取る。`convert_model.py` 経由なら `to_acm`→`create_artifact` で **その artifact を生成してから** compile に渡せる（`steps=to_acm,create_artifact,compile`）。
- **④ PPA 推定は `mythic-ppa-estimators` という独立 CLI**。`convert_model.py` の `step_order` には含まれず、コンパイル成果物を入力とする下流ツール（doc 02）。

### 起動関係の図

```
                        ┌─────────────────────────────────────────────────────┐
ユーザー CLI ①②        │ convert_model.py (= run_conversion_steps)             │
  convert_model.py ────▶│   step_order から steps=... で選択して順に実行         │
                        │   ├─ eval_trained/eval_acm/mc_eval  → ② 精度シミュ(SDK内完結) │
                        │   ├─ to_acm / create_artifact       → artifact 生成    │
                        │   └─ compile (=compile_munc_artifact)                 │
                        └───────────────────────────┬─────────────────────────┘
                                                    │ subprocess: mythic-compiler
ユーザー CLI ③          ┌───────────────────────────▼─────────────────────────┐
  mythic-compiler ─────▶│ rmcr/compile.py:main() → compile_artifact()          │
                        │   → dnn_fw_compile → Compiler コンテナ(compilerd-bin) │
                        │   出力: firmware artifact (tar.gz)                     │
                        └───────────────────────────┬─────────────────────────┘
                                                    │ コンパイル成果物
ユーザー CLI ④          ┌───────────────────────────▼─────────────────────────┐
  mythic-ppa-estimators▶│ perf_analysis.py / power_estimator.py (Compiler側)   │
                        │   出力: latency / fps / power / area                  │
                        └───────────────────────────────────────────────────────┘
```

> **要点**: 「オーケストレーション層」＝`convert_model.py`（`run_conversion_steps`）は ①②③ のステップを回せる汎用ドライバ。②は専ら `convert_model.py` から、③は `mythic-compiler` を直接／`convert_model.py steps=compile` 経由（内部で `mythic-compiler` を呼ぶ）の 2 通りで実行でき結果は等価、④は独立 CLI かつコンパイルの下流。

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

### §2 のステップ列と、この状態遷移の関係

§2.1 の `step_order`（8 ステップの一列フロー）と、本節のモデル状態遷移は**別の見方の同じもの**である。対応づけの鍵は、**ステップには「状態を次へ進めるもの」と「ある状態のまま実行するもの」の 2 種類がある**という点:

| step_order のステップ | 状態への作用 | 遷移後の状態 | ①②③ |
|---|---|---|---|
| `to_structural` | **進める** | ORIGINAL → structural | ① |
| `to_training` | **進める** | structural → **MYTHIC** | ① |
| `train` | MYTHIC のまま（重みを学習）| MYTHIC | ① |
| `eval_trained` | MYTHIC を読むだけ（TorchNet 化して評価）| MYTHIC | ② |
| `summarize_metrics` | 状態に触れない（metrics.json 集計）| — | ② |
| `to_acm` | **進める** | MYTHIC → **BCM** (=ACM) | ③ |
| `eval_acm` | BCM を読むだけ（`SwitchBCM` で忠実度切替→評価）| BCM | ② |
| `create_artifact` | **進める** | BCM → COMPILER | ③ |
| `compile` | COMPILER を消費（Compiler コンテナへ）| （SDK 側状態の終端）| ③ |

つまり **状態を進めるのは `to_structural`/`to_training`/`to_acm`/`create_artifact` の 4 つだけ**で、`train`/`eval_trained`/`eval_acm`/`summarize_metrics`/`compile` はいずれも MYTHIC / MYTHIC / **BCM** / — / COMPILER という**同じ状態の上で動く**（読むだけ／重み学習）。§2 のフローがステップ数の割に下図の遷移が 4 本の矢印なのはこのためである。

**②精度シミュの評価ステップは、どのモデル状態を読むかで位置が決まる**:
- `eval_trained` / `mc_eval_trained` … **MYTHIC** を読む → `to_acm` の**前**（フロー a、§2.1）
- `eval_acm` … **BCM** を読む → `to_acm` の**後**（フロー b、§2.1）。`to_acm` が作った BCM を `create_artifact` に渡す前に評価に流用する枝

### SDK コンテナ側のモデル状態（`munc._constants.MODELType`, `_session.py` の状態遷移）

`munc` は `MODELType`（`_constants.py:327`）でモデルの状態を管理し、`_session.py` の `get_*_conversion_ops()` が状態を進める（矢印ラベル＝状態を進める 4 ステップ。`train`/`eval_*` は各状態上で動く別動作なので矢印には現れない）:

```
              ┌── ① 再学習 ──────────────┐         ┌── ③ compiler ─────────────┐
ORIGINAL ──to_structural──▶ (structural) ──to_training──▶ MYTHIC ──to_acm──▶ BCM ──create_artifact──▶ COMPILER ──compile──▶（Compiler コンテナへ）
OriginalModel               中間graph                    MythicModel        BCMModel               CompilerModel
                                                          │  ▲                │
                       ② eval_trained / mc_eval_trained ──┘  │ train           └── ② eval_acm
                          （MYTHIC を評価。to_acm の前）        （重み学習,        （BCM を SwitchBCM で
                                                              MYTHIC のまま）      忠実度切替して評価。to_acm の後）
```

> ②精度シミュには**入口が 2 つ**ある: `to_acm` の**前**で MYTHIC を評価する `eval_trained`（+モンテカルロ版 `mc_eval_trained`）と、`to_acm` の**後**で BCM を評価する `eval_acm`。どちらも下向き/上向きの分岐＝**評価して metrics.json を出す末端の枝**であり、本線（COMPILER→compile）を進めるわけではない。`eval_acm` が読む BCM は `create_artifact`(③) が消費するのと同じ `to_acm` の出力で、同じ中間表現を評価と firmware 生成に分けて使う。

| モデル状態 / 中間表現 | 実体・クラス | 役割 | 使われるステップ |
|---|---|---|---|
| **ORIGINAL** (`OriginalModel`) | 素の FP32 ONNX | 学習/エクスポート直後の入力モデル | `to_onnx` の出力 |
| **structural** | 中間 ONNX グラフ | on/off-chip マーキング（＝実行区分の第 1 層。後述）・定数畳み込み等の構造整理済み | `to_structural` |
| **MYTHIC** (`MythicModel`) | `MythicConv2d`/`MythicLinear` 等の Mythic ノード（`_constants.py:89-94`）を持つ ONNX | **アナログaware 再学習可能**な量子化グラフ（FSR 分解・DSF 学習可能化済み） | `to_training` の出力 → `train` の対象 |
| **BCM** (`BCMModel`) | `BCMConv2d`/`BCMLinear`（`bcm_layers.py`）= **アナログ MAC(`mma_class`) + デジタルデータパス**（doc 03 A.6.1） | 学習済みモデルを**ハードウェア忠実に数値再現**する中間表現。別名 **ACM (Analog Compute Model)** | `to_acm` で生成 → `eval_acm`・`create_artifact` の入力 |
| **COMPILER** (`CompilerModel`) | コンパイラ入力用に整えた ONNX（`compiler_ready_artifact.tar.gz`） | Compiler コンテナへ渡す最終成果物 | `create_artifact` の出力 |
| **TorchNet** | `torch.nn.Module`（`munc._torchnet.TorchNet`, doc 03 A.13） | ONNX（Mythic/BCM ノード含む）を**実行可能な PyTorch モデル**に変換。`make_torch_net()` で構築 | `eval_trained`・推論動画生成の実行基盤 |

> `RETRAIN`/`PTM` も `MODELType` に定義されているが本解析では未確認。

### 実行区分（off-chip / アナログ / デジタル）はどこで決まるか — 3 層構造

上表の **structural** 行にある「on/off-chip マーキング」は、**実行区分の決定の全体ではなく最初の 1 層**である。「どの演算をアナログコア（ACE/Denali）で、どれをデジタル（SALU）で、どれをホスト CPU で実行するか」は、SDK 側の静的マーキング（2 層）とコンパイラの最適化（1 層）に分かれて決まる。**「structural 時点で実行区分が全部決まる」わけでも「全部コンパイラが決める」わけでもない**:

| 層 | 決めること | 誰が / いつ | 実装 | 根拠 |
|---|---|---|---|---|
| **① off-chip / on-chip** | NPU に**載らない**演算（チップ非対応 op・非対応属性）を **off-chip = ホスト CPU 実行**に回す | SDK・`to_structural` | `MarkUnsupportedOpsOffChip`（`_session.py:199`）→ `is_op_type_supported_on_chip()`（`_session_tools.py:404`） | artifact の `off_chip_0`/`off_chip_2` に対応 |
| **② on-chip 内の一部を先にデジタル確定** | depthwise conv を **on-chip だがデジタル（SALU）**とマーク | SDK・ORIGINAL→MYTHIC 変換 | `MarkDepthwiseConvsAsDigital`（`_session.py:255`, `__digital_onchip` 属性付与） | 「depthwise は SALU で処理」 |
| **③ on-chip の残りをアナログ/デジタルへ最終振り分け + SRAM 分割** | Conv/Gemm/MmaDot 等を **Denali(アナログ) / Digital IPU** のどちらに載せるか、SRAM 容量で**パーティション分割** | **コンパイラ**・`dnn_compiler` | `auto_partition.cpp`（`IsDenali()`/`IsDigital()`, doc 01 §3.3） | `off_chip_0→on_chip_1_bcm→off_chip_2` を生成 |

**重要な誤解ポイント**: 「on-chip = アナログ」ではない。on-chip 対応ノード一覧（`SUPPORTED_ON_CHIP_NODES_BOREAS`, `_constants.py:245-272`）には、アナログ MAC 候補（`Conv`/`Gemm`/`BCMConv2d`/`BCMLinear`）と**デジタル演算（`Mul`/`Add`/`Sum`/`MaxPool`/`AveragePool`/`Relu`/`Concat`/`Slice`/`Resize` 等）が最初から混在**している。チップ上には ACE（アナログコア）と SALU/デジタルデータパスの両方があり、どちらも「on-chip」。

- したがって①は「NPU に載るか否か」だけを分け、②③で on-chip 内をアナログ/デジタルに割り振る。
- デジタル専用 op（`MaxPool`/`Relu`/`Add` 等）は最初からデジタル側で、③の「振り分け」対象は主に `Conv`/`Gemm`/`MmaDot` 系（アナログにも載せうる演算）。
- ②の SDK マーキングは③の**入力**になる。コンパイラは②のマーク（`__digital_onchip`・off-chip）を尊重した上で③を最適化する（doc 01 §3.3.3 に `CHECK FAILED: !hw::IsDenali(...depthwise_conv...)` の検証あり）。

要するに、あなたの直感「structural で実行区分が決まる」は**①の層に限れば正しい**が、**アナログ/デジタルの本来の振り分けは③のコンパイラ（`auto_partition`）**が担い、SRAM 物理制約下で最終確定する。

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
