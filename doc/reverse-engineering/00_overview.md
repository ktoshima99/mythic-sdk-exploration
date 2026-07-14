# Mythic M2000 SDK 処理ロジック解析ドキュメント（総合概要・索引）

対象 SDK: **Mythic M2000 SDK v26.05.0**（内部呼称 `vnnsdk 26.05`）
解析対象展開先: `/home/ubuntu/mythic_sdk/26.05/_extracted_compiler/`（コンパイラコンテナ由来）、`/home/ubuntu/mythic_sdk/26.05/_extracted_sdk/`（SDK コンテナ由来）
作成日: 2026-07-14 / 最終更新: 2026-07-14（精度シミュレーション本体の解析完了）

---

## 0. このドキュメント群について

Mythic M2000（フラッシュ／NVM メモリセルをアナログ乗算器として用いる **アナログ compute-in-memory 型 AI アクセラレータ**）SDK の、処理ロジックをソースコードから解析した記録である。

| # | ドキュメント | 対象ステップ | 解析済みか |
|---|---|---|---|
| 01 | [01_compilation.md](01_compilation.md) | **コンパイル** | ✅ 済（コンパイラコンテナ側） |
| 02 | [02_ppa_estimation.md](02_ppa_estimation.md) | **PPA 推定**（性能・電力・面積） | ✅ 済（コンパイラコンテナ側） |
| 03 | [03_accuracy_simulation.md](03_accuracy_simulation.md) | **精度シミュレーション** | ✅ 済（Part A: SDK コンテナ本体 / Part B: Compiler コンテナ部品） |
| 04 | （未作成） | **再学習** | ⛔ 未解析（SDK コンテナ側） |

> ### ⚠️ 重要な訂正の履歴（2026-07-14）
> 初版は**コンパイラコンテナ（`compilerd-bin`）だけ**を解析対象とし、その中身から各ステップを推定していた。その後 **SDK コンテナ（`sdk.tar`, 9.8GB）を `docker load` して中身を確認・解析**し、以下が判明した。
>
> 1. **精度シミュレーションの「本体」は SDK コンテナ側（`munc`パッケージ）にあり、解析済み。** `convert_model.py`→`munc.cli.helpers.run_conversion_steps`→`conversion_steps.py`（`eval_onnx_step`/`eval_acm_step`）が本体で、Compiler コンテナの `vnnort` は評価メトリクス・推論エンジン等の**部品**にすぎなかった。詳細は [03_accuracy_simulation.md](03_accuracy_simulation.md) の Part A。
> 2. **確率的アナログノイズモデルが SDK コンテナに実在した。** `munc/_pytorch/noise.py`（重みプログラミング誤差・温度ドリフト・ADC 熱雑音等）、`munc/bcm/bcm_models/`（6階層の忠実度モデル）、`munc/_monte_carlo/`（モンテカルロ+NIST片側許容区間による保証精度算出）。初版の「確率ノイズ注入は無い」は Compiler コンテナに限った正しい観察だったが、SDK コンテナ側にはノイズモデルが存在する。
> 3. **"BCM" の正体が判明: Boreas Compute Model**（block-circulant matrix ではない）。`munc/bcm/bcm_layers.py` の docstring 等が根拠。Compiler コンテナに存在しないのは、BCM が学習・精度評価専用（SDK コンテナ限定）の概念だったため。
> 4. **再学習も精度シミュレーションも SDK コンテナに両方入っている。** 「SDK コンテナ＝再学習専用」ではない。`train.py` 群（pythia/yolopx/huggingface/bevformer）と精度評価が共存する、GEN2 ガイド通りの「メイン作業環境」だった（再学習ロジック自体は未解析）。
>
> 詳細なコンテナ役割分担は §1 の表を参照。

### 解析手法
- **コンパイラコンテナ**: `compiler_m2000.tar`（OCI イメージ）を `docker load` し、`compilerd-bin:1.5.2` から生 Python ソース約 27,000 行を `_extracted_compiler/` に抽出。doc 01/02、および doc 03 Part B はこれに基づく。
- **SDK コンテナ**: `sdk.tar` を `docker load` し `mythic-sdk-ubuntu-24.04:m2000-v26.05.0` を起動、`munc` パッケージの核心（約 8,843 行: ノイズモデル・BCM モデル階層・モンテカルロ・評価ワークフロー）を `_extracted_sdk/` に抽出して解析。doc 03 Part A はこれに基づく。**再学習ロジック（`train.py` 群）は存在を確認したのみで本文解析は未実施。**
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

```
┌─ 再学習 【S】(mythic-model-zoo/*/train.py, 未解析) ───────────┐
│  データセット + FP32 ONNX  →  学習済 ONNX (data/trained.onnx) │
└──────────────────────────────────┬───────────────────────┘
                                    │
┌─ 精度シミュ【S】本体+【C】部品 [doc 03] ─────────────────────┐
│ eval_trained(=eval_onnx_step) → make_torch_net()で           │
│   BCM/ACEアナログモデル(6階層の忠実度)をforwardに注入          │
│ ノイズ: 重みプログラミング誤差+温度ドリフト+ADC熱雑音等         │
│   (munc/_pytorch/noise.py, munc/bcm/bcm_models/)              │
│ 実データセット(ImageNet/COCO/nuScenes等)でHF Trainer.evaluate()│
│ (モンテカルロ実行時) NIST片側許容区間で保証精度を算出           │
└──────────────────────────────────┬───────────────────────┘
                                    │ eval後 to_acm→create_artifact
┌─ コンパイル [doc 01] ─────────────▼───────────────────────┐
│  学習済ONNX                                                  │
│   → 最適化: 標準ONNXオペ → com.videantis の vidConv へ書換    │
│           (MatMul/Gemm/Attention も Conv=行列積に統一)        │
│   → 量子化: 8bit対称 power-of-two 固定小数点 (max_exponent)   │
│           重みper-channel / bias 16bit / 最終層per-tensor    │
│   → .vidir (CapnProto) エクスポート                          │
│   → dnn_compiler(バイナリ/auto_partition): アナログ/デジタル  │
│           振り分け=IPUパーティション分割(Denali/Digital)      │
│   → vnnmap(バイナリ): ACEタイル配置・BitSpreading・           │
│           NVM書込み計画 → .vci                                │
│   → vnncodegen/vnnrtgen: → .vcnn → ランタイムバイナリ         │
│                                                              │
│  出力物: compiler_ready_artifact/                            │
│    off_chip_0 → on_chip_1_bcm(ACE) → off_chip_2 の3ステージ   │
└──────────────────────────────────┬───────────────────────┘
                                    │
┌─ PPA推定【C】[doc 02] ────────────▼───────────────────────┐
│ 入力: perf_trace_dump.h5 + vnn JSON                          │
│ 性能: ボトルネックmax(ACEクリティカルパス, SRAM r/w, SIMD)     │
│ 電力: 1推論エネルギー×fps / 面積: 1タイル42.16mm²×タイル数     │
│ 出力: latency/fps/power/area                                 │
└──────────────────────────────────────────────────────────────┘
```

凡例: `eval_trained` は精度シミュ内の1ステップ名（step_type=`eval_onnx`）。コンパイルへの入力は精度シミュとは独立に `to_training`→`train` 済みの ONNX からも到達可（両者は `mythic-model-zoo` 内の同じステップ列 `to_structural→to_training→train→eval_trained→...→to_acm→create_artifact→compile` に統合されている、doc 03 Part A.3 参照）。

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

1. **コンパイラの実マッピングは Python 外の C++ バイナリにあるが、原理は判明**（追補調査 2026-07-14 で更新）
   「アナログ/デジタル振り分け」「off_chip/on_chip 分割」「ACE タイル配置」の実装は Python ソースではなくコンパイル済みバイナリ `dnn_compiler`（23MB）/ `vnnmap`（3MB）内にある。ただし `strings` 解析により**振り分けの所在・単位・判定基準は判明済み**:
   - 振り分け本体 = `dnn_compiler` の `mythic/optimizer/high/passes/auto_partition.cpp`
   - 単位 = **IPU パーティション**（`Denali`=アナログ IPU / `Digital`）。判定関数 `IsDenali()` / `IsDigital()`
   - 基準 = ①演算のアナログ実行可否（Conv/Dense/MmaDot はアナログ、DepthwiseConv 等はデジタル）②物理 SRAM 容量（超過で分割）
   - 分割境界 = infeed/outfeed 接続 → artifact の `off_chip_0 → on_chip_1_bcm → off_chip_2`

   **未確定なのは分割点の探索アルゴリズムとコスト関数の内部実装のみ**（C++ 逆アセンブルが必要）。→ 初版の「完全なブラックボックス」表現は撤回。詳細は [01_compilation.md](01_compilation.md) の 3.3 節。なお "BCM" という語は Python ソース・バイナリ双方に存在しない（artifact ステージ名 `on_chip_1_bcm` の "bcm" の由来は不明）。

2. **精度シミュレーションの「本体」は解析済み（SDK コンテナ側）**（2026-07-14 解析完了）
   本体ワークフロー（`convert_model.py`→`munc.cli.helpers.run_conversion_steps`→`conversion_steps.py` の `eval_onnx_step`/`eval_acm_step`）を解析済み。GEN2 ガイドの `eval_trained` は名前に反して `step_type: eval_onnx` で実装され、`torchnet`（`make_analog_model`+`noise_config`）付きで ONNX を評価する——これが実データセット(ImageNet/COCO/nuScenes等)上でアナログノイズ込みの精度を出す仕組み。詳細は [03_accuracy_simulation.md](03_accuracy_simulation.md) Part A。

3. **確率的アナログノイズモデルは SDK コンテナに実在、解析済み**（2026-07-14 解析完了）
   Compiler コンテナ側（`vnnort`）はアナログ挙動の模擬を QDQ の決定論的固定小数点量子化のみで行う——この観察自体は正しい。**SDK コンテナの `munc` パッケージに確率的ノイズモデルが実在**：`_pytorch/noise.py`（重みプログラミング誤差・温度ドリフト・ADC熱雑音・3次歪み）、`bcm/bcm_models/`（6階層の忠実度、`munc_simple`が最も物理的）、`_monte_carlo/`（NIST片側許容区間で保証精度を算出）。QDQ はこのノイズモデルの「全σ=0」極限に相当する決定論的下部構造（詳細は doc 03 Part C）。
   **BCM の正体も判明: Boreas Compute Model**（block-circulant matrix ではない）。Compiler コンテナに存在しないのは学習・精度評価専用の概念だったため。

4. **PPA の電力モデルに未算入の項目**
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

合計約 8,843 行。`mythic-model-zoo/configs/*.yaml` 等の Hydra 設定はコンテナ内で確認したのみで未抽出（解析用コンテナは削除済み）。

---

## 7. 次のステップ候補

優先度順の候補:

1. **再学習ロジックの解析（最優先候補）**: `mythic-model-zoo/*/train.py` 群（pythia/yolopx/huggingface/bevformer, QAT・蒸留・ACM 変換 `convert_training_to_acm_step`）を調査し `04_retraining.md` を新規作成。SDK コンテナは再ロードが必要（解析用コンテナは削除済み、イメージ自体は `docker images` に残存の可能性あり要確認）。
2. **精度シミュレーションの未解明点の解消**: doc 03 §D に列挙した項目（`hw_model.randomize()` の実パラメータ名、Hydra config の実 YAML、`munc_acm_signoff` バージョン差異の背景等）。`mythic.acm.denali.*` 等の外部参照パッケージの追加解析が必要。
3. **コンパイラバイナリのさらなる解析**: `vnnmap` / `dnn_compiler` の逆アセンブル・動的トレースによる分割アルゴリズムの推定。
4. **実行トレースの取得**: 実際にモデルをコンパイル・PPA・精度評価まで実行し、生成される `.vidir` / `.vci` / `perf_trace_dump.h5` / `metrics.json` の実データで各ドキュメントの記述を検証。
5. **BEVFormer 推論の実機実行（FP32 vs アナログ動画比較 / 再学習→eval_trained）**: 実行環境の調査は完了済み。CAN bus 拡張データが未確認という最大のブロッカーがある。要件・手順・再開チェックリストは [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) を参照。
