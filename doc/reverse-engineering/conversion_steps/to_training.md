# 04. `to_training` ステップ解析

Mythic M2000 (Denali/ACE) アナログ compute-in-memory AI アクセラレータ SDK の **`to_training` ステップ**の解析。対象バージョン `26.05.2`（SDK コンテナ `mythic-sdk-ubuntu-24.04:m2000-v26.05.2`, `mythic-model-zoo` の venv 内 `munc` パッケージ）。[to_structural.md](to_structural.md) の直後段にあたる。

主張はすべて実コードの**ファイルパス:行番号**を根拠に引用する。確定できない箇所は **[推測]** と明記する。パスは特記なき限りコンテナ内 `/root/mythic_sdk/v26.05.2/mythic-model-zoo/` からの相対（`munc/...` は実体としては同ディレクトリ下 `.venv/lib/python3.12/site-packages/munc/...` に存在するpipパッケージだが、[to_structural.md](to_structural.md) と表記を揃えて `munc/...` と記す）。抽出ソースの所在は §11 を参照。

**本ドキュメントの範囲**: `step_order` 上の `to_training` ステップ（structural → MYTHIC の変換）のみを対象とする。後続の `train`（重み学習ループ・QAT・蒸留の実行）は範囲外（[00_overview.md](../00_overview.md) §7 の残課題）。

---

## 目次

- [1. `to_training` とは何か（一言で）](#1-to_training-とは何か一言で)
- [2. 入力と出力](#2-入力と出力)
- [3. 位置づけ — `to_structural` との対比](#3-位置づけ--to_structural-との対比)
- [4. ディスパッチ機構 — `to_structural` と真逆の共通実装](#4-ディスパッチ機構--to_structural-と真逆の共通実装)
- [5. `get_original_to_mythic_conversion_ops` — 45個超のopの固定シーケンス](#5-get_original_to_mythic_conversion_ops--45個超のopの固定シーケンス)
- [6. off-chip / depthwise-digital の扱い](#6-off-chip--depthwise-digital-の扱い)
- [7. 「量子化」の実体 — 数値変更されるのはスケールのみ、丸め・ノイズ注入は実行時](#7-量子化の実体--数値変更されるのはスケールのみ丸め・ノイズ注入は実行時)
- [8. モデル横断比較 — 実装は完全共通、config数値だけが違う](#8-モデル横断比較--実装は完全共通config数値だけが違う)
- [9. 設定（config）の全体像](#9-設定configの全体像)
- [10. 実測 — BEVFormer-Tiny 1600x900](#10-実測--bevformer-tiny-1600x900)
- [11. 参照ファイルと未解明点](#11-参照ファイルと未解明点)

---

## 1. `to_training` とは何か（一言で）

> **structural ONNX（標準op構成、[to_structural.md](to_structural.md)）を、アナログaware再学習可能な `MythicConv2d`/`MythicLinear` 等のノードで構成された ONNX に変換する。ハードウェアスケール因子（FSR/DSF）を数値的に確定させるが、8bit power-of-two丸めとアナログノイズの注入は行わない。**

`step_order`（`configs/common/base_config_generic.yaml:37-50`）上で `to_structural` の直後に位置する。[00_overview.md](../00_overview.md) §3.5 の状態遷移表が「structural → to_training → **MYTHIC**」と描く、実際に状態を進める4ステップの1つである。`to_structural` とは対照的に、**`__type` メタデータが実際に書き込まれる**（`self._model.set_meta_data('__type', MODELType.MYTHIC)`, `munc/_session.py:330`）。§10 の実測でこれを直接確認済み。

本ステップの本質は3点:

1. **標準opからMythic opへのノード変換**（`Conv`→`MythicConv2d`、`Gemm`→`MythicLinear`、`Mul`→`MythicQuantizedMul`、`MatMul`→`MythicMatMul`、`Softmax`→`MythicSoftmax`）。ただし off-chip ノードには適用されない（§6）。
2. **ハードウェアスケール因子（iFSR/pFSR/DSF）の数値確定**。重みを実際にリスケールし、ノード属性として `__iFSR`/`__pFSR`/`__multiplier`/`__shift` 等を付与する。重みのdtypeは float32 のまま変わらない（§7）。
3. **学習可能パラメータのマーキング**（`__trainable_dsf`, QATタグ等）。実際の重み学習ループは次段の `train` ステップが担う。

---

## 2. 入力と出力

| 項目 | 実体 | 根拠 |
|---|---|---|
| 入力 (`src`) | structural ONNX（`to_structural` の出力） | `configs/common/base_config_generic.yaml:73` `src: ${structural_model}` |
| 出力 (`dest`) | MYTHIC ONNX | `configs/common/base_config_generic.yaml:74` `dest: ${mythic_model}` |

load/save の機構は `to_structural` と共通（`SessionFromConfig`, `munc/cli/helpers.py:43-121`）。加えて `to_training` は `dataloader`/`torchnet` キーを config に持つ（`base_config_generic.yaml:75-78`）— 統計収集（アクティベーション範囲のクリッピング用）に実データが必要なため。`stat_n_samples_default: 100`, `stat_clipping_percentile: 0.03` が既定値。

---

## 3. 位置づけ — `to_structural` との対比

| 項目 | `to_structural`（[to_structural.md](to_structural.md)） | `to_training`（本ドキュメント） |
|---|---|---|
| `__type` メタデータ | 書かれない（推論により ORIGINAL と判定される） | **書かれる**（`MODELType.MYTHIC`, `_session.py:330`） |
| `model.hwconfig` | 常に `None`（`SessionFromConfig` が `hwconfig` キーを禁止） | **設定される**（`set_hwconfig_metadata`, `_session.py:222-224`。ハードウェア名は `to_training.torchnet.hw_model.hardware_config_name` 経由） |
| モデル間の実装 | 6モデルすべて別実装（§4.2 に詳述） | **1個の共通関数**（§4） |
| off-chip自動判定（`MarkUnsupportedOpsOffChip`） | 実行不可（`hwconfig is None` で `ValueError`） | **実行される**（`hwconfig` 設定後、`_get_process_original_graph_ops` 内, `_session.py:199`） |
| 量子化・アナログノード変換 | 行わない | **行う**（ノード変換は本ドキュメントの主題。ただし丸め・ノイズ注入は実行時, §7） |

`assert_model_is(self._model, MODELType.ORIGINAL)`（`get_original_to_mythic_conversion_ops` 冒頭, `_session.py:216` 相当の呼び出し）は、[to_structural.md](to_structural.md) §3 で述べた「structural は `__type` 未設定のため ORIGINAL と推論される」という事実に依拠して成立している。すなわち **`to_training` が読める入力は「ORIGINAL相当」と判定されるファイルのみ**であり、structural ONNX はこの条件を満たすからこそ入力として通る。

---

## 4. ディスパッチ機構 — `to_structural` と真逆の共通実装

### 4.1 munc 側の共通実装が主役

[to_structural.md](to_structural.md) §4.2 は「`to_structural` は munc 側に共通実装が無く、6モデルの束縛が互いに無関係」と結論した。**`to_training` はこれと正反対**である。

`configs/common/step_types/common.yaml` に明記:

```yaml
to_training: mythic.model_zoo.common.conversion_steps.to_training_step
```

`step_types/main.yaml` で `to_training` を上書きしているモデルは **yolov8 のみ**（全6モデル中5モデルはこの共通実装をそのまま使う）:

```yaml
# configs/yolov8/training/step_types/main.yaml:8
to_training: mythic.model_zoo.yolov8.conversion_steps.to_training
```

```python
# mythic/model_zoo/yolov8/conversion_steps.py:127-128
def to_training(config):
    to_training_step(add_default_image_size(config))
```

`add_default_image_size`（同ファイル内）は config に既定の画像サイズを補完するだけの前処理で、実変換ロジックは同じ `to_training_step` に委譲される。つまり **6モデル全てが最終的に同一の `to_training_step`（`mythic/model_zoo/common/conversion_steps.py:175-207`）を実行する**。

### 4.2 `to_training_step` の実体

```python
# mythic/model_zoo/common/conversion_steps.py:175-207（要約）
def to_training_step(config):
    hardware_config_name = config.torchnet.hw_model.hardware_config_name

    def convert(sess):
        ops = sess.get_original_to_mythic_conversion_ops(
            hardware_config_name=hardware_config_name,
            **config.conversion_parameters.options)
        sess.run_ops(*configure_model_conversion_ops(ops, config.conversion_parameters.ops))

    run_conversion(config, convert)
```

`run_conversion`（`common/conversion_steps.py:143-166`）が `dataloader` config から実データローダを構築し `SessionFromConfig` を開く。実変換の中身は `Session.get_original_to_mythic_conversion_ops()`（`munc/_session.py:215-332`）が返す**固定の op シーケンス**であり、モデル側が制御できるのは:

- `hardware_config_name`（`Boreas`/`Denali`。§9.3）
- `config.conversion_parameters.options`（関数のキーワード引数 `scale_offchip_nodes`/`scale_concat_inputs`/`optimize_wsf`）
- `config.conversion_parameters.ops`（個々のopの `enabled`/パラメータを `configure_model_conversion_ops` 経由で上書き。§8）

の3点のみ。**opシーケンス自体の構成・順序を変えることはできない**（`to_structural` の6実装が処理内容そのものを自由に組んでいたのとは対照的）。

---

## 5. `get_original_to_mythic_conversion_ops` — 45個超のopの固定シーケンス

`munc/_session.py:215-332`。全体は `op_conf_seq(...)` で1本につながれた munc `ops.*` の列（`munc/ops/` ディレクトリに個別ファイルとして実装、全体で100種類超あるopカタログのうち約45個をこのシーケンスで使用）。役割ごとに分類する。

### 5.1 前段: 標準グラフ整備（`_get_process_original_graph_ops` の再利用）

`_session.py:184-206`。`to_structural` の後にもう一段、**同じグラフ整備関数**が走る（[to_structural.md](to_structural.md) §5.2 で予告した「op種別による自動off-chip判定はここで動く」の箇所）:

```python
# munc/_session.py:190-206
return op_conf_seq(
    *self.get_change_opset_ops(),      # opset変更＋SanityCheckOffChipMarking等（:462-481）
    ops.RemoveDanglingNodes,
    ops.RemoveShapeInferenceNodes,
    ops.GeneralizeBatchSize,
    ops.PostBatchNormFolding,
    ops.PreBatchNormFolding,
    ops.InferStoreTensorShapes,
    ops.MarkUnsupportedOpsOffChip,      # ← §6.1 で詳述。ここで初めて実行可能になる
    ops.AddNetworkOutputsForIntermediateNodes,
    ops.FixDefaultResizeROI,
    ops.RenameNodesAndEdges,
    op_do(self.stats.reset_stats),
    op_do(check_model),                 # onnx.checker.check_model(full_check=True)
)
```

`to_structural` の各モデル実装がすでに opset 変更・BN folding・shape 推論を行っていることが多いため（[to_structural.md](to_structural.md) §8 比較表）、ここでの多くのopは冪等的に「何もしない」か軽微な差分になる。ただし **`MarkUnsupportedOpsOffChip` はここで初めて有効になる**唯一のop（`hwconfig` が未設定だと `to_structural` 内では `ValueError` になっていたもの、[to_structural.md](to_structural.md) §5.2）。

### 5.2 サインネス・入力シフト・スケーリング準備（`_session.py:230-260`付近）

```
ConvertMatMulToGemm, ReplaceSiluPatternWithNode, CloneConvWeights,
MarkSignedNodes(hwconfig.signed),
AddInputShifting, AddInputScaling, AddOutputScaling,
MoveLastBiasOffChip(既定disabled),
AddOnOffChipTransitionScaling(既定 trainable=not scale_offchip_nodes),
MarkSignedNodes(再実行),
MarkQATNodes(self.qat),
MarkDepthwiseConvsAsDigital,           # ← §6.2
InsertReluForPositiveValues, MoveReLUBeforeMaxPoolOp,
SplitInputs(split_bias_fp=True),
FuseAddToSum(既定disabled),
InjectInputClippings,
ConvGemmWeightScaling,                 # ← 重みの実数値スケーリング。§7
RandomizeWeightsAndBiases(既定disabled),
```

`MarkSignedNodes`（`munc/ops/mark_signed_nodes.py`）は Conv/Gemm/Add/Sum/Mul を対象に、統計収集済みの最小値を見て「負の入力を受け取るノードか」を判定し `__signed`/`__mix_signed` 属性を付与する。`AddInputShifting`（`munc/ops/add_input_shifting.py`）はこの判定結果を使い、非signedのConv/Gemmの手前でバイアスシフトを行う（アナログMACが符号なし8bit入力を前提とするための補正）。

### 5.3 スケーリングノードの注入（`_session.py:262-282`付近）

```
InjectScalingOnMulInputs, InjectScalingOnMatMulInputs, InjectScalingOnSoftmaxInputs,
InjectScalingOnAddOutput, InjectScalingOnMatMulOutput, InjectScalingOnMulOutput, InjectScalingOnSoftmaxOutput,
InjectScalingOnAddInputs,
InjectScalingOnConcatInputs(既定 scale_concat_inputs=False で無効),
EqualizeConcatInputs(同上),
```

コメント（`_session.py` 内）曰く、前半4個（Add/MatMul/Mul/Softmax output）は「Mythicノードに吸収されるため必ず必要」、後半（Add input等）は「吸収されないため既存スケーリングノードの有無をチェックしてから注入」という2群に分かれる。

### 5.4 FSR/DSF分解（`_session.py:290-296`）

```
BreakCompositeScaleIntoFSRAndDigitalScales,
ScaleAllNodes(scale_offchip_nodes=scale_offchip_nodes),
BreakFSRIntoPFSRAndIFSR(clip_weights=not optimize_wsf),
RemoveMulByOne(既定 [1.0, -1.0] で無効化),
BreakDigitalScalesIntoFactors(break_into_attrs=True),
```

- `BreakCompositeScaleIntoFSRAndDigitalScales`（`munc/ops/break_composite_scale_into_FSR_and_digital_scales.py:7-15`）: 「複合スケール因子(CSF)をFSRとデジタルスケールに分割する処理群の最初の1手」。`MYTHICType.COMPOSITE_SCALE` ノードの後に値1.0の `Mul(Digital Scale)` を挿入する。
- `ScaleAllNodes`（`munc/ops/scale_all_nodes.py`）: グラフをスキャンしてスケーリングノード／打ち消しノードのグルーピングを行う（詳細実装は本解析で未深掘り、§11）。
- `BreakFSRIntoPFSRAndIFSR`（`munc/ops/break_FSR_into_pFSR_and_iFSR.py:12-24`）: 「ヒューリスティックによりFSRをACEハードウェアスケール因子(WSF・iFSR・pFSR)の集合に分割する。`CSF = (DSF * WSF * pFSR) / iFSR`。重み・バイアス・（必要なら）DSFを更新し、クリッピングを回避する」。**この op が重みを数値的に書き換える**（§7）。

### 5.5 Mythicノード変換本体（`_session.py:298-317`）

```
AutoNameNodes, ConvertUnsupportedToSupportedActivations,
InferStoreTensorShapes,
CreateActivationCompensation, AbsorbActivationShift,
InjectMulClipping,
GroupMMAOps(group_activations=True),
GroupAddOutputOps(group_mul_nodes=False),
GroupSoftmaxScalingNodes, GroupMulMatMulScalingNodes,
ConvertNodesToMythic,                  # ← 本体。§6.3
GroupSumQMul,
AdjustMythicSumActivation,
```

`GroupMMAOps`（`munc/ops/group_mma_ops.py:11-19`）が「Conv/Gemmに付随する活性化・スケーリングノードを吸収し、Compiler-Ready表現（CRM）相当のグラフに単純化する」役割を担い、その直後に `ConvertNodesToMythic` が標準op名を書き換える。`GroupSumQMul` が Add/Sum と量子化Mulをグループ化して `MythicSum` に変換する（§10.1 の実測で確認）。

### 5.6 後処理・学習可能化（`_session.py:320-330`）

```
RenormalizeOffchipNodes,
OptimizeWSF(既定disabled), ReduceADCClipping(既定disabled), OptimizeIFSR(既定disabled),
MakeDSFsTrainable,
PinLastOnChipConvDSF(既定disabled), PinLastOnChipMul(既定disabled),
set_meta_data('__type', MODELType.MYTHIC)   # :330
```

`RenormalizeOffchipNodes`（`munc/ops/renormalize_offchip_nodes.py:9-15`）は「学習可能なoff-chipノードの重みをon-chipノードと同程度のスケールに再正規化する」— off-chipとon-chipの学習率を実質的に均等化するための数値操作であり、Mythic変換の対象外（off-chip）ノードにも及ぶ数値変更である。

`MakeDSFsTrainable`（`munc/ops/make_dsfs_trainable.py:8-13`）が `MythicSum`/`MythicQuantizedMul` ノードの乗数・バイアスに `__trainable` マスクを付与し、`train` ステップでの勾配更新対象を確定する。

---

## 6. off-chip / depthwise-digital の扱い

### 6.1 `MarkUnsupportedOpsOffChip` — 自動off-chip判定がここで初めて動く

[to_structural.md](to_structural.md) §5.2 の予告どおり、`to_structural` 実行時は `hwconfig is None` のため `ValueError` になっていたこの op（`munc/ops/mark_unsupported_ops_off_chip.py:22-37`）が、`to_training` の `_get_process_original_graph_ops`（§5.1、`_session.py:199`）内で実行される。`set_hwconfig_metadata`（`_session.py:222-224`、`get_original_to_mythic_conversion_ops` 冒頭で呼ばれる）が `self.model.hwconfig = hwconfig` を設定した**後**に走るため、`ValueError` は発生しない。

判定基準はハードウェア対応op一覧（`SUPPORTED_ON_CHIP_NODES_BOREAS`/`_DENALI`, `munc/_constants.py:245-304`）とノード属性の適合性（Conv/Gemm/MaxPool/Slice/Resize等の形状条件）。[to_structural.md](to_structural.md) が整理した「off-chip確定の2段階」（① `to_structural` の手動宣言、② `to_training` の自動判定）の②がこれに当たる。

### 6.2 `MarkDepthwiseConvsAsDigital` — SALU（デジタル）扱いの確定

`munc/ops/mark_depthwise_convs_as_digital.py:9-14`:

> Depthwiseな Conv（`group == 出力チャネル数` かつ `入力チャネル数 == 1`）に `__digital_onchip` 属性を付与する。Mythicチップ上ではこれらはSALU、すなわちデジタルで処理される。

実装（:29-37）は `weight.shape[0] == group and weight.shape[1] == 1` で depthwise を判定し、真の場合のみ属性を付与する（誤判定時は警告ログのみで属性は付けない）。[00_overview.md](../00_overview.md) §3.5 レベルAの表で言及されている「on-chip・デジタル（SALU）」区分の確定はここで行われる。**`__digital_onchip` の付いたConvはこの後 `ConvertNodesToMythic` の対象にも残る**（§6.3 の対象op一覧に `Conv` は含まれるが、`__digital_onchip` 属性の有無でMythic化を除外するロジックは `ConvertNodesToMythic` 自体には無い。デジタル固定化の実効果は後段 `to_acm`/`SwitchBCM` 側、[00_overview.md](../00_overview.md) §3.5 レベルAの表の出典 `convert_convs_to_bcm.py:58-63`, `switch_bcm.py:40-45` を参照）。

### 6.3 `ConvertNodesToMythic` — off-chipノードは対象外（`OFFCHIP_IGNORE`）

`munc/ops/convert_nodes_to_mythic.py`（全文40行）。マッピング表:

```python
DEFAULT_MYTHIC_NODE_MAP = {
    ONNXType.CONV: ONNXType.MYTHIC_CONV,          # "Conv" -> "MythicConv2d"
    ONNXType.GEMM: ONNXType.MYTHIC_LINEAR,         # "Gemm" -> "MythicLinear"
    ONNXType.MUL: ONNXType.MYTHIC_QUANTIZED_MUL,   # "Mul" -> "MythicQuantizedMul"
    ONNXType.SOFTMAX: ONNXType.MYTHIC_SOFTMAX,     # "Softmax" -> "MythicSoftmax"
    ONNXType.MATMUL: ONNXType.MYTHIC_MATMUL,       # "MatMul" -> "MythicMatMul"
}
```

`_get_info()` が `'off_chip': _constants.OFFCHIP_IGNORE` を返す（`munc/_constants.py:198`）。これは munc のopディスパッチ機構における「off-chipノードにはこのopのパターンマッチを適用しない」という設定であり、[to_structural.md](to_structural.md) が主張した「off-chipに宣言されたノードはMythic化されない」を実装レベルで裏付ける。`_run()` は**ノードオブジェクトそのものを書き換える**（新規ノード生成ではない）: `node.op_type = self.mythic_node_map[node.op_type]` に加え `__trainable_dsf` 属性を付与するのみ。ONNXの `domain` は変更されない（`ONNXType.MYTHIC_CONV = "MythicConv2d"` は標準ドメイン内の非標準op名であり、pythiaの `to_structural`（[to_structural.md](to_structural.md) §7.3）が明示的に `mythic` ドメインopset v1を追加したのとは異なる方式）。この非標準op名を含むグラフに対して、変換後に `onnx.checker.check_model` は呼ばれない（§5.1のcheckerはMythic変換より**前**の `_get_process_original_graph_ops` 内でのみ実行される）。

---

## 7. 「量子化」の実体 — 数値変更されるのはスケールのみ、丸め・ノイズ注入は実行時

[00_overview.md](../00_overview.md) の記述（「MYTHIC = アナログaware再学習可能な量子化グラフ（FSR分解・DSF学習可能化済み）」）を実測で裏付けると、**「量子化」には2つの異なる意味が混在している**ことが分かる:

1. **スケール因子の数値確定**（`to_training` が実行）: `ConvGemmWeightScaling`（`munc/ops/conv_gemm_weight_scaling.py:6-18`、重み・バイアスをハードウェア対応範囲にスケーリング）と `BreakFSRIntoPFSRAndIFSR`（§5.4）が、Conv/Gemmの重みを**実際にnumpy配列レベルで書き換える**。§10.1 の実測で、変換後もMythicノードの重みinitializerは **float32のまま**（int8化されていない）ことを確認済み。代わりにノード属性として `__pFSR`/`__iFSR`/`__multiplier`/`__shift`/`__activation`/`__activation_clip`/`__trainable_dsf` が付与される（§10.1）。
2. **8bit power-of-two丸め・アナログノイズ注入**（`to_training` は行わない）: [03_accuracy_simulation.md](../03_accuracy_simulation.md) が詳述する確率的ノイズモデル（`munc_pytorch/noise.py`）や `MythicConv2d` の `nn.Module` 実装（TorchNet化後、forward時にfake-quantを適用）は、ONNXファイルの静的な変換としては存在しない。これらは `train`/`eval_trained` が ONNX を `make_torch_net()`（[03_accuracy_simulation.md](../03_accuracy_simulation.md) §4.1）でPyTorch化した**実行時**に、`__pFSR`/`__iFSR` 等の属性値を読んでforward pass内で適用される。

**含意**: `to_training` の出力ONNXは「ハードウェアスケール因子は確定済みだが、重み自体はまだFP32連続値」という中間状態であり、実際のアナログ挙動模擬（丸め誤差・熱ノイズ等）はこのファイルを読み込む側（TorchNet）の責務である。[to_structural.md](to_structural.md) が「`to_structural` は量子化を行わない」と結論したのと対比すると、`to_training` は**量子化の「対応表（スケール因子）」を確定する**が、**量子化の「実行（丸め・ノイズ）」はしない**、という中間的な位置づけになる。

---

## 8. モデル横断比較 — 実装は完全共通、config数値だけが違う

[to_structural.md](to_structural.md) §8 の「モデル横断比較表」は実装内容そのものが6モデルで全く異なることを示した。`to_training` については**opシーケンスは全モデル共通**（§4）なので、比較すべきは各モデルの `model_setup.conversion_to_training` config が上書きするパラメータのみである。

| モデル | config ファイル | `options` の上書き | `ops` の主な上書き |
|---|---|---|---|
| 共通既定（generic） | `configs/common/model_setup/generic.yaml:9-16` | `scale_concat_inputs: true`, `optimize_wsf: false` | `BreakFSRIntoPFSRAndIFSR.half_pFSR_arr/half_iFSR_arr` を `noise_config` から注入 |
| huggingface_classifiers (resnet50) | `.../resnet50_imagenet.yaml:16-30` | `scale_offchip_nodes: True`, `scale_concat_inputs: False` | `InjectScalingOnAddInputs.enabled=False`（レイテンシ改善）, `PinLastOnChipConvDSF/PinLastOnChipMul.enabled=False` |
| huggingface_classifiers m2000上書き | `.../resnet50_imagenet_m2000.yaml:6-11` | （継承） | `BreakFSRIntoPFSRAndIFSR.max_dsf=4.0, half_iFSR_arr=[10.0], half_pFSR_arr=[2.5]` |
| huggingface_robot_hand | `.../robot_hand.yaml:26-41` | resnet50と同型 | resnet50と同型 |
| pythia | `configs/pythia/model_setup/pythia.yaml:4-16` | `{}`（関数既定値のまま） | `AddInputScaling.enabled=False`, `GeneralizeBatchSize.enabled=False`, `RemoveShapeInferenceNodes.enabled=False`, `BreakFSRIntoPFSRAndIFSR.max_dsf=4.0` |
| yolopx | `configs/yolopx/model_setup/yolopx.yaml:3-11` | `{}` | `AddOutputScaling.enabled=false`, `BreakFSRIntoPFSRAndIFSR.max_dsf=4.0` |
| yolov8 (yolov8s) | `.../yolov8s.yaml:13-16` | （継承） | `AddOnOffChipTransitionScaling.trainable=true` |
| yolov8 m2000上書き | `.../yolov8s-m2000.yaml:9-14` | （継承） | `BreakFSRIntoPFSRAndIFSR.max_dsf=4.0, half_iFSR_arr=[10.0], half_pFSR_arr=[2.5]` |
| zero_dce | `configs/zero_dce/model_setup/zero_dce.yaml:18-29` | （継承） | `PinLastOnChipConvDSF/PinLastOnChipMul.enabled=true`, `AddOnOffChipTransitionScaling.trainable=true` |
| bevformer | `configs/bevformer/bevformer_tiny.yaml:114-131` | `to_training:` ブロックを直接上書き（`model_setup.conversion_to_training` 経由ではない） | `GeneralizeBatchSize.enabled=false`（コメント: 「batch sizeが常に先頭次元ではないため失敗する」）, `BreakFSRIntoPFSRAndIFSR(half_iFSR_arr=[10], half_pFSR_arr=[2.5], max_dsf=3)`, `AddOnOffChipTransitionScaling.trainable=true` |

**繰り返し現れるパターン**: `BreakFSRIntoPFSRAndIFSR` の `half_iFSR_arr`/`half_pFSR_arr`/`max_dsf` は resnet50(m2000)・pythia・yolov8(m2000)・zero_dce(m2000) の4箇所で**ほぼ同一の値**（`half_iFSR_arr=[10.0], half_pFSR_arr=[2.5], max_dsf=4.0`、bevformerのみ `max_dsf=3`）に固定されている。これはノイズモデル（`noise_config`）とハードウェアの実測特性に由来する共通ヒューリスティックと考えられる[推測]。`AddOnOffChipTransitionScaling.trainable=true` も yolov8s/zero_dce/bevformerの3モデルで共通して有効化されている。

**huggingface_classifiers/robot_hand特有の点**: `off_chip_layers` キーが `conversion_to_training` 名前空間の下にあるが、これは[to_structural.md](to_structural.md) §9.3 で述べた通り**`to_structural` 側が読む**キーであり（`to_structural.conversion_parameters: ${model_setup.conversion_to_training}` という間接参照経由）、`to_training_step` 自身は `config.conversion_parameters.options`/`.ops` しか読まないため `off_chip_layers` は無視される（未知キーとして単に読まれないだけで、エラーにはならない）。**同じconfig名前空間 `conversion_to_training` を `to_structural` と `to_training` の双方が異なる部分集合として共有する**、という設計になっている。

---

## 9. 設定（config）の全体像

### 9.1 generic既定値

`configs/common/base_config_generic.yaml:72-85`:

```yaml
to_training:
  src: ${structural_model}
  dest: ${mythic_model}
  torchnet: ${default_torchnet}
  dataloader: ${conversion_dataloader}
  stat_n_samples_default: 100
  stat_clipping_percentile: 0.03
  conversion_parameters: ${model_setup.conversion_to_training}
  debug: ${oc.decode:${oc.env:DEBUG_N_SAMPLES,100}}
```

### 9.2 `hardware_config_name` の出自 — モデルリポジトリではなく munc パッケージ側

`configs/` 以下（model-zoo リポジトリ全体）を `hardware_config_name`/`Denali`/`Boreas` で grep しても**1件もヒットしない**。実際の値は `munc` パッケージが自前で持つ Hydra structured config（`munc/hydra_configs/training_model/{boreas,denali,denali_ref,...}.yaml`）から来る:

```yaml
# munc/hydra_configs/training_model/denali.yaml（抜粋）
_target_: munc._denali_ace_separable_model.make_denali_separable_model
noise_config: ${noise_config}
hardware_config_name: Denali
name: denali
```

`default_torchnet`（`base_config_generic.yaml:7` `- torchnet@default_torchnet: default` というHydra defaultsグループ経由）が最終的にこの `hw_model` 設定を参照する。**どのモデルが `Boreas` を使い、どのモデルが `Denali` を使うかを決める設定ファイルはmodel-zoo側のconfigsディレクトリには存在せず、Hydra defaultsリストの解決順序に依存する**[推測: BEVFormer は §10 の実測で `Denali` を使用していることを直接確認したが、他モデルの既定値は本解析では未確認]。

### 9.3 `conversion_parameters` の2系統

- `options`: `get_original_to_mythic_conversion_ops` のキーワード引数（`scale_offchip_nodes`/`scale_concat_inputs`/`optimize_wsf`）にそのまま展開される。**未指定キーはエラーになる**（`to_training_step` が `**config.conversion_parameters.options` と無条件展開するため、`options` キー自体が欠けていると `ConfigAttributeError` になる。pythia/yolopxが明示的に `options: {}` を書いているのはこのため）。
- `ops`: `configure_model_conversion_ops`（`munc/cli/helpers.py:552-572`）経由で、op名をキーとした辞書で個々のopの `enabled` やパラメータを上書きする。§5で列挙した約45個のopのうち、config で明示的に触れられていないものは全モデル共通のデフォルト動作のまま実行される。

---

## 10. 実測 — BEVFormer-Tiny 1600x900

### 10.1 実行環境と成果物

[to_structural.md](to_structural.md) §10.1 と**同じ実測ホスト**の `/mnt/nvme_scratch/mythic_untrained_probe/` に、`to_structural` の出力（`structural-1600x900.onnx`）を入力として `steps=to_training` を単体実行した際の成果物が実在する。Hydra起動時オーバーライド（`outputs/2026-08-04/10-23-47/.hydra/overrides.yaml`）:

```yaml
- steps=to_training
- data_dir=/workspace/untrained_probe
- model_setup.mmcv_config=mythic/model_zoo/bevformer/bevformer_lib/projects/configs/bevformer/bevformer_tiny_nuscenes_mini.py
- to_training.dest=/workspace/untrained_probe/mythic-1600x900-untrained.onnx
- to_training.stat_n_samples_default=20
- conversion_dataloader.workers_per_gpu=0
- ++to_training.device_name=cpu
```

| ファイル | サイズ | 所要時間 |
|---|---|---|
| `structural-1600x900.onnx`（入力） | 139,607,407 B | — |
| `mythic-1600x900-untrained.onnx`（出力） | 148,810,703 B | 2026-08-04 10:23:43 起動 → 11:05:24 保存、**約42分**（CPU実行。`to_training.log`〜`to_training_gpu2.log` に残る過去4回の失敗試行を除いた最終成功ログ `to_training_final.log` の実測） |

`to_training_final.log` の進行は §5 で整理したop列と完全に一致する順序で進む（"Marking unsupported off-chip layers..." → "Converting MatMul into Gemm..." → ... → "Converting convs and gemms to Mythic nodes..." → "Making DSF nodes trainable..." → 保存）。特に統計収集（`Collecting required stats...`、20サンプル）が複数回（`MarkSignedNodes`, `ConvGemmWeightScaling`, `EqualizeConcatInputs`, `ScaleAllNodes`, `ConvertNodesToMythic` 直前の `Switch from hardtanh to ReLU` 相当箇所）で繰り返し実行され、うち `ScaleAllNodes` 直後の統計収集1回で**約15分**（1回あたり約45秒 × 20サンプル）を要しており、全体の処理時間の大半を占める。

### 10.2 `onnx.load` による直接比較

```python
import onnx
s = onnx.load("structural-1600x900.onnx")
m = onnx.load("mythic-1600x900-untrained.onnx")
```

| 項目 | structural | mythic（未学習） |
|---|---|---|
| ノード数 | 2,164 | 2,178 |
| initializer数 | 1,851 | 2,232 |
| `__off_chip` 付きノード数 | 1,990 | 2,026 |
| `metadata_props` | 空 | **`hardware_config: Denali`, `__type: MythicModel`**（§3の主張を直接裏付け） |
| opset_import | `[('',20),('',20)]` | `[('',20),('',20)]`（不変） |
| graph 入出力名 | `img,can_bus,lidar2img,prev_bev,use_prev_bev → bev_embed,outputs_classes,outputs_coords` | 完全同一 |

op種別の差分（非ゼロのみ）:

| op_type | structural | mythic | 差分 |
|---|---|---|---|
| Conv | 55 | **0** | −55 |
| BatchNormalization | 53 | **0** | −53 |
| MythicConv2d | 0 | **76** | +76 |
| MythicQuantizedMul | 0 | **12** | +12 |
| MythicSum | 0 | **29** | +29 |
| Relu | 96 | 47 | −49 |
| Add | 272 | 263 | −9 |
| Mul | 165 | 193 | +28 |
| Slice | 116 | 150 | +34 |
| Clip | 2 | 3 | +1 |

**[to_structural.md](to_structural.md) §10.1 との整合性チェック**: doc06 はBEVFormer-Tinyのon-chip側（ResNet backbone）の内訳を「Conv55 / BatchNormalization53 / Relu49 / Add16 / MaxPool1 = 174ノード」、transformer側は `everything_off_chip` で全ノードoff-chipと報告していた。今回の実測で `Conv: 55→0` と `BatchNormalization: 53→0` が**完全に一致**しており、on-chip側のConv・BNが**すべて**`ConvertNodesToMythic`（Conv）と `PostBatchNormFolding`/`PreBatchNormFolding`（BN→Convへ吸収後にMythic化）によって処理されたことが確認できる。新規に現れた `MythicConv2d`(76) が元のon-chip Conv数(55)より多いのは、`CloneConvWeights` によるConv複製や、BN吸収で生成された新規Conv相当ノードがすべてMythic化された結果と考えられる[推測]。同様に `MythicQuantizedMul`(12)・`MythicSum`(29) は、on-chip側にはstructural時点でMul/Sum/Add-groupノードが存在しなかった（doc06実測の174ノード内訳にMul/Sumは無い）ことから、**`to_training` が新規に注入したスケーリング・活性化補正ノード**（§5.3のInjectScalingOn系、§5.2のAddInputShifting系等）のうちon-chip側に属するものが `ConvertNodesToMythic`/`GroupSumQMul` でMythic化された結果である[推測]。

### 10.3 重みは float32 のまま、スケール因子が属性として付与される（§7の裏付け）

代表ノード `n_ResNet_conv1_Conv`（`MythicConv2d`）の実際の属性・入力:

```
attr: __mythic_type, __pFSR, __iFSR, __multiplier, __shift, __activation, __activation_clip, __trainable_dsf
input model_ResNet_conv1_weight  dtype=float32  shape=(64, 3, 7, 7)
input new285                      dtype=float32  shape=(64,)   # bias
```

重みinitializerのdtypeは変換前後で float32 のまま変化がない。§7で述べた「量子化の対応表（スケール因子）は確定するが実行（丸め・ノイズ注入）はしない」という結論を直接裏付ける実測結果である。

### 10.4 未学習MYTHICモデルの評価結果（参考）

この `mythic-1600x900-untrained.onnx` に対して `steps=eval_trained` を実行した結果（`metrics_untrained.json`, 2026-08-04 11:14）は、`car_AP_dist_*` を含む全検出クラスのAPが **0.0**（`car_trans_err=1.2803` 等、誤差指標も大きい）。これは当然の結果である—`to_training` は重みのスケール変換のみを行い、実際のQAT（アナログaware再学習）は次段の `train` ステップの責務であるため、`to_training` の出力を学習なしで評価すれば性能が崩壊するのは想定通りである。この一点は「`to_training` は学習を行わない」ことの動作面での確認材料として記録しておく。

---

## 11. 参照ファイルと未解明点

### 抽出ソースの所在

| 分類 | ファイル |
|---|---|
| ディスパッチ | `mythic/model_zoo/common/conversion_steps.py`（`to_training_step`）、`configs/common/step_types/common.yaml` |
| 変換op列本体 | `munc/_session.py`（`get_original_to_mythic_conversion_ops`, `_get_process_original_graph_ops`） |
| off-chip/depthwise確定 | `munc/ops/mark_unsupported_ops_off_chip.py`, `munc/ops/mark_depthwise_convs_as_digital.py` |
| Mythicノード変換本体 | `munc/ops/convert_nodes_to_mythic.py` |
| スケール因子分解 | `munc/ops/break_composite_scale_into_FSR_and_digital_scales.py`, `munc/ops/break_FSR_into_pFSR_and_iFSR.py`, `munc/ops/scale_all_nodes.py`, `munc/ops/conv_gemm_weight_scaling.py` |
| 学習可能化 | `munc/ops/make_dsfs_trainable.py`, `munc/ops/mark_qat_nodes.py` |
| 定数・型定義 | `munc/_constants.py`（`ONNXType.MYTHIC_*`, `MODELType`, `OFFCHIP_IGNORE`, `SUPPORTED_ON_CHIP_NODES_BOREAS/DENALI`） |
| ハードウェア設定の出自 | `munc/hydra_configs/training_model/{boreas,denali}.yaml` |
| モデル別config | `configs/<model>/model_setup/*.yaml` の `conversion_to_training` ブロック |
| 実測データ | ホスト `/mnt/nvme_scratch/mythic_untrained_probe/`（`structural-1600x900.onnx`, `mythic-1600x900-untrained.onnx`, `to_training_final.log`, `metrics_untrained.json`） |

### 未解明点

1. `ScaleAllNodes`（`munc/ops/scale_all_nodes.py`）の内部アルゴリズム（`MODE_ALL_NODES`/`MODE_ALL_PATHS` の使い分け、`_scan_node`/`_group_adjacent_edges` によるスケーリングノードのグルーピング手法）は本解析では表層のdocstringのみ確認し、深掘りしていない。
2. `hardware_config_name` の既定値がモデルごとに `Boreas`/`Denali` のどちらに解決されるかは、BEVFormer（実測で `Denali` 確認）以外のモデルについて未確認。Hydra defaultsリストの解決順序（`torchnet@default_torchnet: default` が指す実体）を追う必要がある。
3. `GroupMMAOps`/`GroupAddOutputOps`/`GroupSoftmaxScalingNodes`/`GroupMulMatMulScalingNodes` の4つのグルーピングopが `ConvertNodesToMythic` の前後でグラフをどう単純化するか、ノード単位の差分は未取得（§10.2は集計後のop種別カウント差分のみ）。
4. `MakeDSFsTrainable`/`PinLastOnChipConvDSF`/`PinLastOnChipMul` が付与する `__trainable`/`__trainable_dsf` の値が、後続の `train` ステップ（TorchNetのoptimizer構築時）でどう解釈されるかは範囲外（`train.py` 本体の解析が必要、[00_overview.md](../00_overview.md) §7 参照）。
5. 5モデル共通の `BreakFSRIntoPFSRAndIFSR` パラメータ（`half_iFSR_arr=[10.0], half_pFSR_arr=[2.5]`）がどのハードウェア特性・ノイズモデルから導出された値かは未調査。
