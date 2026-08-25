# `to_acm` ステップ解析

Mythic M2000 (Denali/ACE) アナログ compute-in-memory AI アクセラレータ SDK の **`to_acm` ステップ**の解析。対象バージョン `26.05.2`（SDK コンテナ `mythic-sdk-ubuntu-24.04:m2000-v26.05.2`, `mythic-model-zoo` の venv 内 `munc` パッケージ）。[to_training.md](to_training.md) の直後段にあたる。

主張はすべて実コードの**ファイルパス:行番号**を根拠に引用する。確定できない箇所は **[推測]** と明記する。パスは特記なき限りコンテナ内 `/root/mythic_sdk/v26.05.2/mythic-model-zoo/` からの相対（`munc/...` は実体としては同ディレクトリ下 `.venv/lib/python3.12/site-packages/munc/...` に存在するpipパッケージ）。

---

## 目次

- [1. `to_acm` とは何か（一言で）](#1-to_acm-とは何か一言で)
- [2. 入力と出力](#2-入力と出力)
- [3. 位置づけ — `hwconfig` はファイル経由で伝播する](#3-位置づけ--hwconfig-はファイル経由で伝播する)
- [4. ディスパッチ機構 — `to_training`と同型の共通実装](#4-ディスパッチ機構--to_trainingと同型の共通実装)
- [5. `get_mythic_to_bcm_conversion_ops` — 18個のopの固定シーケンス](#5-get_mythic_to_bcm_conversion_ops--18個のopの固定シーケンス)
- [6. 量子化の実行 — ここで初めて重みが固定小数点格子に丸められる](#6-量子化の実行--ここで初めて重みが固定小数点格子に丸められる)
- [7. BCM忠実度（`mma_class`）— `to_acm`は常に`munc_fp`に固定する](#7-bcm忠実度mma_class--to_acmは常にmunc_fpに固定する)
- [8. モデル横断比較](#8-モデル横断比較)
- [9. 実測 — YOLOPX / BEVFormer の実artifactから](#9-実測--yolopx--bevformer-の実artifactから)
- [10. 参照ファイルと未解明点](#10-参照ファイルと未解明点)

---

## 1. `to_acm` とは何か（一言で）

> **MYTHICノード（`MythicConv2d`等、重みはまだ連続値）を、BCMノード（`BCMConv2d`等、重み・バイアスが実際に固定小数点格子へ丸め済み）に変換する。[to_training.md](to_training.md) §7 が「実行しない」と結論した量子化の「実行（丸め）」がここで初めて起こる。**

`step_order`（`configs/common/base_config_generic.yaml`）上で `train` の直後、`create_artifact` の前段に位置する。[00_overview.md](../00_overview.md) §3.5 の状態遷移表が描く「MYTHIC → to_acm → **BCM**」の遷移そのものであり、`__type` メタデータが `MODELType.BCM` に書き換わる（`munc/_session.py:413` 付近、§5参照）。

本ステップの本質は3点:

1. **学習専用の残骸を除去する**（トレーニング用マスク・ノイズ注入op・補助出力）。
2. **重み・バイアスを実際に量子化する**（`WeightBiasQuantClip` が power-of-two固定小数点格子へ丸める。§6の実測で確認）。
3. **標準ONNX op（Conv/Gemm/Add/Sum/Mul）をBCM専用op（`BCMConv2d`/`BCMLinear`/`BCMAdd`/`BCMSum`/`BCMMul`）に変換する**。

---

## 2. 入力と出力

| 項目 | 実体 | 根拠 |
|---|---|---|
| 入力 (`src`) | 学習済みMYTHIC ONNX（`train`の出力。§3.5 §2.1参照, `trained.onnx`） | `configs/common/base_config_generic.yaml:120` `src: ${trained_model}` |
| 出力 (`dest`) | BCM(ACM) ONNX | `configs/common/base_config_generic.yaml:121` `dest: ${acm_model}` |

`torchnet`/`dataloader` キーは保持されるが（`base_config_generic.yaml:123-126`）、[to_training.md](to_training.md) の `torchnet.hw_model.hardware_config_name` のような明示的なハードウェア指定キーは**generic既定値には無い**。これは §3 で述べる伝播メカニズムのためである。

---

## 3. 位置づけ — `hwconfig` はファイル経由で伝播する

[to_training.md](to_training.md) §4.2 は `to_training_step` が `config.torchnet.hw_model.hardware_config_name` を明示的に読み、`sess.get_original_to_mythic_conversion_ops(hardware_config_name=...)` に渡すことを示した。**`to_acm` にはこの引数が存在しない**（`get_mythic_to_bcm_conversion_ops()` は引数を取らない、`munc/_session.py:392`）。

これは `hwconfig` がモデルファイル自体に永続化されているためである:

- `ONNXModel.hwconfig` のセッター（`munc/_onnx_model.py:2200` 付近）が `self.set_meta_data("hardware_config", config.name)` を呼び、値をONNXの `metadata_props` に書き込む。
- ゲッター（`munc/_onnx_model.py:182-183`）が `hardware_config_name = self.get_meta_data("hardware_config")` でロード時に読み戻し、`self._hwconfig = hw_config_registry[hardware_config_name]` を再構築する。

つまり `to_training` が `set_hwconfig_metadata`（[to_training.md](to_training.md) §5.1）で一度 `model.hwconfig` を設定すると、**その値はONNXファイルの`metadata_props`に焼き込まれ、以降のステップ（`to_acm`, `create_artifact`, `eval_acm` 等）がファイルを再ロードするだけで自動的にハードウェア種別を継承する**。§9の実測でも、to_acmの出力・create_artifactの出力の両方に `hardware_config: Denali` が一貫して残っていることを確認済み。

---

## 4. ディスパッチ機構 — `to_training`と同型の共通実装

[to_training.md](to_training.md) §4 で確認した「6モデル全てが共通関数へ委譲し、model-zoo側は薄いラッパーのみ」というパターンが、`to_acm` にもそのまま当てはまる。

`configs/common/step_types/common.yaml`:

```yaml
to_acm: mythic.model_zoo.common.conversion_steps.convert_training_to_acm_step
```

これを上書きするのは **yolopx / zero_dce / yolov8 の3モデル**（huggingface_classifiers / huggingface_robot_hand / bevformer / pythia は共通実装を素通しで使う）:

```yaml
# configs/yolopx/step_types/main.yaml:6
to_acm: mythic.model_zoo.yolopx.conversion_steps.adjust_and_convert_training_to_acm
```

3モデルの上書き内容は、いずれも共通の `convert_training_to_acm(config, extra_ops_after_conversion=..., op_configs_override=...)`（`mythic/model_zoo/common/conversion_steps.py:210-233`）への**パラメータ渡しのみ**であり、opシーケンス自体を書き換えるものではない:

| モデル | 追加処理 | 実装 |
|---|---|---|
| yolopx | `pad_outputs_to_8 and not pad_segmentation_outputs_to_8` の場合、セグメンテーションヘッドConvのパディングを外す（`unpad_segmentation_heads`）。`ChannelPaddingTo8` の `number_of_input_channels` を3に固定 | `yolopx/conversion_steps.py:47-72` |
| zero_dce | `ChannelPaddingTo8` を完全disable | `zero_dce/conversion_steps.py:123-125` |
| yolov8 | タスク別（detect/pose/segment/segpose）の `adjust_before_converting_to_acm()` を `extra_ops` として先頭に追加 | `yolov8/conversion_steps.py:81-87` |

`convert_training_to_acm` 自身（`common/conversion_steps.py:210-233`）は「MythicConvの入力チャネル数が全て3なら3にパディング、そうでなければ8にパディング」というヒューリスティック（`first_layer_can_be_padded_to_3`）を持ち、これはモデル共通のロジックであってモデル固有の分岐ではない。

---

## 5. `get_mythic_to_bcm_conversion_ops` — 18個のopの固定シーケンス

`munc/_session.py:392-414`:

```python
def get_mythic_to_bcm_conversion_ops(self):
    return op_conf_seq(
        nop("BeforeConversionToBCM", plot="trained_mythic"),
        ops.UpdateSumMulAttributes,
        ops.ClearTrainableValueList,
        ops.RemoveAllHWFidelityOps,
        ops.RemoveNetworkOutputsFromIntermediateNodes,
        op_conf(ops.InferStoreTensorShapes, update_outputs=True),
        ops.ChannelPaddingTo8,
        op_conf(ops.InferStoreTensorShapes, update_outputs=True),
        ops.ConvertMythicToConvs,
        ops.WeightBiasQuantClip,
        ops.HardCodeWeightAndBias,
        ops.RenameNodesAndEdges,
        ops.BreakBiasIntoRows,
        ops.AddInputShiftingOnchipToHardsigmoid,
        ops.ConvertConvsToBCM,
        ops.ConvertSumsToBCM,
        ops.ConvertParallelTransitionsToChannelwideMul,
        ops.AddCompilerOutputDType,
        ops.AbsorbOffchipMulNodes,
        op_do(lambda: self._model.set_meta_data('__type', MODELType.BCM)),
        nop("AfterConversionToBCM", plot="bcm")
    )
```

[to_training.md](to_training.md) §5 と同じ「固定opシーケンス、モデル側は`op_configs`でパラメータのみ上書き」という設計。役割ごとに分類する。

### 5.1 学習専用の残骸除去

- `UpdateSumMulAttributes`（`munc/ops/update_sum_mul_attributes.py:8-12`）: 「MythicのMUL/SUMノードの`multiplier`/`shift`属性を、対応するDSFオペランドに合わせて更新する。DSFが学習可能な場合、ハードウェア設定で指定された範囲にクリップする」。`train`ステップで動いたDSFの勾定更新結果を、ノード属性側に反映する後始末op。
- `ClearTrainableValueList`（`munc/ops/clear_trainable_value_list.py:6-9`）: 「全ノードから学習可能値リストを削除する。BCMモデルは学習可能パラメータを持つべきではないため」。[to_training.md](to_training.md) §5.6 の `MakeDSFsTrainable` が付けた `__trainable` マスクをここで消す。
- `RemoveAllHWFidelityOps`（`munc/ops/remove_all_HW_fidelity_ops.py:24-27`）: 「ノイズ注入op（`SYSTEMATIC_WEIGHT_NOISE_GENERATOR`等）と量子化op（`RELU_CLIP`/`HTAN_CLIP`/`DSF_CLIP`）を削除する」。**注意**: これらはMYTHICグラフ上に存在した「ハードウェア忠実度シミュレーション用のノード」であり、削除される。実際のノイズ・クリップ挙動はBCM変換後、TorchNet実行時に`bcm_layers.py`側の別実装（[03_accuracy_simulation.md](../03_accuracy_simulation.md) §5.4）が担う。
- `RemoveNetworkOutputsFromIntermediateNodes`: [to_training.md](to_training.md) §5.1 の `AddNetworkOutputsForIntermediateNodes`（損失計算用の補助出力）を除去する、対になる後始末op。

### 5.2 チャネルパディング・Mythic復元

- `ChannelPaddingTo8`（`munc/ops/channel_padding_to_8.py:11-16`）: `N_PAD = 8`。off-chip→on-chip境界のエッジ、およびConv/Gemm/MythicConv/MythicLinearの入力チャネル数をハードウェアの最小粒度（8）にパディングする。
- `ConvertMythicToConvs`（`munc/ops/convert_mythic_to_convs.py:5-10`）: `MythicConv2d→Conv`, `MythicLinear→Gemm`, `MythicQuantizedMul→Mul`, `MythicSum→Sum` と**op名を標準ONNXへ戻す**。[to_training.md](to_training.md) §6.3 の `ConvertNodesToMythic` の逆変換。「onnxruntimeで実行可能にする」ためであり、この時点ではまだBCM型（`BCMConv2d`等）ではない。

### 5.3 量子化・定数化（§6で詳述）

- `WeightBiasQuantClip`: **実際の重み・バイアス丸め**。§6で実測込みで詳述。
- `HardCodeWeightAndBias`（`munc/ops/hard_code_weight_and_bias.py:5-10`）: 「重み・バイアスをinitializerとしてハードコードし、それらを生成していたノードをグラフから削除する。入力がpFSRノード経由の場合はpFSRノードの入力に対して操作する」。TorchNet実行結果（`collect_outputs_first_batch`）を使って値を確定させる。
- `BreakBiasIntoRows`（`munc/ops/break_bias_into_rows.py:14-23`）: 「ACEのフラッシュセルに合わせてバイアスを複数行に分割する。1つのACEが持つバイアス行数を超える値を表現するため。例: ACEが6行なら`[32]`→`[32,6]`に変形し、各行の値の合計が元の値と一致するようスケールダウンする」。既定は`BiasSplittingMethod.BALANCED`。§9の実測でDenali(15行)の場合の具体的な形状変化を確認。

### 5.4 BCM型への変換

- `AddInputShiftingOnchipToHardsigmoid`（`munc/ops/add_input_shifting_onchip_to_hardsigmoid.py:9-20`）: 負のMMA入力を補正するために挿入されていた`ADDInputShiftingOnchip`ノードを、条件を満たす場合はhard sigmoid活性化に置き換えて除去する。
- `ConvertConvsToBCM`（`munc/ops/convert_convs_to_bcm.py:6-16`）: `Conv→BCMConv2d`, `Gemm→BCMLinear`。`bcm_class_str`（既定`"munc_fp"`, §7）・`bcm_attr_str`をノード属性 `__mma_class`/属性名 として設定する（`bcm_utils.MMA_CLASS_ATTRIBUTE_NAME`）。
- `ConvertSumsToBCM`（`munc/ops/convert_sums_to_bcm.py:4-8`）: `Add→BCMAdd`, `Sum→BCMSum`, `Mul→BCMMul`。
- `ConvertParallelTransitionsToChannelwideMul`（`munc/ops/convert_parallel_transitions_to_channelwide_mul.py:6-11`）: 「Concatノードの直前に並列するoff-chip transition乗算器を、Concat後段の単一チャネル幅乗算器に変換する」。bevformerはこのopを`conversion_parameters.ops.ConvertParallelTransitionsToChannelwideMul.enabled=False`で無効化している（§8）。

### 5.5 コンパイラ向け仕上げ

- `AddCompilerOutputDType`（`munc/ops/add_compiler_output_dtype.py:17-22`）: 各ノードの出力統計（最小値）を見て `int8`/`uint8` をノードに付与する。活性化関数別の既定マッピング（`ACTIVATION_TO_DATATYPE`）も持つ（例: `hardtanh→int8`, 他は`uint8`）。
- `AbsorbOffchipMulNodes`（`munc/ops/absorb_offchip_mul_nodes.py:6-12`）: off-chipのConv/Gemmに前後する乗算専用ノードを削除し、その係数をConv/Gemmの重み・バイアスに吸収する。

---

## 6. 量子化の実行 — ここで初めて重みが固定小数点格子に丸められる

[to_training.md](to_training.md) §7 は「`to_training`はスケール因子を確定するが、丸め・ノイズ注入は実行時」と結論した。`to_acm`の`WeightBiasQuantClip`（`munc/ops/weight_bias_quant_clip.py:11-17`）の実装を見ると、**重みの実際の丸めはここ（`to_acm`）で行われる**ことが分かる:

```python
initializer = initializer * weight_scale
quantized_initializer = _session_tools.quantize_weight(initializer, self.hwconfig.weight_fractional_bits)
quantized_initializer = _session_tools.clip_weight(quantized_initializer, min_clip, max_clip)
self.model.set_initializer_np(edge, quantized_initializer)
```

`quantize_weight`（`munc/_session_tools.py:344-347`）:

```python
def quantize_weight(weight, num_fractional_bits=0):
    scale = (2 ** num_fractional_bits)
    return np.round(weight * scale) / scale
```

これは典型的なpower-of-two固定小数点量子化そのものである（配列のdtypeはfloat32のまま変わらないが、値は`1/2^num_fractional_bits`刻みの格子点に丸められる）。`num_fractional_bits`はハードウェア設定（`munc/hw_specs.py`）から取る:

| ハードウェア | `weight_fractional_bits` | `bias_rows` | `weight_min`/`max` | `signed` |
|---|---|---|---|---|
| Boreas（`hw_specs.py:237-240`） | 0 | 6 | -128 / 127 | False |
| Denali（`hw_specs.py:269-272`） | 8 | 15 | -128 / 127 | True |

Boreasでは`weight_fractional_bits=0`のため`quantize_weight`は実質「整数への丸め」に退化する。Denaliでは8bitの小数部を持つ固定小数点（格子幅 `1/256`）になる。**iFSR/pFSR自体も連続値ではなく、ハードウェアごとに定義された離散集合から選ばれる**（`hw_specs.py`, 例: Denali `pFSR_values=[1.0, 3.0, 5.0, 10.0]`）。[to_training.md](to_training.md) §5.4.1 で導出した「FSR系スケール因子=量子化スケール」という理解と、本ステップの実際の丸め処理は表裏一体である。

§9の実測で、この`1/256`格子への丸めが実際のartifactファイル上で正確に確認できる。

---

## 7. BCM忠実度（`mma_class`）— `to_acm`は常に`munc_fp`に固定する

[03_accuracy_simulation.md](../03_accuracy_simulation.md) §5.4 が詳述する「6階層忠実度モデル」（`munc_fp`/`munc_int8`/`munc_simple`/`munc_digital`/`munc_tacm`/`munc_acm_signoff`）のうち、`to_acm`（`convert_training_to_acm`, `common/conversion_steps.py:220`）が`ConvertConvsToBCM`に渡す既定値は常に **`bcm_class_str="munc_fp"`**（浮動小数点理想、出力のround/clipのみ）である。これはモデルの`conversion_parameters.ops`で上書きされない限り固定。

**`to_acm`の出力ファイル自体は特定の忠実度に「固定」されているのではなく、`munc_fp`という最も理想的な忠実度から出発する**。後続の`eval_acm`（`SwitchBCM`で6階層のいずれかに切替、[to_training.md](to_training.md)は範囲外）や`create_artifact`（`SwitchBCM(munc_digital)`で必ずノイズ無しデジタル忠実に固定、[create_artifact.md](create_artifact.md) 参照）が、同じBCMノード構造の上で`__mma_class`属性だけを書き換えて忠実度を選び直す。§9の実測で、`to_acm`出力（`munc_fp`）と`create_artifact`出力（`munc_digital`）の違いを直接確認する。

---

## 8. モデル横断比較

[to_training.md](to_training.md) §8 と同様、opシーケンスは全モデル共通なので、比較対象はconfigのop上書きのみ。

| モデル | config | 上書き内容 |
|---|---|---|
| 共通既定（generic） | `base_config_generic.yaml:120-127` | `conversion_parameters`キー自体が無い（`convert_training_to_acm`は`config.get("conversion_parameters", {})`で安全に空dict扱いする、[to_training.md](to_training.md)の`to_training_step`が`config.conversion_parameters.options`を無条件アクセスするのとの対比） |
| yolopx | `base_config.yaml:71-74` | `pad_outputs_to_8=True`, `pad_segmentation_outputs_to_8=False`, `segmentation_head_convs`（2ノード名） |
| yolov8 | `training/base_config.yaml:63-64` | `task: ${ultralytics_task}`（タスク別`adjust_before_converting_to_acm()`のディスパッチに使う） |
| bevformer | `bevformer_tiny.yaml:171-182` | `ChannelPaddingTo8.enabled=False`, `ConvertParallelTransitionsToChannelwideMul.enabled=False` |
| huggingface_classifiers / robot_hand | `training/base_config.yaml` | `model_setup: ${model_setup}`のみ（op上書き無し） |
| zero_dce | 上書きなし（`adjust_and_convert_training_to_acm`側で`ChannelPaddingTo8.enabled=False`をハードコード） | — |

bevformerが`ChannelPaddingTo8`を無効化している点は、[to_structural.md](to_structural.md) §7.1で確認した「bevformerは`GeneralizeBatchSize`も無効化（batch次元が一貫しないため）」という特殊事情と同系統の対応と考えられる[推測]。

---

## 9. 実測 — YOLOPX / BEVFormer の実artifactから

### 9.1 実測データの所在

`mythic_ppa_explore_2605_2b`コンテナ（v26.05.2）上の以下のパスに、実際にPPA探索作業で生成された`compiler_ready_artifact`ディレクトリが残存していた:

| パス | モデル | 内容 |
|---|---|---|
| `/tmp/yx_probe/compiler_ready_artifact/reference/` | YOLOPX | `source_model.onnx`（**to_acmの出力に相当**）+ `compiler_ready_artifact_{off_chip_0,on_chip_1_bcm,off_chip_2}.onnx`（create_artifact出力の分割） |
| `/tmp/bev_probe/compiler_ready_artifact/reference/` | BEVFormer | `compiler_ready_artifact_on_chip_1_bcm.onnx`のみ |

`create_artifact_from_config`（`munc/cli/helpers.py:595`）が`source_model = sess.model.deepcopy()`で**`create_artifact`の変換が走る直前のモデルをコピーして`source_model.onnx`として同梱する**ため、これは実質的に`to_acm`の出力そのものである。

### 9.2 `source_model.onnx`（YOLOPX, to_acm出力）の実測

```python
import onnx
src = onnx.load("source_model.onnx")
```

| 項目 | 値 |
|---|---|
| ノード数 | 406 |
| op種別（上位） | `BCMConv2d`(206) / `Slice`(94) / `BCMSum`(41) / `Concat`(30) / `Resize`(9) / `Conv`(9, off-chip分) / `MaxPool`(8) / `Mul`(3) |
| `metadata_props` | `hardware_config: Denali`, `__type: BCMModel` |
| `BCMConv2d`の`__mma_class`属性 | `munc_fp`（§7で述べた既定値と一致） |
| `BCMConv2d`の`__pFSR`/`__iFSR` | `5` / `20`（Denaliの離散集合値の一部, §6） |

**重みの量子化格子を直接検証**: 任意の`BCMConv2d`ノードの重みinitializer（float32のまま）に対し、`weight * 256`が誤差0で整数になることを確認した（複数ノードで検証、最大誤差 `0.0e+00`）。`weight * 64`/`weight * 128`では整数にならない（誤差0.5）。これは`weight_fractional_bits=8`（Denali, `hw_specs.py:271`）による`1/256`刻みの固定小数点格子と厳密に一致する。§6の理論的主張（`quantize_weight`が実際に丸めを行う）の直接的な実測裏付けである。

**バイアスの行分割を直接検証**: `BCMConv2d`の3番目の入力（バイアス）は形状`(32, 15)`（元は`(32,)`のはず）で、値は`[2., 2., 2., ...]`のように整数値。Denaliの`bias_rows=15`（`hw_specs.py:272`）と一致し、`BreakBiasIntoRows`が実際に15行へ分割していることを確認した。

**off-chip側は量子化されない**: `create_artifact`出力の`off_chip_2`（Conv, 9ノード）の重みは`7.5453099e-06`のような任意精度のfloat32値であり、`1/256`格子への丸めは一切見られない（`weight*256`の誤差は`0.499`程度、格子に全く乗っていない）。これは[to_training.md](to_training.md) §6.3で確認した「`ConvertNodesToMythic`がoff-chipノードをスキップする（`OFFCHIP_IGNORE`）」構造が、量子化の実行段階（`to_acm`）でも一貫していることを示す——off-chipのConv/Gemmは`WeightBiasQuantClip`の対象op種別（`ONNXType.CONV, ONNXType.GEMM`）に**形式的には含まれる**が、その属性`off_chip: OFFCHIP_IGNORE`（`munc/ops/weight_bias_quant_clip.py`内`_get_info`）により実際にはスキップされる。

### 9.3 BEVFormerとの横断比較

`bev_probe`の`compiler_ready_artifact_on_chip_1_bcm.onnx`（create_artifact後）でも同じ確認が取れた:

| 項目 | YOLOPX (on_chip_1_bcm) | BEVFormer (on_chip_1_bcm) |
|---|---|---|
| ノード数 | 387 | 148 |
| `BCMConv2d`数 | 206 | 74 |
| `hardware_config` | Denali | Denali |
| `__mma_class` | `munc_digital` | `munc_digital` |
| 重み量子化格子 | `1/256` | `1/256` |

両モデルとも`Denali`・`1/256`格子・`munc_digital`（create_artifact後）で一致しており、[to_training.md](to_training.md) §10.2で確認したBEVFormerの`MythicConv2d`数（76、未学習構造）と、本測定の学習済みBEVFormerの`BCMConv2d`数（74）が近い値であることも、パイプラインの一貫性を支持する材料である（`ChannelPaddingTo8`等によるノード数の微増減はあり得るため厳密一致は期待しない）。

### 9.4 create_artifact前後のノード数の対応関係

YOLOPXの`source_model.onnx`（406ノード）と、分割後の3ファイル（`on_chip_1_bcm`=387 + `off_chip_0`=2 + `off_chip_2`=14 = 403）を比較すると、差分3ノードは`source_model.onnx`にあった3個の`Mul`ノードに一致する。これは`create_artifact`側の`AbsorbOffchipMulNodes`（§5.5、`to_acm`にも同名opが存在するが`create_artifact`側でも重ねて実行される想定[推測: 二重実行の詳細な理由は未確認]）や`ConvertParallelTransitionsToChannelwideMul`相当の処理で吸収されたためと考えられる。`Concat`は`source_model.onnx`で30、分割後`on_chip_1_bcm`(27)+`off_chip_2`(3)=30で完全一致しており、グラフ分割自体はノードの欠落なく行われていることが確認できる。

---

## 10. 参照ファイルと未解明点

### 抽出ソースの所在

| 分類 | ファイル |
|---|---|
| ディスパッチ | `mythic/model_zoo/common/conversion_steps.py`（`convert_training_to_acm`, `convert_training_to_acm_step`） |
| 変換op列本体 | `munc/_session.py`（`get_mythic_to_bcm_conversion_ops`） |
| 量子化実行 | `munc/ops/weight_bias_quant_clip.py`, `munc/_session_tools.py`（`quantize_weight`, `clip_weight`） |
| BCM型変換 | `munc/ops/convert_convs_to_bcm.py`, `munc/ops/convert_sums_to_bcm.py`, `munc/bcm/bcm_utils.py` |
| ハードウェア仕様 | `munc/hw_specs.py`（`boreas_hw_config`, `denali_hw_config`） |
| hwconfig永続化 | `munc/_onnx_model.py`（`hwconfig`プロパティのgetter/setter, `metadata_props["hardware_config"]`） |
| 実測データ | `mythic_ppa_explore_2605_2b`コンテナの`/tmp/yx_probe/`, `/tmp/bev_probe/` |

### 未解明点

1. `AbsorbOffchipMulNodes`が`to_acm`（`get_mythic_to_bcm_conversion_ops`）と`create_artifact`（`get_bcm_to_artifact_conversion_ops`には同名opは無いが、実測で3ノード分の吸収相当の効果が見られた）の両方で走っているように見える正確な経路は未確認（§9.4）。
2. `HardCodeWeightAndBias`が呼ぶ`collect_outputs_first_batch`（実データをTorchNetに1バッチ流して出力を確定させる）が、具体的にどの中間値を「ハードコード」の対象にしているかの詳細は未深掘り。
3. `bcm_utils.py`のBCM層基盤（`MMA_CLASS_ATTRIBUTE_NAME`等）自体の実装は本解析では表層のみ確認。
4. iFSR/pFSRが離散集合（`hw_specs.py`の`pFSR_values`/`iFSR_values`）からどのように選択されるかのアルゴリズム（[to_training.md](to_training.md) §5.4の`BreakFSRIntoPFSRAndIFSR`側）は`to_training.md`側で「未深掘り」としており、本ドキュメントでも再確認していない。
