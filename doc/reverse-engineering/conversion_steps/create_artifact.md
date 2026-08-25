# `create_artifact` ステップ解析

Mythic M2000 (Denali/ACE) アナログ compute-in-memory AI アクセラレータ SDK の **`create_artifact` ステップ**の解析。対象バージョン `26.05.2`（SDK コンテナ `mythic-sdk-ubuntu-24.04:m2000-v26.05.2`, `mythic-model-zoo` の venv 内 `munc` パッケージ）。[to_acm.md](to_acm.md) の直後段、`step_order` 上の最終SDK側ステップであり、この直後に Compiler コンテナ側（[01_compilation.md](../01_compilation.md)）へ渡る。

主張はすべて実コードの**ファイルパス:行番号**を根拠に引用する。確定できない箇所は **[推測]** と明記する。パスは特記なき限りコンテナ内 `/root/mythic_sdk/v26.05.2/mythic-model-zoo/` からの相対（`munc/...` は実体としては同ディレクトリ下 `.venv/lib/python3.12/site-packages/munc/...` に存在するpipパッケージ）。

---

## 目次

- [1. `create_artifact` とは何か（一言で）](#1-create_artifact-とは何か一言で)
- [2. 入力と出力](#2-入力と出力)
- [3. 位置づけ — BCM → COMPILER、忠実度は`munc_digital`に固定](#3-位置づけ--bcm--compiler忠実度はmunc_digitalに固定)
- [4. ディスパッチ機構](#4-ディスパッチ機構)
- [5. `get_bcm_to_artifact_conversion_ops` — モデル変換op列](#5-get_bcm_to_artifact_conversion_ops--モデル変換op列)
- [6. グラフ分割 — off_chip_N / on_chip_N_bcm](#6-グラフ分割--off_chip_n--on_chip_n_bcm)
- [7. アーティファクトのファイル構成](#7-アーティファクトのファイル構成)
- [8. モデル横断比較](#8-モデル横断比較)
- [9. 実測 — YOLOPX / BEVFormer の実artifact](#9-実測--yolopx--bevformer-の実artifact)
- [10. 参照ファイルと未解明点](#10-参照ファイルと未解明点)

---

## 1. `create_artifact` とは何か（一言で）

> **BCM(ACM) ONNXを、on-chip/off-chip サブグラフに物理分割し、Compilerコンテナが読める`compiler_ready_artifact.tar.gz`（複数のONNXファイル＋パイプライン定義＋メタデータ）に packaging する。モデル変換としては「BCM忠実度を`munc_digital`（ノイズ無し）に固定する」ことが唯一の実質的な数値変更である。**

`step_order`上で`to_acm`の直後、`compile`の前段（[00_overview.md](../00_overview.md) §3.5の「BCM → create_artifact → COMPILER」）。SDKコンテナ側の処理としては最後の変換ステップであり、この後は`mythic-compiler`（Compilerコンテナ）が主体になる（[01_compilation.md](../01_compilation.md)）。

本ステップは2つの異なる仕事を1つのステップ名に統合している:

1. **モデル変換**（`get_bcm_to_artifact_conversion_ops`, §5）: BCM忠実度の固定・チャネルパディング除去・RGB/BGR調整・検証・opset変更。
2. **アーティファクト生成**（`generate_artifact`, §7）: グラフをoff-chip/on-chip境界で物理分割し、複数のランタイム形式（標準/RISV2/Inference Engine）向けファイル・パイプライン定義・メタデータを1つの`tar.gz`にまとめる。

---

## 2. 入力と出力

| 項目 | 実体 | 根拠 |
|---|---|---|
| 入力 (`src`) | BCM(ACM) ONNX（`to_acm`の出力, `acm_model`） | `configs/common/base_config_generic.yaml:175` `src: ${acm_model}` |
| 出力 (`dest`) | `compiler_ready_artifact.tar.gz`（コピー先） | `configs/common/base_config_generic.yaml:176` `dest: ${artifact}` |
| 中間生成物 | `artifact_directory`（既定`${data_dir}`）下に一時展開される`compiler_ready_artifact/`ディレクトリ | `base_config_generic.yaml:177` |

`create_training_artifact`（`mythic/model_zoo/common/conversion_steps.py:270-296`）内のコメントに注意点がある:

```python
# SessionFromConfig uses `dest` as an ONNX output file. That's not what we want here.
sess_config = config.copy()
sess_config.dest = None
```

`create_artifact`の`dest`は「ONNXファイルの保存先」ではなく「tar.gzのコピー先」であるため、`SessionFromConfig`（[to_structural.md](to_structural.md) §2の`__exit__`保存機構）が誤ってONNXとして保存してしまわないよう、`dest`を意図的に`None`に潰してから`run_conversion`に渡す。実際のtar.gzコピーは`create_artifact_from_config`内の`copy_file(local_artifact_file, config.dest)`（`munc/cli/helpers.py:607`）が行う。

---

## 3. 位置づけ — BCM → COMPILER、忠実度は`munc_digital`に固定

`get_bcm_to_artifact_conversion_ops`（`munc/_session.py:447-461`）の先頭で入力状態を検証する:

```python
op_do(lambda: assert_model_is(self.model, MODELType.BCM)),
```

[to_acm.md](to_acm.md) §7で述べた通り、`to_acm`の出力は`__mma_class="munc_fp"`（浮動小数点理想忠実度）で固定されている。`create_artifact`はここで**忠実度を`munc_digital`に切り替える**:

```python
op_conf(ops.SwitchBCM, bcm_class_str=digitalmodel.FACTORY_NAME),
```

[03_accuracy_simulation.md](../03_accuracy_simulation.md) §5.4の6階層忠実度モデルのうち`munc_digital`は「量子化+クリップ+マルチサイクル（8bit分解）、デジタル忠実（ノイズ無しアナログ上限）」と位置づけられている。すなわち**実際にチップに書き込まれる決定論的な挙動**を表現する忠実度であり、`create_artifact`が生成するコンパイラ入力は「アナログノイズを含まない、ハードウェアの理論上限の挙動」を前提にしている——実チップ上の推論精度は、ここからさらにノイズが乗った分だけ劣化する、という関係になる。

`hwconfig`の継承メカニズムは[to_acm.md](to_acm.md) §3と同一（`metadata_props["hardware_config"]`経由）であり、`create_artifact`側で明示的に再指定する必要はない。

処理完了後、`__type`メタデータが`MODELType.COMPILER`に書き換わる（`munc/_session.py:459`付近）。

---

## 4. ディスパッチ機構

[to_acm.md](to_acm.md) §4と同型: 共通実装`create_training_artifact_step`（`configs/common/step_types/common.yaml`: `create_artifact: mythic.model_zoo.common.conversion_steps.create_training_artifact_step`）を、**4モデル**（yolopx / zero_dce / yolov8 / synthetic）が独自の`create_artifact`関数で上書きする（huggingface_classifiers / huggingface_robot_hand / bevformer / pythia は共通実装を素通しで使う）。いずれも共通の`create_training_artifact(config, extra_ops=..., num_outputs_to_keep=..., delete_outputs_from_beginning=...)`（`common/conversion_steps.py:270-296`）へのパラメータ渡しに留まる:

| モデル | 追加処理 | 実装 |
|---|---|---|
| yolopx | `remove_redundant_transition_muls`を`extra_ops`で追加実行 | `yolopx/conversion_steps.py:83-90` |
| zero_dce | `model.set_meta_data('__BGR', "True")`＋入力shape更新、`remove_red_shift`（バイアス行あたりのシフト値を調整）、`num_outputs_to_keep=1`（余分な出力を削除） | `zero_dce/conversion_steps.py:197-211` |
| yolov8 | タスク別`num_main_outputs`・`delete_outputs_from_beginning=True`・タスク別`get_before_generating_artifact_ops()` | `yolov8/conversion_steps.py:245-251` |
| synthetic | `mythic.model_zoo.synthetic.steps.create_artifact`（テスト/デモ用モデル。本解析では内容未確認） | `configs/synthetic/step_types/main.yaml:5` |

zero_dceの`__BGR`メタデータ設定は、`_generate_artifact.py`の`generate_standard_artifact_files`（§7）が`expected_input = 'RGB' if model.get_meta_data('__BGR') is None else 'BGR'`という分岐を持つことに対応する——**この`__BGR`メタデータの有無が、アーティファクトメタデータ上の`expected_input`フィールドを決める唯一の経路**である。

---

## 5. `get_bcm_to_artifact_conversion_ops` — モデル変換op列

`munc/_session.py:447-461`:

```python
def get_bcm_to_artifact_conversion_ops(self):
    return op_conf_seq(
        op_do(lambda: assert_model_is(self.model, MODELType.BCM)),
        op_conf(ops.SwitchBCM, bcm_class_str=digitalmodel.FACTORY_NAME),
        ops.RemoveDummyChannelPadding,
        ops.InferStoreTensorShapes,
        op_conf(ops.AdjustFirstConvForRGBToBGR, enabled=False),
        ops.AttachLUTAttribute,
        op_conf(op_do(lambda: _verify.verify_compiler_model(self.model, self.hwconfig)), name="VerifyCompilerModel"),
        op_do(lambda: self._model.set_meta_data('__type', MODELType.COMPILER)),
        op_do(lambda: self.change_opset(COMPILER_ONNX_OPSET)),
    )
```

[to_acm.md](to_acm.md) §5と比べると**opの数は少ない**（9個）。個々の役割:

- `SwitchBCM(munc_digital)`: §3で述べた忠実度固定。
- `RemoveDummyChannelPadding`（`munc/ops/remove_dummy_channel_padding.py:6-11`）: 「実際のパディングを行わないパディングノード（形状維持だけのダミー）を削除する」。[to_acm.md](to_acm.md) §5.2の`ChannelPaddingTo8`が付けたパディングのうち、実質値0のものを整理する後始末。
- `AdjustFirstConvForRGBToBGR`（既定`enabled=False`, `munc/ops/adjust_first_conv_for_rgb_to_bgr.py:9-17`）: 「最初のConv層の0次元目と2次元目（チャネル）を入れ替えることで、RGB⇄BGRの入力順を切り替える。ネットワークが分岐し始めたら探索を止める」。configで`enabled=True`にするモデルのみ有効（§8）。
- `AttachLUTAttribute`（`munc/ops/attach_lut_attribute.py:14-30`）: Swish活性化を持つノードに対して、`-128〜127`の入力レンジに対する出力をあらかじめ計算したルックアップテーブル(LUT)を属性として付与する。ハードウェアがSwishを直接計算せずLUT参照で近似することに対応。
- `VerifyCompilerModel`（`munc/_verify.py:17-42`）: 出力直前の妥当性検証。モデル状態が`BCM`であること（`COMPILER`へ遷移する**前**の検証）、Boreasの場合は外部入力数が1〜3個であること、出力数が1個以上であること等を`Exception`/`ValueError`でチェックする。
- `change_opset(COMPILER_ONNX_OPSET)`: `COMPILER_ONNX_OPSET`（`munc/_constants.py`, 値は`20`で`MUNC_INTERNAL_ONNX_OPSET`と同一）へ最終的なopset変更。

**model-zoo側は`AdjustFirstConvForRGBToBGR`以外このシーケンスに触れない**（§8の比較表参照）。

---

## 6. グラフ分割 — off_chip_N / on_chip_N_bcm

モデル変換op列自体にはグラフ分割ロジックは含まれない。分割は`create_artifact_from_config`（`munc/cli/helpers.py:575-607`）が呼ぶ`generate_artifact`→`generate_standard_artifact_files`（`munc/_generate_artifact.py:18-77`）内の`graph_splitter.split_onnx_graph`が担う:

```python
subgroups = _pattern_detector.find_on_and_off_chip_neighborhoods(model)
```

`__off_chip`属性（[to_structural.md](to_structural.md) §5で導入された、`to_training`/`to_acm`を経ても保持され続ける属性）が連続する区間を1つのサブグラフとしてまとめ、on-chip/off-chip交代点でグラフを切り出す。命名規則（`munc/_artifact/graph_splitter.py:130`付近）:

```python
models["off_chip_" + str(i)] = new_model
```

on-chip側は`on_chip_N_bcm`という命名になる（実測ファイル名`compiler_ready_artifact_on_chip_1_bcm.onnx`より）。[00_overview.md](../00_overview.md) §5(1)で触れられていた「artifactステージ名`on_chip_1_bcm`の"bcm"はBoreas Compute Modelの略と推定される」という記述と、本ステップで実際にBCMノード（`BCMConv2d`等）で構成されたサブグラフに`_bcm`サフィックスが付くことが、実装レベルで確認できる。

`merge_subgraphs=True`（`generate_standard_artifact_files`の既定値, `_generate_artifact.py:20`）により、入力側・出力側でそれぞれ複数に分かれうるoff-chipサブグラフはトポロジカルソートで1つにマージされ、結果的に典型的な構成は`off_chip_0 → on_chip_1_bcm → off_chip_2`の3分割になる（§9の実測で確認）。モデルによってon-chip/off-chip境界が複数回入れ替わる場合はより多い分割数になり得る[推測: 本解析で確認したのは3分割の例のみ]。

---

## 7. アーティファクトのファイル構成

`generate_standard_artifact_files`（`_generate_artifact.py:18-77`）とその呼び出し先が生成するファイル群:

| ファイル群 | 生成関数 | 内容 |
|---|---|---|
| `compiler_ready_artifact_{off_chip_N,on_chip_N_bcm}.onnx` | `artifact_writer.write_artifact`（`_generate_artifact.py:57-69`） | 分割後の各サブグラフの標準ONNX（レイアウト変換なし） |
| `RISV2_compiler_ready_artifact_off_chip_N.onnx` + `pipeline_ris.yml` | `generate_risv2_artifact_files`（`_generate_artifact.py:99-111`） | RISV2ランタイム向けoff-chip ONNX＋パイプライン定義 |
| `ie_compiler_ready_artifact_off_chip_N.onnx` | `generate_ie_artifact_files`（`_generate_artifact.py:114-121`） | Inference Engine向け（NHWCレイアウトに変換） |
| `pipeline.yml` | `generate_pipeline_artifact_files`（`_generate_artifact.py:135-139`） | 標準パイプライン定義（§9.2で実例を確認） |
| `source_model.onnx` | `create_artifact_from_config`（`munc/cli/helpers.py:595, 604`） | **`create_artifact`の変換前**（＝`to_acm`の出力そのもの）のモデルをそのまま同梱。[to_acm.md](to_acm.md) §9の実測データの根拠 |
| `*.h5`（データサンプル） | `generate_h5_artifact_files`（`_generate_artifact.py:142-166`） | `image_numbers`で指定した画像番号の中間出力を各層ごとにHDF5化（デバッグ・数値検証用） |
| `contents.json` | `artifact_writer.write_artifact_metadata`（`_artifact/artifact_writer.py:345-357`） | 上記すべてのファイルパス・`munc_artifact_version`（現在値`2`）等をまとめたメタデータ |

これらは`Artifact`コンテキストマネージャ（`munc/_artifact/artifact_writer.py:376-453`）の中で一時ディレクトリ`{artifact_root_dir}/compiler_ready_artifact/`（`MUNC_ARTIFACT_DIRECTORY_NAME`, `_artifact/artifact_writer.py:30`）に書き込まれ、`__exit__`時に:

```python
subprocess.run(["tar", "-C", str(self.artifact_root_dir), "-cpzf", str(self.out_file),
                MUNC_ARTIFACT_DIRECTORY_NAME], check=True)
```

で単一の`compiler_ready_artifact.tar.gz`へ固められる。`Artifact`クラスは`in_file`指定時には逆に`tar -xf`で展開する汎用I/Oラッパーであり、`compile`ステップ側（`add_image_shape_to_compiler_config_file_name`, `munc/cli/helpers.py:624`）でも同じクラスがアーティファクトの読み出しに使われる。

`config.include_debug`が`True`の場合のみ、`config.artifact.debug_dir`（既定`DEBUG_DIR`）の内容もアーティファクトにコピーされる（generic既定は`include_debug: False`, `base_config_generic.yaml:178`）。

---

## 8. モデル横断比較

| モデル | config | `AdjustFirstConvForRGBToBGR` | その他 |
|---|---|---|---|
| 共通既定（generic） | `base_config_generic.yaml:185-188` | **`enabled: True`** | `artifact.padding_value=0`, `image_numbers=[]`（デバッグh5生成は既定オフ） |
| yolopx | `base_config.yaml:94-98` | `enabled: False` | `remove_redundant_transition_muls`（§4） |
| zero_dce | `base_config.yaml:124-130` | `enabled: False` | `resolution: [1280,720]`, `__BGR`メタデータ付与（§4） |
| yolov8 | `training/base_config.yaml:74-78` | **上書き無し（generic既定`True`のまま）** | `artifact.padding_value=114`（YOLO系で標準的なグレー系パディング値） |
| huggingface_classifiers | `training/base_config.yaml:107-108` | **上書き無し（`True`のまま）** | `model_setup`のみ |
| huggingface_robot_hand | `training/base_config.yaml:131-136` | `enabled: False` | — |
| bevformer | `bevformer_tiny.yaml:186-192` | `enabled: False` | — |
| dummy | `base_config.yaml:25-29` | `enabled: False` | テスト用の最小モデル |

**RGB⇄BGR入れ替えを有効化しているのはyolov8とhuggingface_classifiers（resnet50等の画像分類系）のみ**であり、yolopx・zero_dce・robot_hand・bevformer・dummyは無効化している。generic既定が`True`であることから、後者のグループは「明示的にオプトアウトした」形になっている。画像分類・検出タスク（学習データセットが標準的なRGB前提のことが多い一般公開モデル）でBGR変換が必要という組み合わせは、各モデルの学習パイプラインがどのカラーチャネル順序を前提にしていたかに依存すると考えられる[推測: 個々のモデルの学習コード側の前提までは本解析では追っていない]。

---

## 9. 実測 — YOLOPX / BEVFormer の実artifact

[to_acm.md](to_acm.md) §9で使用した`mythic_ppa_explore_2605_2b`コンテナ上の実データをそのまま利用する。

### 9.1 ファイル一覧（YOLOPX, `/tmp/yx_probe/compiler_ready_artifact/reference/`）

| ファイル | サイズ | 対応する生成関数 |
|---|---|---|
| `source_model.onnx` | 132,266,598 B | `create_artifact_from_config`（to_acm出力そのまま） |
| `compiler_ready_artifact_on_chip_1_bcm.onnx` | 132,293,702 B | `write_artifact`（標準off-chip/on-chip分割） |
| `compiler_ready_artifact_off_chip_0.onnx` | 18,153 B | 同上 |
| `compiler_ready_artifact_off_chip_2.onnx` | 45,897 B | 同上 |
| `RISV2_compiler_ready_artifact_off_chip_{0,2}.onnx` | 18,521 B / 47,652 B | `generate_risv2_artifact_files` |
| `ie_compiler_ready_artifact_off_chip_{0,2}.onnx` | 18,521 B / 48,292 B | `generate_ie_artifact_files` |
| `pipeline.yml`, `pipeline_ie.yml`, `pipeline_ris.yml`, `pipeline_torchnet.yml`, `pipeline_update.yml` | 数百B | `generate_pipeline_artifact_files`等 |

**注**: この特定のプローブディレクトリには`contents.json`（`Artifact`が本来書き出すメタデータファイル）が見当たらなかった。`reference/`直下のファイル群は直接検証できたが、`tar.gz`本体を経由した完全なアーティファクト一式としての生成過程は未確認（§10の未解明点）。

### 9.2 `pipeline.yml`の内容

```yaml
Pipeline:
  1:
    class_cfg_data:
      address: 0
      runtime: ../runtime.yml
    class: AMPProcessor
    name: on_chip_1
  2:
    class_cfg_data:
      filename:
        onnx: reference/ie_compiler_ready_artifact_off_chip_2.onnx
    class:
    - ONNXProcessor
    name: off_chip_2
Contents:
  expected_input: RGB
  padding_value: 0
  input_shapes:
    input:
    - 3
    - 736
    - 1280
```

`on_chip_1`ステージは`AMPProcessor`（アナログチップ実行、実行時パラメータは`../runtime.yml`参照）、`off_chip_2`ステージは`ONNXProcessor`（`ie_`プレフィックス付きONNXをonnxruntime等で実行）という**異なるプロセッサクラスを繋いだパイプライン定義**になっている。`expected_input: RGB`は§4で述べた`__BGR`メタデータが未設定（YOLOPXは`AdjustFirstConvForRGBToBGR`を無効化しているモデル）であることと整合する。

### 9.3 `__type`とBCM忠実度の遷移を実測で確認

| ファイル | `__type` | `__mma_class`（BCMConv2dノード） |
|---|---|---|
| `source_model.onnx`（to_acm出力） | `BCMModel` | `munc_fp` |
| `compiler_ready_artifact_on_chip_1_bcm.onnx`（create_artifact出力） | `CompilerModel` | `munc_digital` |

§3で述べた「`create_artifact`が忠実度を`munc_fp`→`munc_digital`に切り替える」という主張を、同一モデルの変換前後ファイルで直接確認した。

### 9.4 ノード数の対応（グラフ分割の整合性検証）

| ファイル | ノード数 |
|---|---|
| `source_model.onnx`（分割前） | 406 |
| `on_chip_1_bcm.onnx` | 387 |
| `off_chip_0.onnx` | 2 |
| `off_chip_2.onnx` | 14 |

`387 + 2 + 14 = 403`。差分3ノードは`source_model.onnx`にあった3個の`Mul`ノードに一致し（op種別カウントより）、`create_artifact`側の変換（`AdjustFirstConvForRGBToBGR`は無効化されているため対象外。恐らく`RemoveDummyChannelPadding`または未特定の吸収処理[推測、[to_acm.md](to_acm.md) §9.4でも同一の未解明点として記録]）で削除されたと考えられる。`Concat`ノード数は分割前30個に対し、分割後`on_chip_1_bcm`(27)+`off_chip_2`(3)=30個で**完全一致**しており、グラフ分割そのものはノード欠落なく行われている。

### 9.5 BEVFormerとの横断比較

`/tmp/bev_probe/compiler_ready_artifact/reference/compiler_ready_artifact_on_chip_1_bcm.onnx`も同様に`__type: CompilerModel`, `__mma_class: munc_digital`, `hardware_config: Denali`であることを確認済み（[to_acm.md](to_acm.md) §9.3）。2モデルで一貫した忠実度切替パターンが確認できたことは、`get_bcm_to_artifact_conversion_ops`がモデル非依存の共通ロジックであることの実測的裏付けである。

---

## 10. 参照ファイルと未解明点

### 抽出ソースの所在

| 分類 | ファイル |
|---|---|
| ディスパッチ | `mythic/model_zoo/common/conversion_steps.py`（`create_training_artifact`, `create_training_artifact_step`） |
| モデル変換op列 | `munc/_session.py`（`get_bcm_to_artifact_conversion_ops`） |
| グラフ分割 | `munc/_artifact/graph_splitter.py`（`split_onnx_graph`） |
| アーティファクト生成本体 | `munc/_generate_artifact.py`（`generate_artifact`, `generate_standard_artifact_files`等） |
| パッキング/アンパッキング | `munc/_artifact/artifact_writer.py`（`Artifact`クラス, `write_artifact`, `write_artifact_metadata`） |
| 検証 | `munc/_verify.py`（`verify_compiler_model`） |
| 実測データ | `mythic_ppa_explore_2605_2b`コンテナの`/tmp/yx_probe/`, `/tmp/bev_probe/` |

### 未解明点

1. §9.4で確認した「分割前406ノード→分割後403ノード（Mul 3個相当が消失）」の正確な削除op経路は未特定。`get_bcm_to_artifact_conversion_ops`のop列（§5）には該当しそうなopが見当たらないため、グラフ分割自体（`graph_splitter`）またはoff-chip専用の後処理の可能性がある。
2. `contents.json`（アーティファクト全体のメタデータ）を実データで直接確認できていない（§9.1の注）。`munc_artifact_version`の実際の値、`ris_pipeline_artifacts`/`ie_artifacts`のキー構造は`_generate_artifact.py`のソースコードからのみ確認した。
3. `generate_h5_artifact_files`が生成するHDF5データの実際のフォーマット・用途（コンパイラ側でどう使われるか）は未調査。
4. `remove_redundant_transition_muls`（yolopx）, `remove_red_shift`（zero_dce）の内部実装は本解析では未確認（関数呼び出しのみ確認）。
5. `synthetic`モデルの`create_artifact`実装は本解析の対象外（テスト/デモ用と推定されるが未確認）。
6. Boreas向けの`create_artifact`実測データは未取得（本ドキュメントの実測はいずれもDenali）。`verify_compiler_model`のBoreas固有チェック（外部入力数1〜3）が実際に機能する具体例は確認していない。
