# 06. `to_structural` ステップ解析

Mythic M2000 (Denali/ACE) アナログ compute-in-memory AI アクセラレータ SDK の **`to_structural` ステップ**の解析。対象バージョン `26.05.2`（SDK コンテナ `mythic-sdk-ubuntu-24.04:m2000-v26.05.2`, `mythic-model-zoo` の venv 内 `munc` パッケージ）。

主張はすべて実コードの**ファイルパス:行番号**を根拠に引用する。確定できない箇所は **[推測]** と明記する。抽出ソースの所在は §11 を参照。パスは特記なき限りコンテナ内 `/root/mythic_sdk/v26.05.2/mythic-model-zoo/` からの相対。

---

## 目次

- [1. `to_structural` とは何か（一言で）](#1-to_structural-とは何か一言で)
- [2. 入力と出力](#2-入力と出力)
- [3. 位置づけ — `step_order` と状態遷移](#3-位置づけ--step_order-と状態遷移)
- [4. ディスパッチ機構とモデル固有束縛](#4-ディスパッチ機構とモデル固有束縛)
- [5. 共通ペイロード — off-chip マーキング](#5-共通ペイロード--off-chip-マーキング)
- [6. off-chip 選択戦略の分類](#6-off-chip-選択戦略の分類)
- [7. モデル別処理詳細](#7-モデル別処理詳細)
- [8. モデル横断比較表](#8-モデル横断比較表)
- [9. 設定（config）の全体像](#9-設定configの全体像)
- [10. 実測](#10-実測)
- [11. 参照ファイルと未解明点](#11-参照ファイルと未解明点)

---

## 1. `to_structural` とは何か（一言で）

> **FP32 ONNX を Mythic フローが要求する形（静的 shape・整理済みグラフ・on/off-chip 区分の宣言）に整える。量子化もアナログノード変換も行わない。**

`step_order`（`configs/common/base_config_generic.yaml:37-50`）の先頭ステップであり、①再学習（doc00 §2.5）の最初の変換段。doc00 は本ステップを「ORIGINAL → structural へ状態を進める」（`00_overview.md:206`）と位置づけているが、**「structural」という状態は ONNX 上に実在する型ではない**（§3 で詳述）。

本ステップの本質は 2 点に尽きる:

1. **グラフの掃除・整理**（余分な出力エッジの除去、定数の畳み込み、shape の静的化 等）— モデルごとに内容が大きく異なる。
2. **on-chip / off-chip の**手動**宣言**（`__off_chip` 属性の付与）— これが全実装に共通する唯一のペイロード。

量子化（8bit power-of-two 固定小数点）・アナログノイズモデルの注入・`MythicConv2d`/`BCMConv2d` へのノード変換は**次のステップ `to_training` の担当**であり、`to_structural` はいずれも行わない（§11 の裏付け根拠を参照）。

---

## 2. 入力と出力

| 項目 | 実体 | 根拠 |
|---|---|---|
| 入力 (`src`) | FP32 ONNX（学習済みでない、`to_onnx` の出力） | `configs/common/base_config_generic.yaml:12,57` `fp_model: ${model_setup.fp_model}` |
| 出力 (`dest`) | 構造整理済み FP32 ONNX | `configs/common/base_config_generic.yaml:17,58` `structural_model: '${data_dir}/structural.onnx'` |

load/save は `munc.cli.helpers.SessionFromConfig`（`munc/cli/helpers.py:43-121`）が自動化する:

- `__init__`（:83-89）が `config['src']` から `ONNXModel` をロード。ファイルが無ければ `FileNotFoundError`。
- `__exit__`（:115-121）が `with` ブロックを**正常終了した場合のみ** `config['dest']` へ保存する:
  ```python
  def __exit__(self, exc_type, exc_val, exc_tb):
      if self.save_model and exc_type is None and self.config.get('dest'):
          dest_path = AnyPath(self.config['dest'])
          dest_path.parent.mkdir(parents=True, exist_ok=True)
          self.model.save(dest_path)
      return False
  ```
  例外発生時は保存されない（`return False` で例外は伝播）。`dest` キーが無ければ保存自体がスキップされる。

---

## 3. 位置づけ — `step_order` と状態遷移

doc00 §3.5（`00_overview.md:196-234`）は SDK コンテナ側のモデル状態を `MODELType`（`_constants.py:327-335`）で管理される連鎖として示す:

```
ORIGINAL ──to_structural──▶ (structural) ──to_training──▶ MYTHIC ──to_acm──▶ BCM ──create_artifact──▶ COMPILER
```

**「structural」が括弧付きなのは、ONNX ファイル上に対応する型が実在しないため**である。裏付け:

- `'__type'` メタデータを設定する箇所は SDK 全体で 3 か所のみ: `_session.py:330`（`MODELType.MYTHIC`）、`:414`（`MODELType.BCM`）、`:458`（`MODELType.COMPILER`）。ORIGINAL/structural を書き込む箇所は存在しない。
- 下流の型判定 `get_model_type()`（`_session_tools.py:458-481`）:
  ```python
  model_type = model.get_meta_data('__type')
  if not model_type:
      nodes_matrix_multiply = model.get_nodes_with_mythic_type(MYTHICType.MATRIXMULTIPLY)
      if len(nodes_matrix_multiply) == 0:
          model_type = MODELType.ORIGINAL
      else:
          model_type = _infer_model_type(model)
  return model_type
  ```
  メタデータが無く Mythic ノードも無ければ `ORIGINAL` と**推論**する。structural ONNX はこの推論経路で ORIGINAL と判定される。
- `to_training_step`（`common/conversion_steps.py:203` 経由で呼ばれる `get_original_to_mythic_conversion_ops`）は先頭で `assert_model_is(self._model, MODELType.ORIGINAL)`（`_session.py:227`）を実行し、これは structural ONNX に対して**推論経由で**通る。

**含意**: structural ONNX は ONNX として見れば依然 FP32 の標準オペのみで構成された ORIGINAL 相当のファイルであり、「structural」は SDK のワークフロー上の呼称（ファイル名・ステップ名）にすぎない。実測（§10）でも `metadata_props` が空であることを確認済み。

---

## 4. ディスパッチ機構とモデル固有束縛

### 4.1 ステップ名解決の仕組み

`run_conversion_step`（`munc/cli/helpers.py:327-355`）が実際のディスパッチを行う。要点（:343-355）:

```python
step_config = cfg[step]
OmegaConf.resolve(step_config)
step_config = copy(step_config)
with open_dict(step_config):
    step_type = step_config.pop('step_type', step)
step_impl = step_types[step_type]
num_params = len(inspect.signature(step_impl).parameters)
if num_params == 1:
    step_impl(step_config)
else:
    step_impl(step_config, step)
```

**`step_type` が config に無ければ、ステップ名自体が step_type として使われる**（`step_config.pop('step_type', step)`）。`step_types` は `configs/<model>/step_types/main.yaml` の YAML 文字列を `resolve_step_type_definitions`（:684-698）が `importlib` 経由で実 Python 関数に解決したもの。

### 4.2 munc 側に共通実装は存在しない

`configs/common/step_types/common.yaml`（全 12 行）に `to_structural` キーは無い。munc パッケージ全体を grep しても "structural" の語は 0 件。**`to_structural` は常にモデル固有の Python 関数**であり、6 モデルの束縛は互いに無関係:

| モデル | 束縛先（`configs/<model>/step_types/main.yaml`） |
|---|---|
| bevformer | `mythic.model_zoo.bevformer.conversion_steps.to_structural` |
| yolopx | `mythic.model_zoo.yolopx.conversion_steps.to_structural` |
| pythia | `mythic.model_zoo.pythia.conversion_steps.to_structural` |
| huggingface_classifiers | `mythic.model_zoo.huggingface_classifiers.conversion_steps.to_structural` |
| huggingface_robot_hand | 上記を再利用（`step_types/main.yaml:6`） |
| yolov8 | `mythic.model_zoo.yolov8.conversion_steps.adjust_fp_model`（**ステップ名と関数名が異なる**） |
| zero_dce | `mythic.model_zoo.zero_dce.conversion_steps.adjust_fp_model` |

`munc` パッケージが提供するのは「機構」のみ: `SessionFromConfig`（load/save）、`Session.change_opset()`、`ops.*` カタログ、`op_conf_seq`/`op_do`/`run_ops`、`graph_utils`/`_node_utils` の off-chip マーキング関数。各モデルリポジトリがこれらを個別に組み合わせる。

---

## 5. 共通ペイロード — off-chip マーキング

### 5.1 `__off_chip` 属性の実体

`munc/_node_utils.py:18` が属性名を定義:

```python
OFF_CHIP_ATTRIBUTE_NAME = '__off_chip'
```

`mark_off_chip`（:498-510）:

```python
def mark_off_chip(node):
    if not is_off_chip(node):
        value = ""
        if node.op_type == ONNXType.CAST:
            value = onnx_proto.TensorProto.STRING
        create_attribute_with_value(node, OFF_CHIP_ATTRIBUTE_NAME, value)
```

`NodeProto.attribute` に `__off_chip` エントリを追加する**存在フラグ**であり、値そのものに意味は無い（空文字列。`Cast` ノードだけは空文字列が `to` 属性の型と衝突しうるため `TensorProto.STRING` を値にする）。`create_attribute_with_value` は属性が既に存在すると例外を投げるため、`mark_off_chip` 自身は冪等（`if not is_off_chip(node)` で二重付与を防止）。

`is_off_chip`（:536-549）は存在チェックのみ:

```python
def is_off_chip(node):
    return is_attribute(node, OFF_CHIP_ATTRIBUTE_NAME)
```

**`mark_on_chip` は存在しない**。on-chip とは「`__off_chip` 属性が無いこと」であり、明示的なマークではない。

`munc.graph_utils` はこれらを `_node_utils` から再エクスポートするだけの薄い層（`munc/graph_utils/__init__.py:1-4`, `from ._node_utils import *`）。

### 5.2 op 種別による自動 off-chip 判定は `to_structural` では動かない

`ops.MarkUnsupportedOpsOffChip`（`munc/ops/mark_unsupported_ops_off_chip.py:22-37`）は、on-chip 対応 op 一覧（`SUPPORTED_ON_CHIP_NODES_BOREAS`/`_DENALI`, `_constants.py:245-304`）とノード属性の適合性（`_check_conv`/`_check_gemm`/`_check_max_pool`/`_check_slice`/`_check_resize`）から機械的に off-chip 判定を行う:

```python
def _run(self, nodes):
    if self.model.hwconfig is None:
        raise ValueError("Hardware configuration is not set. Please set the hardware configuration before running.")
    ...
```

**この op は `to_structural` の中で実行できない**。`SessionFromConfig.__init__`（`munc/cli/helpers.py:78-81`）が `hwconfig`/`noise_config` キーをアサーションで禁止しているため、`to_structural` 実行時の `model.hwconfig` は常に `None` であり、上記 `ValueError` が発生する。

`MarkUnsupportedOpsOffChip` が実際に走るのは `to_training` の内部（`_session.py:184-206` `_get_process_original_graph_ops()`）であり、これは `get_original_to_mythic_conversion_ops`（`_session.py:215-332`）の先頭付近、`set_hwconfig_metadata`（:222-224, `to_training_step` から `hardware_config_name` 経由で呼ばれる）で `hwconfig` が設定された**後**に実行される（:229）。

**結論**: `to_structural` が付ける `__off_chip` は**すべて人間（モデル実装者）が明示したもの**。op 種別に基づく機械的な off-chip 化は次段 `to_training` の担当である。doc00 §3.5 レベル A の表（`00_overview.md:257-261`）は「off-chip 化の確定は再学習の前」とまとめているが、その内訳は「`to_structural` での手動宣言」＋「`to_training` での自動判定（hwconfig 依存）」の 2 段階である。

---

## 6. off-chip 選択戦略の分類

`to_structural` 実装 6 種の off-chip 選択方法は次の 5 パターンに分かれる:

| 戦略 | 採用モデル | 詳細 |
|---|---|---|
| config のノード名リスト | yolopx（`off_chip_layers`, 12 ノード）、huggingface_classifiers（`model_setup.conversion_to_training.off_chip_layers`, 3 ノード。M2000 版は 1 ノード） | §9 参照 |
| コード内ハードコード名リスト | bevformer ResNet 側（`RESNET_OFFCHIP_NODES = ["/Reshape"]`, `bevformer/conversion_steps.py:260`） | §7.1 |
| 全ノード一括 | bevformer transformer 側（`everything_off_chip`, `bevformer/conversion_steps.py:246-248`） | §7.1 |
| 名前/op 種別ヒューリスティック | pythia（`query_key_value`/`dense` を含む名前、`Sum`、`Relu` は on-chip 維持。他は off-chip） | §7.3 |
| グラフトポロジ DFS | yolov8（出力から後方 DFS で最終 Conv 以降を off-chip）、zero_dce（`dfs_forward_from_edges` で前方 DFS） | §7.5, §7.6 |

**config で off-chip リストをユーザー設定可能にしているのは 8 個の model config のうち 2 モデル（yolopx / huggingface_classifiers 系列）のみ**。config 内の `off_chip_layers` 定義は全体で 4 箇所（yolopx / `resnet50_imagenet.yaml` / `resnet50_imagenet_m2000.yaml` / `robot_hand.yaml`）。

**ノード名リストは ONNX export 時のノード命名に強く結合し、脆い**:

- `configs/huggingface_robot_hand/training/model_setup/robot_hand.yaml:27` に旧アンダースコア形式のリストがコメントで残存: `# off_chip_layers: ['_fc_fc_Gemm', '_fc_Slice_1', '_fc_Slice']`（現行は `/fc/fc/Gemm` 形式）。
- resnet50 は fp32 ファイルを差し替えると（HF export → opset11 export）ノード名の付与規則が変わり、3 個のパス形式名（`resnet50_imagenet.yaml:17`）から 1 個の `Gemm_174`（`resnet50_imagenet_m2000.yaml:7`）へ変化する。
- yolopx の `mark_nodes_off_chip`（`yolopx/conversion_steps.py:26-29`）は null チェックを行わず、存在しないノード名を指定すると `mark_off_chip(None)` で `AttributeError` になる。

---

## 7. モデル別処理詳細

### 7.1 bevformer（`mythic/model_zoo/bevformer/conversion_steps.py:224-411`）

唯一、グラフを 2 分割して個別処理し再結合する実装。

**入出力パス派生**（:251-269）。`SAVE_FP32_SUBGRAPHS = False`（:234, デバッグ用フラグ）により通常運用では FP32 分割サブグラフは保存されない。常に保存されるのは定数畳み込み後のサブグラフ 2 個（`*-resnet-subgraph-const-folded.onnx`, `*-transformer-subgraph-const-folded.onnx`）と structural 版サブグラフ 2 個。すなわち通常の 1 回の実行で**中間 ONNX を 4 個** `src` の隣に書き出す。

**手順**:

1. `full_model = onnx_from_path(FULL_MODEL_SRC_PATH)`（polygraphy）。ファイル名から解像度（`800x450` / `1600x900`）を正規表現で抽出（:314-316）。`SHAPE_MAPPING`（:271-280）でその解像度の `img`/`img_neck_output` shape を取得。
2. **ResNet 側サブグラフ抽出**（:319-327）: `extract_subgraph`（polygraphy）で入力 `img`・出力 `/img_neck/fpn_convs.0/conv/Conv_output_0` を切り出す。
3. **定数畳み込み**（:284-286, 332-339）: コメントに理由あり:
   ```
   # If we let it fold everything that appears constant, it can fold ops still needed for retraining.
   # We use fold_constants from graph_surgeon instead of polygraphy for lower level API
   ```
   `PERMIT_FOLDING_OPS`（:287-302）= `{Identity, Unsqueeze, Squeeze, Shape, ConstantOfShape, Transpose, Reshape, Flatten, Concat, Split, Pad, Slice, Expand, Cast}`（15 種、いずれも純形状/レイアウトオペ）。`node_filter`（:304-305）はこれを**反転**して除外条件にする:
   ```python
   def node_filter(node: gs.Node) -> bool:
       return node.op not in PERMIT_FOLDING_OPS
   ```
   実際の呼び出しは `onnx_graphsurgeon`（`gs.import_onnx(...).fold_constants(should_exclude_node=node_filter, partitioning="recursive", fold_shapes=True).cleanup()`）。重みを持つ Conv/MatMul/Add/BN 等は畳み込みの対象から除外され、再学習可能性を保つ。
4. **transformer 側サブグラフ抽出**（:342-364）: 入力 5 個（`can_bus (1,18)`, `lidar2img (1,6,4,4)`, `prev_bev (1,2500,256)`, `use_prev_bev (1,)`, 継ぎ目テンソル `/img_neck/fpn_convs.0/conv/Conv_output_0`）、出力 3 個（`bev_embed`, `outputs_classes`, `outputs_coords`）。同様に定数畳み込み（:368-375）。
5. **ResNet 側 off-chip マーキング**（:379-389）: `RESNET_OFFCHIP_NODES = ["/Reshape"]`（:260。`(1,6,3,H,W)`→`(6,3,H,W)` の初期 reshape 1 ノードのみ）を munc `Session` 経由で `mark_off_chip`。`session.model.verify()`（`onnx.checker.check_model` 相当）で検証。
6. **transformer 側 off-chip マーキング**（:393-403）: `everything_off_chip`（:246-248）で全ノードを一括 off-chip 化。
7. **再結合**（:406-411）: `onnx.compose.merge_models`（`from onnx.compose import merge_models`, :20）で継ぎ目を io_map 指定して融合:
   ```python
   merged = merge_models(
       resnet, transformer, [("/img_neck/fpn_convs.0/conv/Conv_output_0", "/img_neck/fpn_convs.0/conv/Conv_output_0")]
   )
   onnx.save(merged, STRUCTURAL_MODEL_DEST_PATH)
   ```
   io_map に指定されたテンソルは融合後、モデルの入出力から消える（`merge_models` の仕様）。両サブグラフとも export 時点で opset 20（`MUNC_INTERNAL_ONNX_OPSET`, `_constants.py:312`）のため opset の食い違いは無い。

**munc `ops.*` は一切呼ばない**。使うのは `op_do`/`op_conf_seq` のラッパーと `_node_utils.mark_off_chip` のみ。`sess.change_opset()` も `ops.GeneralizeBatchSize` も実行しない（前者は export が既に opset 20、後者は `configs/bevformer/bevformer_tiny.yaml` で明示的に無効化されている。理由コメントは「batch size not consistently the first dimension」）。

### 7.2 yolopx（`mythic/model_zoo/yolopx/conversion_steps.py:19-39`）

最小構成。docstring も無い。

```python
def to_structural(config: DictConfig):
    def clean_up_export_onnx(model: ONNXModel):
        for node in model.get_nodes():
            while len(node.output) > 1:
                node.output.pop()

    def mark_nodes_off_chip(model: ONNXModel, off_chip_nodes: Sequence[str]):
        for node_name in off_chip_nodes:
            mark_off_chip(model.get_node_with_name(node_name))

    def convert(sess):
        ops = op_conf_seq(
            op_do(lambda: clean_up_export_onnx(sess.model)),
            op_do(lambda: mark_nodes_off_chip(sess.model, config.off_chip_layers)))
        sess.run_ops(*ops)
    run_conversion(config, convert)
```

- `clean_up_export_onnx`（:20-24）は torch.onnx export が残す余分な出力エッジをすべて除去。
- `mark_nodes_off_chip`（:26-29）は `config.off_chip_layers`（`configs/yolopx/base_config.yaml:53-56`、検出ヘッドの 12 ノード）を名前指定で off-chip 化。前述の通り null チェック無し。
- munc `ops.*` は使わない（局所変数 `ops` が `munc.ops` モジュールを覆い隠す）。

### 7.3 pythia（`mythic/model_zoo/pythia/conversion_steps.py:441-692`）

最も侵襲的な実装。ネストしたヘルパー 11 個を持つ。`convert(sess)` の本体（:670-691、要約）:

```python
ops_list = [op_do(lambda: clean_up_export_onnx(sess.model))]
if not config.use_kv_cache:
    ops_list.extend([
        op_do(lambda: fix_sequence_length(sess.model, config.block_size)),
        op_do(lambda: simplify_inputs(sess.model, config.block_size)),
        ops.ConstantFold,
    ])
ops_list.extend([
    op_do(lambda: fuse_attention(sess.model, config.num_transformer_layers, use_kv_cache=config.use_kv_cache)),
    op_do(lambda: remove_identity_nodes(sess.model)),
    op_do(lambda: ensure_one_node_per_initializer(sess.model)),
    op_do(lambda: merge_adds_to_sum(sess.model, config.num_transformer_layers)),
    op_do(lambda: insert_section_splitting_identity_nodes(sess.model, config.num_transformer_layers)),
    op_do(lambda: mark_off_chip_if_not_linear(sess.model)),
    op_do(lambda: mark_non_gemm_params_as_trainable(sess.model)),
    op_do(lambda: mark_relu_off_chip(sess.model)),
    ops.RemoveDanglingNodes,
])
```

要点:

- **`fix_sequence_length`**（:485-490）が全入力の seq 次元を静的化、**`simplify_inputs`**（:453-483）が `attention_mask`/`position_ids` を入力から削除し `Shape`+`Expand` で in-graph 合成（→ 単一入力 `input_ids` 化）。KV キャッシュ利用時はこの 2 手順と `ops.ConstantFold` をスキップする。
- **`fuse_attention`**（:514-547）が GPT-NeoX の attention 部分木を、ハードコードされたエッジ名パターン（`/gpt_neox/layers.N/attention/...`）を基準に単一の `Attention` ノードへ融合し、**`mythic` ドメイン opset v1 を新規登録する**（:545,547）:
  ```python
  fused_node.domain = "mythic"
  ...
  model._model_proto.opset_import.append(onnx.helper.make_opsetid("mythic", 1))
  ```
  以降このグラフは標準 ONNX ドメインだけでは checker を通らない。
- **`ensure_one_node_per_initializer`**（:645-668）が initializer の重複消費を解消（1 initializer = 1 consumer に強制）、**`mark_non_gemm_params_as_trainable`**（:606-620）が `__trainable` 属性を付与、**`merge_adds_to_sum`**（:549-576）が残差 `Add`×2 を `Sum` に統合、**`insert_section_splitting_identity_nodes`**（:622-637）がブロック境界に `Identity` を挿入。
- **`mark_off_chip_if_not_linear`**（:578-587）が名前/op 種別ヒューリスティックで off-chip 化: 名前に `query_key_value`/`dense` を含むノード、および `Sum`/`Relu` op を on-chip 維持、他はすべて off-chip。**`mark_relu_off_chip`**（:639-643）が非 MLP の `Relu` を追加で off-chip 化（前段の一括維持を絞り込む）。

munc `ops.*` の使用は `ops.ConstantFold`（KV キャッシュ非使用時のみ）と `ops.RemoveDanglingNodes` の 2 つ。`sess.change_opset()` は呼ばない（export が既に opset 20, :437）。

### 7.4 huggingface_classifiers（`mythic/model_zoo/huggingface_classifiers/conversion_steps.py:64-74`）

```python
def to_structural(config: DictConfig):
    def convert(sess):
        ops = op_conf_seq(
            op_do(lambda: resize_inputs(config.model_setup.input_sizes, sess.model)),
            op_do(lambda: mark_nodes_off_chip(sess.model, config.conversion_parameters.off_chip_layers)),
        )
        sess.change_opset()
        sess.run_ops(*ops)
    run_conversion(config, convert)
```

実行順は `sess.change_opset()` が `run_ops` より先に呼ばれる（opset 変更 → 入力 shape 固定 → off-chip マーキング）。

- `resize_inputs`（:44-47）: `model.set_edge_type_and_shape(input, shape=...)`。docstring 曰く "since they are left unspecified in Huggingface ONNX export" — HF export が入力 shape を未指定にするための補正。
- `mark_nodes_off_chip`（:50-53）: config の名前リストで off-chip 化。
- `sess.change_opset()` → `get_change_opset_ops()`（`_session.py:462-481`）が実際に munc `ops.*` を動かす: `ops.SanityCheckOffChipMarking, ops.AutoNameNodes, ops.MakeEdgesUnique, ops.ConvertConstsToInitializers, ops.RemoveDanglingNodes` ＋ `MUNC_INTERNAL_ONNX_OPSET`（=20）への `ChangeOpset`。

### 7.5 yolov8（`adjust_fp_model`, `mythic/model_zoo/yolov8/conversion_steps.py:45-67`）

`run_conversion` ではなく `SessionFromConfig` を直接使う:

```python
with SessionFromConfig(config, allow_other_keys=True) as s:
    m = s.model
    s.change_opset()
    s.node_filter = lambda node: node.op_type == ONNXType.IDENTITY
    s.run_all_nodes(ops.RemoveOpsWithConstantOutputs())
    s.node_filter = lambda node: True
    s.run_ops(ops.RemoveShapeInferenceNodes(), ops.InferStoreTensorShapes(),
              ops.GeneralizeBatchSize(), ops.InferStoreTensorShapes())
    get_task_config(task).adjust_fp_model(s, m, config)
    s.run_ops(ops.InferStoreTensorShapes(update_outputs=True))
    onnx.checker.check_model(m._model_proto)
```

`node_filter` を一時的に `Identity` のみへ絞って `RemoveOpsWithConstantOutputs`（統計駆動の定数除去、`munc/ops/remove_ops_with_constant_outputs.py:18`）を適用し、直後に元に戻す。タスク別（`detect`/`pose`/`segment`/`segpose`）の `adjust_fp_model` がモデル固有の構造修正を担い、例えば detect タスクは `put_everything_past_the_last_convs_off_chip`（`structural_mods.py:24-39`、出力から後方 DFS で最終 Conv 手前まで off-chip 化）を実行する。pose/segment は加えて `ops.AddNetworkOutputsForIntermediateNodes` で損失計算用の補助出力を追加。

**off-chip 選択はグラフトポロジ DFS**（config の名前リストではない）。config が渡すのは `task` のみ。

### 7.6 zero_dce（`adjust_fp_model`, `mythic/model_zoo/zero_dce/conversion_steps.py:102-114`）

yolov8 とほぼ同一の骨格（`RemoveOpsWithConstantOutputs` パスを除く）:

```python
with SessionFromConfig(config, allow_other_keys=True) as s:
    s.change_opset()
    s.run_ops(ops.RemoveShapeInferenceNodes(), ops.InferStoreTensorShapes(),
              ops.GeneralizeBatchSize(), ops.InferStoreTensorShapes())
    adjust_zero_dce_fp_model(s, config)
    s.run_ops(ops.InferStoreTensorShapes(update_outputs=True))
    onnx.checker.check_model(s.model._model_proto)
```

`adjust_zero_dce_fp_model`（`zero_dce/structural_mods.py:10-60`）が唯一**重み値を書き換える** `to_structural` 実装:

```python
first_mul_node = model.get_nodes()[0]
first_conv_node = model.get_nodes_with_op_type(ONNXType.CONV)[0]
first_conv_node.input[0] = first_mul_node.input[0]      # 入力を先頭Mulを経由せず直接Convへ
set_mythic_type(first_mul_node, MYTHICType.OFFCHIP_TRANSITION_SCALE)  # 後段への指示マーカー

first_mul_factor = model.get_initializer_np(first_mul_node.input[1])
first_conv_weights = model.get_initializer_np(first_conv_node.input[1])
model.set_initializer_np(first_conv_node.input[1], first_mul_factor * first_conv_weights)  # 係数をConv重みへ吸収
```

先頭の `Mul` を配線から外し、その乗数を最初の `Conv` の重みに吸収させることで数値的に等価な変換を行う。`set_mythic_type(..., OFFCHIP_TRANSITION_SCALE)`（:25）は munc の後続変換フローに「この Mul の位置に trainable off-chip 乗算器を挿入するな」と指示するマーカーであり、Mul をアナログ/BCM 演算へ変換するものではない。off-chip 化は `dfs_forward_from_edges`（前方 DFS）で AveragePool 出力以降・transition Mul 出力以降を対象にする（:52-60）。

---

## 8. モデル横断比較表

| 特徴 | bevformer | yolopx | pythia | hf_classifiers | yolov8 | zero_dce |
|---|---|---|---|---|---|---|
| opset 変更（`change_opset()` → 20） | 不要（export済み20） | 無し | 不要（export済み20）。**`mythic` ドメイン v1 追加** | **有り** | **有り** | **有り** |
| off-chip 選択 | ハードコード名リスト(ResNet) + 全ノード一括(transformer) | config 名リスト | 名前/op ヒューリスティック | config 名リスト | トポロジ DFS（後方） | トポロジ DFS（前方） |
| 定数畳み込み | **有り**（`gs.fold_constants` + ホワイトリスト） | 無し | **有り**（`ops.ConstantFold`、非KVキャッシュ時のみ） | 無し | 部分的（`RemoveOpsWithConstantOutputs`, Identity限定） | 無し |
| 部分グラフ抽出・再結合 | **有り**（`extract_subgraph` ×2 + `merge_models`） | 無し | 無し | 無し | 無し | 無し |
| shape/入力の固定 | 有り（`SHAPE_MAPPING`で抽出時指定） | 無し | **有り**（`fix_sequence_length`+`simplify_inputs`で入力削減） | **有り**（`resize_inputs`） | 間接（`InferStoreTensorShapes`） | 間接（`InferStoreTensorShapes`） |
| 重み/initializer の書き換え | 無し（畳み込み対象から除外） | 無し | **有り**（initializer重複排除・`__trainable`付与） | 無し | 無し | **有り**（Mul係数をConv重みに吸収） |
| shape 推論ノード除去 | 無し（Shapeは畳み込み対象） | 無し | 無し（`ConstantFold`がShapeも処理） | 無し | **有り**（`RemoveShapeInferenceNodes`） | **有り**（`RemoveShapeInferenceNodes`） |
| batch size 一般化 | **無し**（config で明示的に無効化） | 無し | 無し | 無し | **有り**（`GeneralizeBatchSize`） | **有り**（`GeneralizeBatchSize`） |
| 補助出力追加 | 無し | 無し | 無し | 無し | **有り**（検出/kpt/maskヘッド） | **有り**（Tanh出力） |
| 最終検証 | `model.verify()`（分割毎） | 無し | 無し | 無し | `onnx.checker.check_model` | `onnx.checker.check_model` |
| ドライバ | `run_conversion` ×2 + 生 onnx I/O | `run_conversion` | `run_conversion` | `run_conversion` | `SessionFromConfig` | `SessionFromConfig` |

---

## 9. 設定（config）の全体像

### 9.1 knob の 2 系統

`to_structural:` ブロック配下のキーは 2 系統に分かれる:

1. **`SessionFromConfig` → `Session.__init__` に渡される knob**（`src`/`dest`/`torchnet` を除いた残り）。`allow_other_keys=True` 時は `Session.__init__` の引数名（`stats, verbose, loader, stat_clipping_percentile, stat_n_samples_default, stat_shuffle, device_name, qat, debug, node_filter, torchnet_layer_factory, activation_ckpt_config`, `munc/_session.py:34-48`）に無いキーは黙って捨てられる。
2. **ステップ関数が `config.xxx` として直接読む knob**（`off_chip_layers`, `model_setup`, `task`, `block_size` 等）。Session には渡らない。

### 9.2 generic 既定値

`configs/common/base_config_generic.yaml:57-60`:

```yaml
to_structural:
  src: ${fp_model}
  dest: ${structural_model}
  stat_n_samples_default: 10
```

`off_chip_layers` も `dataloader` も無く、ダミーローダ（`Session.__init__` → `_loader_tools.dummy_loader`, `_session.py:91`）で動く。

### 9.3 off_chip_layers を露出する config（全 4 箇所）

| config:行 | エントリ数 | 内容 |
|---|---|---|
| `configs/yolopx/base_config.yaml:53` | 12 | `to_structural:` ブロック内に直接記載。`/model.2/Concat*` 3 個 + 検出ヘッド Conv 9 個 |
| `configs/huggingface_classifiers/training/model_setup/resnet50_imagenet.yaml:17` | 3 | `model_setup.conversion_to_training.off_chip_layers`。stem Conv/Relu + 最終 classifier Gemm |
| `configs/huggingface_classifiers/training/model_setup/resnet50_imagenet_m2000.yaml:7` | 1 | 上記を `['Gemm_174']` で上書き（M2000 用 fp32 ファイルはノード命名規則が異なるため） |
| `configs/huggingface_robot_hand/training/model_setup/robot_hand.yaml:28` | 3 | `/fc/fc/Gemm`, `/fc/Slice_1`, `/fc/Slice`。旧アンダースコア形式が :27 にコメントで残存 |

huggingface_classifiers 系は `to_structural.conversion_parameters: ${model_setup.conversion_to_training}` という間接参照でこの値を受け取る（`training/base_config.yaml:58`）。

### 9.4 モデル固有ブロックの要約

| モデル | ブロック位置 | `src`/`dest` 上書き | 独自 knob |
|---|---|---|---|
| yolopx | `base_config.yaml:50` | 無し | `dataloader`, `off_chip_layers`（12） |
| bevformer | **コメントアウト**（`bevformer_tiny.yaml:108-111`、`# uses default config`） | 無し（実効的に generic の3キーのみ。実行時に Python 側で `config.src`/`dest` を書き換える） | 無し |
| huggingface_classifiers | `training/base_config.yaml:53-59` | 無し | `dataloader`, `stat_n_samples_default: 100`, `stat_clipping_percentile: 0.03`, `conversion_parameters`, `debug` |
| huggingface_robot_hand | `training/base_config.yaml:58-65` | **有り**（`initial_fp.onnx`→`untrained_fp.onnx`） | 同上 |
| pythia | `base_config.yaml:52-55` | 無し | `block_size`, `num_transformer_layers`, `use_kv_cache` |
| yolov8 | `training/base_config.yaml:55-56` | 無し | `task` のみ |
| zero_dce | **ブロック自体が存在しない** | 無し | 無し（generic のみ） |

bevformer の `bevformer_tiny.yaml:108-110` は次の通り:

```yaml
# mythic/model_zoo/bevformer/conversion_steps.py::to_structual
# uses default config
# to_structural:
```

**注**: `mythic/model_zoo/bevformer/README.md:172` は `configs/bevformer/bevformer_tiny.yaml::to_structural` を参照先として案内しているが、上記の通り当該ブロックはコメントアウトされている。実効設定は generic の 3 キーのみである（§11 で既存ドキュメントの陳腐化として言及）。

### 9.5 `step_order` の m2000 overlay パターン

`m2000.yaml` を持つのは huggingface_classifiers / yolov8(training) / zero_dce の 3 モデル（yolopx・bevformer・pythia には無い）。3 例とも `to_structural` は列の先頭に残したまま、次の一貫したパターンで短縮する: **`eval_fp`/`eval_digital`/`eval_signoff_v0p4`/`eval_signoff_v0p5` を削除し `eval_trained` を追加**。例（`configs/huggingface_classifiers/training/m2000.yaml:11-18`）:

```yaml
step_order:
  - to_structural
  - to_training
  - train
  - eval_trained
  - summarize_metrics
  - to_acm
  - create_artifact
  - compile
```

いずれの overlay も `to_structural:` ブロック自体は変更しない。

---

## 10. 実測

### 10.1 BEVFormer-Tiny 1600x900 — 実際の `to_structural` 実行成果物

再学習前（untrained）状態の BEVFormer に `steps=to_structural` を単体実行した際の before/after ペアが、SDK 稼働ホストの `/mnt/nvme_scratch/mythic_untrained_probe/` に実在する。実行時の Hydra スナップショット（`outputs/2026-08-04/04-58-58/.hydra/overrides.yaml`）で素性を確認済み:

```yaml
- steps=to_structural
- data_dir=/workspace/untrained_probe
- fp_model=/root/mythic_sdk/v26.05.2/models/training/bevformer/bevformer-tiny-fp32-1600x900.onnx
```

| ファイル | サイズ | 役割 |
|---|---|---|
| `bevformer-tiny-fp32-1600x900.onnx` | 137,102,019 B | `to_structural.src` |
| `structural-1600x900.onnx` | 139,607,407 B | `to_structural.dest`（2026-08-04 05:00、所要 ≈90秒） |

`onnx.load` で両ファイルを直接比較した結果:

| 項目 | FP32 | structural |
|---|---|---|
| ノード数 | 5,871 | **2,164**（−63%） |
| initializer 数 | 589 | **1,851** |
| `__off_chip` 付きノード数 | 0 | **1,990**（on-chip は 174） |
| opset_import | `[('',20)]` | `[('',20),('',20)]`（**重複エントリ**、`merge_models` の副産物） |
| `metadata_props` | 空 | 空（`__type` 無し。§3 の裏付け） |
| graph 入出力名 | `img,can_bus,lidar2img,prev_bev,use_prev_bev → bev_embed,outputs_classes,outputs_coords` | **完全に同一**（継ぎ目テンソルはモデル I/O に現れず内部化） |

op 種別の変化（非ゼロ差分のみ、定数畳み込みの効果）:

| op_type | FP32 | structural | 差分 |
|---|---|---|---|
| Constant | 2,139 | 0 | −2,139 |
| Unsqueeze | 1,053 | 202 | −851 |
| Concat | 275 | 86 | −189 |
| Transpose | 261 | 138 | −123 |
| Shape | 160 | 24 | −136 |
| ConstantOfShape | 66 | 0 | −66 |
| Squeeze | 76 | 20 | −56 |
| Slice | 158 | 116 | −42 |
| Split | 50 | 12 | −38 |
| Add | 309 | 272 | −37 |
| Expand | 84 | 76 | −8 |
| Gather | 60 | 53 | −7 |
| Reshape | 223 | 214 | −9 |
| Cast | 33 | 27 | −6 |

**重み保持オペは全て不変**（§11 の「純粋に構造のみ」の実測裏付け）:

| op_type | FP32 | structural |
|---|---|---|
| Conv | 55 | 55 |
| MatMul | 155 | 155 |
| Gemm | 6 | 6 |
| Softmax | 18 | 18 |
| LayerNormalization | 40 | 40 |
| Relu | 96 | 96 |
| Sigmoid | 8 | 8 |

on-chip に残る 174 ノードの op 種別は **Conv 55 / BatchNormalization 53 / Relu 49 / Add 16 / MaxPool 1**、範囲は `/ResNet/conv1/Conv` から `/img_neck/fpn_convs.0/conv/Conv` まで。これは `mythic/model_zoo/bevformer/README.md:178-179`「ResNet backbone を on-chip、transformer head を digital」という記述と厳密に一致する。`/Reshape`（`RESNET_OFFCHIP_NODES` 指定ノード）は `__off_chip=True`。継ぎ目テンソル `/img_neck/fpn_convs.0/conv/Conv_output_0` はグラフ入出力ではなく `/Reshape_1` の入力として内部接続されている（`merge_models` の io_map 融合の実効果）。

**注**: bevformer の中間サブグラフ 4 ファイルは実行後に削除済みで差分が取れない。ファイル名のみ `convert_model.log` に記録が残る。

### 10.2 resnet50 — huggingface_classifiers 相当の再現実験

SDK コンテナ内で `resnet50_0.8098_opset11_224x224.onnx`（FP32, 102,157,968 B）に対し `to_structural`（§7.4）を素の Python で再現:

```python
from munc.cli.helpers import SessionFromConfig
from munc.op_config import op_conf_seq, op_do
from munc.graph_utils import mark_off_chip

OFFCHIP = ["/resnet/embedder/embedder/convolution/Conv",
           "/resnet/embedder/embedder/activation/Relu",
           "/classifier/classifier.1/Gemm"]
cfg = {"src": SRC, "dest": DEST, "stat_n_samples_default": 100}
with SessionFromConfig(cfg, None, allow_other_keys=True) as sess:
    sess.model.set_edge_type_and_shape("pixel_values", shape=[-1, 3, 224, 224])
    for name in OFFCHIP:
        mark_off_chip(sess.model.get_node_with_name(name))
    sess.change_opset()
```

結果:

| 項目 | FP32 | structural |
|---|---|---|
| ノード数 | 122 | **122**（**不変**） |
| opset | 18 | **20** |
| 入力 `pixel_values` shape | `['batch_size','num_channels','height','width']` | `[-1,3,224,224]` |
| `__off_chip` 付きノード | 0 | 3（指定ノードのみ） |
| `metadata_props` | 空 | 空 |

**bevformer はノード数が 63% 減るのに対し resnet50 は 1 個も減らない**という対比が、「`to_structural` は単一のアルゴリズムではない」（§4.2）ことを最も端的に示す実測材料である。

---

## 11. 参照ファイルと未解明点

### 抽出ソースの所在

| 分類 | ファイル |
|---|---|
| ディスパッチ・オーケストレーション | `munc/cli/helpers.py`（`SessionFromConfig`, `run_conversion_step`, `run_conversion_steps`） |
| off-chip マーキング基盤 | `munc/_node_utils.py`（`mark_off_chip`, `is_off_chip`, `OFF_CHIP_ATTRIBUTE_NAME`） |
| 自動 off-chip 判定（`to_training` 側） | `munc/ops/mark_unsupported_ops_off_chip.py`, `munc/_session_tools.py`（`is_op_type_supported_on_chip`） |
| Session 基盤 | `munc/_session.py`（`_get_process_original_graph_ops`, `get_change_opset_ops`, `get_original_to_mythic_conversion_ops`） |
| モデル固有実装 | `mythic/model_zoo/{bevformer,yolopx,pythia,huggingface_classifiers,yolov8,zero_dce}/conversion_steps.py` |
| yolov8/zero_dce の構造修正詳細 | `mythic/model_zoo/yolov8/{detect,pose,segment}_structural_mods.py`, `mythic/model_zoo/zero_dce/structural_mods.py` |
| config | `configs/common/base_config_generic.yaml`, `configs/<model>/base_config.yaml`, `configs/<model>/m2000.yaml` |

### 「`to_structural` が量子化・アナログ変換を行わない」ことの確認

`get_original_to_mythic_conversion_ops`（Mythic ノードへの変換一式を含む、`_session.py:215-332`）の呼び出し元を全リポジトリで grep すると、**唯一の呼び出し元は `to_training_step`**（`mythic/model_zoo/common/conversion_steps.py:203`）である。6 個の `to_structural`/`adjust_fp_model` 実装のいずれからも呼ばれていない。`ops.ConvertNodesToMythic` も model_zoo 側からは一切呼ばれない（`_session.py` 内部でのみ使用）。§10.1 の実測（重み保持オペ数が FP32/structural で完全一致）はこれを裏付ける。

### 未解明点

1. bevformer の中間サブグラフ 4 ファイル（`*-resnet-subgraph.onnx` 等）は削除済みのため、抽出→畳み込み→マーキングの各段階を個別に差分できていない。再実行すれば取得可能。
2. yolov8/zero_dce のタスク別構造修正（`detect_structural_mods.py` 等）は本ドキュメントでは概要のみ記載。損失計算用補助出力の完全な仕様は未確認。
3. pythia の `fuse_attention` が KV キャッシュ利用時（`use_kv_cache=True`）に辿る経路（`present.N.key`/`present.N.value` 分岐）は本文で触れたが実行トレースでは未検証。
4. `configs/bevformer/bevformer_tiny.yaml` の `to_structural` コメントアウトが、`README.md:172` の記述を追随し忘れた結果か意図的な整理漏れかは不明。
5. `robot_hand.yaml:27` の旧命名リストがいつのバージョンの ONNX export に対応していたかは未調査。
