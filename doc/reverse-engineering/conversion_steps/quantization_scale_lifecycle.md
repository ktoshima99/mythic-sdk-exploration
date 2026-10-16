# 量子化スケール(WSF/pFSR/iFSR/DSF)のライフサイクル — FP32モデルへの組み込みの全体像

`to_structural`→`to_training`→`train`→`to_acm`を横断して、「元のFP32 ONNXモデルに対して何が追加され、いつ追加され、その値がどう決まるのか」を1枚にまとめたクロスカット・ドキュメント。[to_training.md](to_training.md)（特に§5.4系）の詳細を前提とし、重複する derivation は再掲せず相互参照する。[to_structural.md](to_structural.md)・`train`ステップ（本解析で新規に踏み込んだ範囲）・[to_acm.md](to_acm.md)の該当箇所も統合する。

主張はすべて実コードの**ファイルパス:行番号**を根拠に引用する。確定できない箇所は**[推測]**と明記する。パスは[to_training.md](to_training.md)と表記を揃えて`munc/...`と記す。

---

## 目次

- [1. 根本思想 — ハードウェア制約と量子化誤差の分離](#1-根本思想--ハードウェア制約と量子化誤差の分離)
- [2. 全体の時系列: 何が・いつ・どう決まるか](#2-全体の時系列-何がいつどう決まるか)
- [3. `to_structural` — スケールには関与しない](#3-to_structural--スケールには関与しない)
- [4. `to_training` — スケールに関する全ての追加が起きる](#4-to_training--スケールに関する全ての追加が起きる)
- [5. `train` — 新規ノードを増やさず、丸め・ノイズ・DSFだけが動く](#5-train--新規ノードを増やさず丸めノイズdsfだけが動く)
- [6. `to_acm` — スケール因子が実際の物理表現に変換される（参考）](#6-to_acm--スケール因子が実際の物理表現に変換される参考)
- [7. 参照ファイルと未解明点](#7-参照ファイルと未解明点)

---

## 1. 根本思想 — ハードウェア制約と量子化誤差の分離

Mythicのアナログcompute-in-memoryチップは、ソフトウェア量子化と異なり**任意の場所でスケールを変更できない**。重みの書き込み(NVM, WSF)、クロスバー入力(iFSR)、ADC出力(pFSR)、デジタル後段(DSF)という、物理的に決まった数カ所の"ノブ"でしかスケールを調整できない（[to_training.md](to_training.md) §5.4.1/§5.4.3）。この制約から、以下の設計方針が導かれている。

1. **「どこで変えられるか」(構造)と「いくつに変えるか」(数値)を分離する**。前者は`to_structural`/`to_training`の固定opシーケンスが静的に決める構造的な話、後者はキャリブレーション/学習で決まる数値の話（本ドキュメント§2の表の列区分そのもの）。
2. **グラフを「スケーリングノードで区切られた区間」に分解**し（`ScaleAllNodes`の`_group_adjacent_edges`、[to_training.md §5.4.2](to_training.md#542-scaleallnodes-の内部アルゴリズム深掘り§11-旧未解明点1の解消)）、各区間に許される**ただ1つの数値**を、区間内の最も厳しい制約（最小値）で決める。クリッピング(情報破壊)は常に分解能の無駄(軽微な劣化)より悪いものとして扱われる。
3. **複数の物理ノブ（WSF・pFSR・iFSR・DSF）に分担させる**。離散的で自由度の低いノブ(pFSR/iFSR)はベストエフォートで近似し、連続値を取れるノブ(WSF, DSF)が余りを吸収する。縮小方向(安全)は重みに、拡大方向(クリップの危険)はできるだけデジタル乗数(DSF)に逃がす（`BreakFSRIntoPFSRAndIFSR._run`, `munc/ops/break_FSR_into_pFSR_and_iFSR.py:143-183`、[to_training.md §5.4.3-4](to_training.md)）。
4. **`to_training`では量子化誤差を一切発生させない**。重みのスケール変更も活性化のスケール整合も、FP32のまま「掛けてすぐ割り戻す」数学的に正確な再パラメータ化で行う（[to_training.md §7](to_training.md#7-量子化の実体--数値変更されるのはスケールのみ丸めノイズ注入は実行時)で確認済み：dtypeはfloat32不変）。丸め・ノイズ注入という本当の精度劣化は、できるだけ後段（`train`シミュレーション、実機）まで遅延させる。
5. **劣化が実際に発生する箇所（round, clip, ノイズ）には、学習が破綻しないための数学的な仕掛け（Straight-Through Estimator）を用意**し、量子化後も通常の勾配降下でネットワーク全体を再学習できるようにする（§5）。クリップだけは意図的にSTEから外し、物理的に不可能な値への勾配を本当にゼロにする。
6. **ゼロ点を持たない対称量子化に統一**し、符号あり/符号なしの切り替えとFP32レベルのバイアス補正付きシフト（`AddInputShifting`）で、アフィン量子化相当のことを量子化パラメータを複雑にせずに実現する（`munc/_constants.py`の`HWConfig.signed`、§4参照）。

一言でまとめると：**「ハードウェアの物理制約を満たすための数値操作は全部静的かつ無誤差で済ませ、本当に避けられない誤差（丸め・ノイズ）だけを学習で吸収させる」**という分業がMythicの哲学である。

### 1.1 AIモデルの視点から見たハードウェア制約の全体像

本ドキュメント全体で確認してきた制約を、AIモデル（計算グラフ）の視点から4つのカテゴリに整理する。**精度(A)・物理配置(C)・学習(D)はアナログ計算に比較的固有の制約だが、グラフ構造(B)の大半は実は一般的なデジタル量子化・専用ハードウェアにも共通する原則であり、本当にアナログに固有なのはそのうち2点だけ**である点に注意（§1.1.B参照）。

#### A. 精度・数値表現

- 重み（Denaliで実質16bit相当のデジタル表現、ノイズにより実効9〜10bit）と活性化（真の8bit）のbit幅は一致していない。アナログMACにはオペランドのbit幅を揃える制約が無いため（§前々回の会話）
- 対称量子化（zero_point無し）のみ。非負分布への対応は符号あり/符号なしの切り替えと`AddInputShifting`によるFP32レベルの補正に限られる（§前々回）
- pFSR/iFSRは離散的、かつ実運用ではモデル全体で1点に固定（§4.4実測）。層ごとに最適な物理レンジを選べない
- アナログノイズ（書き込みノイズ・ADC熱雑音・温度ドリフト）はデジタル量子化には存在しない誤差要因（§5.2）

#### B. グラフ構造・配置（ほとんどは一般的な量子化原則、アナログ固有は2点のみ）

一般的なデジタル量子化・専用ハードウェアにも共通する原則（§4.7、[前々回の会話]）:

- スケールは「壁」（透過性で判定）で区切られた区間内で1つしか持てない——演算の融合（fusion）・QDQラッピングという標準的な量子化工学と同じ
- ハードウェア上でスケール調整できる演算の種類が限定される——融合済みカーネルを持つ演算は無料、それ以外は追加のopが必要という、専用アクセラレータ一般の制約
- on-chip/off-chipの境界とそのコスト——専用アクセラレータが汎用CPUにフォールバックする、という一般的な構造

**真にアナログに固有な制約は以下の2点のみ**:

- **活性化の入力チャネル軸スケーリングが困難**（SmoothQuant相当）。クロスバー入力行（最大1280本/タイル）ごとに個別調整可能な駆動回路を用意できないため。出力チャネル軸（`OptimizeIFSR`）は既存のADC単位の構造を使えるので対応可能
- **pFSR/iFSRの物理的な柔軟性の欠如**。ソフトウェア定数ではなく物理的なアナログ回路のゲイン設定（つまみ）であるため、デジタルの量子化スケールのように層ごとに自由かつ連続的に変更できない

#### C. 物理的なタイル・配置（PPA観点）

- ACEタイルの物理サイズが固定（Denaliで1280入力×272出力）。層の次元がこれと合わないとタイル分割・利用率低下が起きる
- ACE1回の実行時間が固定（160ns）。タイル利用率に関わらず同じ時間がかかる

#### D. 学習・適応

- 学習（`train`）で調整できるのはDSFだけ。WSF・pFSR・iFSR・活性化のスケールは`to_training`の静的キャリブレーションで確定した後は変化しない
- これらの制約に対する頑健性はQATで初めて獲得される（§5、§10.4で確認した「学習無しでは精度崩壊」という実測結果）

---

## 2. 全体の時系列: 何が・いつ・どう決まるか

| ステップ | 追加されるもの | **役割（何のために存在するか）** | 形 | **挿入される場所（グラフ上の位置）** | **挿入・付与の判断基準** | 値の決定方法 | 詳細 |
|---|---|---|---|---|---|---|---|
| `to_structural` | `__off_chip`属性 | ハードウェア非対応・精度に不向きな演算をホストCPU実行に回す区分を宣言する | 既存ノードへの**存在フラグ**（値に意味はない空文字列） | 宣言対象となった個々の既存ノード（Conv/Gemm/Add等、ノード種別は問わない） | モデル実装者がconfigのノード名リスト・コード内ハードコードリスト・トポロジDFS等（モデルごとに異なる5種の戦略）で明示的に選んだノードのみ。op種別による自動判定は`hwconfig`未設定のためこの時点では不可能 | 計算ではない、手動宣言 | §3 |
| 同上 | グラフ構造の整理（BN folding、定数畳み込み等） | 後続のスケール計算が前提とする、標準化・単純化されたグラフ構造を用意する（スケール自体の計算ではない） | ノードの追加・削除・融合 | モデル全体（典型的にはBN直後のConvとのペア等、固定パターンに一致する部分木） | モデル固有スクリプトの固定実装（`munc`側に共通実装は無い）。スケールとは無関係の構造クリーンアップ | モデル依存の固定ロジック | §3 |
| 同上（モデル依存） | `mythic`ドメインopset、`__trainable`（非Gemmパラメータ用）等 | pythia固有の構造（Attention融合等）に対応するための準備 | opset宣言・属性 | pythiaのattention部分木（ハードコードされたエッジ名パターン`/gpt_neox/layers.N/attention/...`に一致する範囲） | 固定のエッジ名パターンマッチ。**他モデルには存在しない** | モデル固有のルールベース処理 | §3 |
| `to_training` | `Add(-offset)`シフトノード | 符号あり分布の活性化を、アナログMACが前提とする符号なし入力として使える形に変換する（量子化ではなくFP32レベルの再パラメータ化） | 新規ノード（Conv/Gemm手前） | 「非signedと判定されたConv/Gemm」の入力エッジの直前。新規Addが既存の入力エッジを乗っ取る形で挿入される | `_is_non_signed_conv_gemm`——対象Conv/Gemmが`MarkSignedNodes`で「signed」と判定されなかった場合のみ。signed判定済みのノードには挿入されない | 符号なしMAC対応の固定オフセット、biasを`+wgt·offset`補正 | §4.1 |
| 同上 | WSF（重みスケール） | 重みをNVMセルの表現範囲（`weight_min`/`weight_max`）に収める。4要素中、唯一「重みの数値そのもの」を変える | **既存Conv/Gemmの重み配列を直接上書き**（新規ノードなし） | 対象Conv/Gemmノード自身のweight/bias initializer | `ConvGemmWeightScaling`が走る全Conv/Gemm（on-chip/off-chip問わず`OFFCHIP_INCLUDE`）が対象。重み自体の分布だけで決まり、条件分岐はない | 重み自体の分布のpercentile/sigmaヒューリスティック | §4.2 |
| 同上 | `CompositeScale`ノード（一時的） | WSFによる出力のズレを一時的に打ち消す"仮置き"の値を保持し、後にCSF（§4.4）の最終値を運ぶ中継点になる | 新規ノード、後で削除 | 対象Conv/Gemmの**出力エッジの直前**（`_insert_node_before_edge`で既存の全consumerをこの新規ノード経由に差し替える） | `ConvGemmWeightScaling`が走った全Conv/Gemmノードに**1対1で**挿入される（例外なし） | 初期値`1/WSF`、後に`ScaleAllNodes`が上書き | §4.2/§4.4 |
| 同上 | 活性化スケーリングノード群（Mul、cancellation/restoration） | 「スケールを変えてよい地点」をグラフ上に明示する壁を立て、`ScaleAllNodes`のグルーピング境界を作る。非透過な演算（§4.7）の前後で、計算に必要な既知のレンジを保証する | 新規ノード、初期値1.0 | `Mul`(2テンソル入力)/`MatMul`/`Softmax`の**入力・出力エッジそれぞれ**、`Add`/`Sum`/`Concat`の**入力・出力エッジ**（条件付き） | **対象op種別は「後で`ConvertNodesToMythic`/`GroupSumQMul`がMythic化する演算」と完全一致**（`Mul→MythicQuantizedMul`, `MatMul→MythicMatMul`, `Softmax→MythicSoftmax`, `Add/Sum→MythicSum`）——これらはスケール変更の物理的手段を持つ演算に限られる。ただし吸収先での表現形式はConv/Gemmとは異なり、**単純な浮動小数点のテンソル入力としてそのまま使われる**（`__multiplier`/`__shift`のような固定小数点分解はされない。`mythic_sum.py:23`, `mythic_softmax.py:33,35`）。ReLU/MaxPool/Reshape等、Mythic化されてもスケール調整手段を持たない演算の境界には一切挿入されない。`MatMul`/`Softmax`/`Mul`(2テンソル)は無条件（`Mul`は`is_mul_without_constant_inputs`で定数乗算を除外）、`Add`/`Sum`/`Concat`は「入力元が既にスケーリングノードでない場合のみ」。**全てoff-chipノードは対象外**（`OFFCHIP_IGNORE`） | 挿入時は未確定（初期値1.0） | §4.3 |
| 同上 | 上記の値確定 | 各edge group（壁で区切られた区間）内の全エッジが、ハードウェアが要求するレンジに収まる、最も安全な共通スケールを決める | 既存initializer上書き（ノード増減なし） | 上記スケーリングノードのinitializerそのもの（ノード自体は不変） | スケーリングノードで区切られた連結成分（edge group）単位。ハードウェアtarget_rangeが明示的に定義されたエッジが混在する範囲ごとに、その中の最小スケール（最も厳しい制約）を採用 | キャリブレーション統計とtarget_rangeの比、edge group内最小値 | §4.3 |
| 同上 | `DigitalScale`ノード（→DSFの器） | DSF（ADC読み取り後のデジタル後段調整値）を保持する"器"を用意する。この時点ではまだ値は未確定 | 新規ノード | `CompositeScale`ノードの**出力エッジの直後**（`_insert_node_after_edge`） | `CompositeScale`が存在する全ノードに**1対1で**挿入される | 初期値1.0→上書き | §4.4 |
| 同上 | pFSR/iFSR | クロスバーに重みを書き込む側（pFSR）とADCが積和を読み取る側（iFSR）、アナログ計算の物理的なレンジを設定する。CSFを実現する離散的（かつ本解析ではモデル全体で固定）な物理ノブ | **既存Conv/Gemmの属性**（新規ノードなし） | `CompositeScale`ノードから**入力方向に遡って最初に見つかるConv/Gemmノード自身**の属性（`get_first_preceeding_node_with_op_type`） | `CompositeScale`が削除される全ノードに1対1で付与。値は離散候補リストの中からCSFに最も近い比を選択（実運用では候補1個に固定） | 離散候補リストから最も近い組を選択 | §4.4 |
| 同上 | DSFの初期値、`__multiplier`/`__shift` | pFSR/iFSR（離散的・固定）とWSF（重みのクリップ安全限界）だけでは実現できないCSFの"余り"を、ADC読み取り**後**のデジタル乗算で吸収する最後の調整項。4要素中唯一、学習でも調整される | `DigitalScale`ノードの属性 | `DigitalScale`ノードのinitializer（float値）→同じノードの属性（`__multiplier`/`__shift`） | pFSR/iFSRで吸収しきれなかった余りの大小で、WSFに回すかDSFに回すかが分岐する（§4.4の4段階の判断基準：安全な縮小→重み、危険な拡大→DSF優先、それでも不足なら両方の限界まで） | CSFの余り、`max_dsf`でクランプ | §4.4 |
| 同上 | ノード統合、`__trainable_dsf`属性 | グラフを最終的な実行形態（`MythicConv2d`等、1つの物理演算に対応する1ノード）に単純化し、DSFを学習パラメータとして識別可能にする | `DigitalScale`等が既存Conv/Gemmに吸収され消滅 | 既存Conv/Gemmノード**自身**（`DigitalScale`・活性化ノードがこのノードに吸収される） | `GroupMMAOps`が「Conv/Gemmに直接付随する既知のパターンのスケーリング/活性化ノード」を機械的に検出・吸収。`ConvertNodesToMythic`のop_typeマッピング（Conv→MythicConv2d等）に該当する全ノードに`__trainable_dsf`が付与される | `trainable_dsfs`パラメータ（学習グループID、固定値） | §4.5 |
| `train` | DSFの値更新 | 静的キャリブレーションだけでは解消できない誤差（§10.4で考察したpFSR/iFSR固定による実効精度低下等）を、タスクの損失を見ながら補正する | ノード追加なし、既存属性の数値更新 | 統合済みMythicノードの`__multiplier`/`__shift`相当の数値（ノード自体は不変） | `__trainable`マスクが付いた初期化子のみが対象（`PinLastOnChipConvDSF`等でピン留めされたものは除外） | 誤差逆伝播（勾配降下） | §5.1 |
| 同上 | 丸め・ノイズのシミュレーション | 実機で発生する量子化誤差・アナログノイズを学習時に"経験"させ、それに頑健な重みを獲得させる | 永続的なグラフ追加ではない。forward実行時のみ適用 | ONNXグラフには一切残らない。TorchNetレイヤー（`MythicConv2d`等のPyTorch実装）のforward計算内 | 各レイヤー実装（`BaseWeightModel`等）がquantize/clip/noiseをどの順で適用するか、ハードコードされたレイヤー実装次第（全対象ノードに適用、条件分岐なし） | STE付きround/clip/noise関数 | §5.2 |
| `to_acm`（参考） | 重みの実際の整数格子への丸め、ハードコード | `to_training`/`train`で確定した「対応表（スケール因子）」を、実際に物理的な数値表現へ変換・固定する | 既存initializerの最終上書き | 既存initializer自身 | §6既存記述（`to_acm.md`）に委ねる | スケール因子の確定→物理表現への変換 | §6 |

---

## 3. `to_structural` — スケールには関与しない

> **`to_structural`は量子化（8bit power-of-two固定小数点）・アナログノイズモデルの注入・Mythicノード変換のいずれも行わない**（[to_structural.md §1](to_structural.md#1-to_structural-とは何か一言で)）。

全モデル共通の唯一のペイロードは**on/off-chip区分の手動宣言**である。

```python
# munc/_node_utils.py:18
OFF_CHIP_ATTRIBUTE_NAME = '__off_chip'
```

`mark_off_chip`（`munc/_node_utils.py:498-510`）はこの属性を**存在するかどうかだけが意味を持つフラグ**として付与する（値は空文字列、`Cast`ノードのみ型衝突回避で`TensorProto.STRING`）。`is_off_chip`（:536-549）は存在チェックのみ。**`mark_on_chip`は存在しない**——on-chipとは「`__off_chip`属性が無いこと」である。

この宣言は**計算によって決まるものではなく**、モデル実装者がconfig（ノード名リスト）やモデル固有スクリプトのヒューリスティック（例: pythiaの`mark_off_chip_if_not_linear`, `mythic/model_zoo/pythia/conversion_steps.py:578-587`）で明示的に決める（[to_structural.md §5-6](to_structural.md)）。op種別による**自動**off-chip判定（`MarkUnsupportedOpsOffChip`）は、この時点では`hwconfig`が未設定のため実行できず（`ValueError`）、`to_training`の担当になる（[to_structural.md §5.2](to_structural.md#52-op-種別による自動-off-chip-判定は-to_structural-では動かない)）。

モデル固有の構造クリーンアップ（BN folding、定数畳み込み、shape静的化等）もこの段で行われるが、実装内容はモデルごとに大きく異なり（`munc`側に共通実装は存在しない、[to_structural.md §4.2](to_structural.md#42-munc-側に共通実装は存在しない)）、スケール計算とは無関係である。

一部モデル（pythia）はさらに侵襲的な構造変更を行う。`fuse_attention`がGPT-NeoXのattention部分木を単一`Attention`ノードへ融合する際、**`mythic`ドメインopset v1を新規登録**する（`mythic/model_zoo/pythia/conversion_steps.py:545,547`）。

```python
fused_node.domain = "mythic"
model._model_proto.opset_import.append(onnx.helper.make_opsetid("mythic", 1))
```

また`mark_non_gemm_params_as_trainable`（:606-620）が`__trainable`属性を付与する。これらは**全モデル共通ではなく**、pythia固有の構造的事情である（[to_structural.md §7.3](to_structural.md#73-pythiamythicmodel_zoopythiaconversion_stepspy441-692)）。

---

## 4. `to_training` — スケールに関する全ての追加が起きる

本章の全ての導出は[to_training.md](to_training.md) §5.2〜§5.6に詳細があるため、ここでは要点と相互参照のみを記す。

### 4.1 サインネス補正（§5.2）

`AddInputShifting`（`munc/ops/add_input_shifting.py`）は、符号あり分布をハードウェアの符号なしMAC前提に適合させるための**FP32レベルの再パラメータ化**を行う。クラスdocstringがそのまま仕様になっている。

```python
"""
Conv or Gemm computes activation according to:
    A = wgt . input + bias
We want to introduce a shift but keep the activation unchanged:
    A = wgt . (input - offset + offset) + bias
    A = wgt . input' + wgt . offset + bias
"""
```

`Add(-offset)`ノードを入力の手前に新規挿入し、`bias_new = wgt·offset + bias`（`_shift_bias`, :35-51）としてbiasを補正する。この操作は量子化ではなく、出力`A`の数値を完全に保ったままグラフを変形するだけである。これは**アフィン量子化のzero_point**とは異なる概念であることに注意（後述§4.5および会話内で既出の確認）。

### 4.2 WSF — 重みテンソルへの直接上書き（§5.2/§7、§5.4.3）

`ConvGemmWeightScaling`（`munc/ops/conv_gemm_weight_scaling.py:79-136`）が重み・バイアスのnumpy配列に`scale`を掛けて直接上書きする（新規ノードは増えない）。`scale`は**活性化統計を使わず**、重みテンソル自体の分布から`get_weight_scale`（percentile/sigmaヒューリスティック、`munc/_session_tools.py:779-860`）で決まる。直後に打ち消し用の`CompositeScale`ノード（初期値`1/scale`）が挿入される（`_inject_composite_scale_node`, :138-151）。

第2段の調整は`BreakFSRIntoPFSRAndIFSR`（§5.4）内で行われ、pFSR/iFSRで吸収しきれない余りのうち安全な分（≤1）がさらにWSFとして重みに追加で乗算される（[to_training.md §5.4.3-4](to_training.md)）。

### 4.3 活性化スケーリングノード群とScaleAllNodes（§5.3/§5.4.2）

`InjectScalingOn*`系op（§5.3）が、Mul/MatMul/Softmax/Addの入出力に初期値1.0のMulノード（スケーリングノード）を新規挿入する。これらは**「スケールを変えてよい"壁"」**としてグラフ上に明示的に設置され、`ScaleAllNodes`の連結成分探索（`_group_adjacent_edges`/`_scan_node`, [to_training.md §5.4.2](to_training.md)）における境界になる。

**実行順序に関する注意**: `ScaleAllNodes`（§5.4）が連結成分を探索する時点では、**ノードのMythicノードへの統合（§4.5の`GroupMMAOps`/`ConvertNodesToMythic`、§5.5）はまだ実行されていない**。つまりグルーピングの対象となるグラフでは、`ReLU`・`Conv`・`CompositeScale`・`DigitalScale`は全て別々の独立したノードのままである（`ConvGemmWeightScaling`による`CompositeScale`の挿入は§5.2、すなわち`ScaleAllNodes`より前に完了しているため、壁自体は既に高い密度で存在するが、ノードの"結合"という意味での単純化はまだ起きていない）。「活性化がMythicノードに吸収されているから連鎖が短い」という理解は、グルーピングの実行時点については誤りであり、連鎖が短くなる直接の理由は、§4.2で見た通り`ConvGemmWeightScaling`が**全てのConv/Gemmの出力直後に無条件で壁を挿入する**こと自体にある。また`Conv`/`Gemm`自身は`is_scaling_node`等のいずれにも含まれないため、グルーピングの観点では透過的に扱われる点に注意（§4.7）。

**この"壁"の置き場所はハードウェア制約の直接の表現である**。対象op種別（`Mul`/`MatMul`/`Softmax`/`Add`/`Sum`）は、ConvGemmWeightScaling（Conv/Gemm）と合わせて、`ConvertNodesToMythic`の`DEFAULT_MYTHIC_NODE_MAP`（`munc/ops/convert_nodes_to_mythic.py:26-32`）および`GroupSumQMul`（`munc/ops/group_sum_qmul.py`）が最終的にMythic化する演算と**完全に一致**する。つまりスケーリングノードは「スケールを変更できる何らかの物理的手段が実際に存在する演算の境界」にしか置かれない。逆に`ReLU`/`MaxPool`/`Reshape`等、Mythic化後も独自のスケール調整手段を持たない（直前のMythicノードの`__activation`属性等に吸収されるだけの）演算の境界には、スケーリングノードは一切挿入されない。

**ただし「物理的手段」の厳しさはop種別で異なり、WSF/pFSR/iFSR/DSFが全く同じ扱いを受けるわけではない**。実際のMythicレイヤー実装（`_o2t_ops/*.py`）を確認すると:

- `pFSR`/`iFSR`、および`__multiplier`/`__shift`という**厳密な固定小数点形のDSF**を使うのは`mythic_conv.py`/`mythic_linear.py`/`mythic_mma.py`（＝`Conv`/`Gemm`由来）**だけ**である。これはNVM書き込み・クロスバー入力・ADC出力というアナログMMA経路に物理的に実在する3つのノブ（§1）そのものであり、Conv/Gemm以外にはこの経路自体が存在しない。
- `Add`/`Sum`の出力に挿入された`ADD_SCALE_OUTPUT`のMulノードは、`GroupSumQMul`によって**MythicSumノードに吸収される**（:17-30、`sum_node`に`mul_node`の属性をコピーし`mul_node`を削除）が、吸収された先の`mythic_sum.py:23`を見ると、スケールは`__multiplier`/`__shift`に分解されず、**単純な浮動小数点のテンソル入力（initializer）としてそのまま使われる**（"Get the last input: the scale factor"）。`mythic_softmax.py:33,35`の`softmax_scale_in`/`softmax_scale_out`も同様に単純な浮動小数点乗算であり、`mythic_quantized_mul.py`/`mythic_matmul.py`にはpFSR/iFSR・`__multiplier`/`__shift`いずれも登場しない。

つまり**「スケール変更可能な地点に限って挿入される」という原則自体は全スケールに共通**だが、Conv/Gemmは「アナログMMA経路に実在する4つの物理ノブ（WSF・pFSR・iFSR・固定小数点DSF）」という最も厳しい制約を受け、Mul/MatMul/Sum/Softmaxは「デジタル演算なので任意精度の掛け算がついでに実行できる」という緩い制約（スケールは単なる浮動小数点定数で済む）しか受けない、という**制約の強さの違い**がある。

値は挿入時点では未確定で、`ScaleAllNodes`が後から確定させる。アルゴリズムは:

1. スケーリングノードを跨がない範囲でエッジをグルーピング（edge group）
2. 各エッジについて、ハードウェアtarget_range（`hw_specs.py`の`_BOREAS_TARGET_RANGES`/`_DENALI_TARGET_RANGES`）とキャリブレーション統計（`clip_min`/`clip_max`、約20サンプル）の比を計算
3. edge group内の最小値（最も厳しい制約）を区間全体のスケールとして採用し、各スケーリングノードのinitializerへ書き込む

Softmax/Sigmoid/Tanh等のrange-sensitiveな非線形関数の前後には、cancellation/restorationノードのペアが同様に挿入される。

### 4.4 CSFの決定とpFSR・iFSR・DSF・WSFへの分担（§5.4、§5.4.3-4で詳細済み）

#### CSFはどうやって決まるか

CSFは単一の式で一度に求まるのではなく、**§4.2（WSF）と§4.3（活性化スケーリング）の結果が1つのノード（`CompositeScale`）に重ね合わされて**決まる。

```
CSF = (1 / weight_scale)  ×  (scale_out / scale_in)
      └─ §4.2: WSFの副作用を打ち消す項 ─┘   └─ §4.3: ScaleAllNodesが計算する活性化側の補正 ─┘
```

- `weight_scale`は`ConvGemmWeightScaling`が重み自体の分布だけから計算した値（§4.2）。`CompositeScale`ノードの初期値`1/weight_scale`は、この重みスケーリングの副作用（Conv出力が`weight_scale`倍ズレること）を打ち消すための項。
- `scale_out`/`scale_in`は`ScaleAllNodes`の`_scale_node`が、`CompositeScale`ノード自身の出力/入力エッジに割り当てたスケール（§4.3）。`scale_out`は、`CompositeScale`ノードの出力エッジに対する明示的なハードウェアtarget_range（`hw_specs.py`の`is_composite_scale_output`述語、Denaliで`[-127.5,127.5]`）とキャリブレーション統計の比。`scale_in`はConv自身を透過して遡った上流側のスケール（Convはスケーリングノードではなく§4.3のグルーピングにおける"壁"にならないため、グループが上流まで拡張される）。

つまりCSFは、「**重みを勝手にスケールした分の後始末**」と「**この層の出力を次段が要求するレンジに合わせる補正**」という本来無関係な2つの要求を、たった1つの数値に押し込んだものである。これは、スケールを変更できる物理的な場所がこの1点（`CompositeScale`ノード、最終的にpFSR/iFSR/WSF/DSFへ分解される地点）に限られているという、§1で述べたハードウェア制約の直接の結果である。

#### CSFをWSF/pFSR/iFSR/DSFにどう分担するか — 考え方

`BreakFSRIntoPFSRAndIFSR`（`munc/ops/break_FSR_into_pFSR_and_iFSR.py:88-190`）が、確定したCSFを4つの物理ノブに分担させる。分担の判断基準は一貫して**「クリッピング（情報破壊）を避け、できるだけ安全な側に倒す」**ことである。

1. **まずpFSR/iFSRで近似する**: 両方とも離散的な候補リストしかない（Denali: `pFSR∈{1,3,5,10}`, `iFSR∈{1.25,...,20}`）ため、CSFにピッタリ一致する組はまず無い。候補の比`pFSR/iFSR`がCSFに最も近い組を選ぶ（`calculate_half_pfsr_ifsr_and_digital_scale`, `munc/_session_tools.py:202-243`）。この時点で必ず「ずれ」（余り）が残る。
2. **余りが1以下（縮小方向）ならWSFへ**: 重みを縮小するだけなのでクリッピングのリスクが無い。常に安全なので無条件にWSFへ回す（`break_FSR_into_pFSR_and_iFSR.py:146-148`「Scaling weights down is always safe」）。
3. **余りが1超（拡大方向）だが小さいならDSFへ**: 重みを拡大するとハードウェアの表現範囲（`[-128,127]`）を超えてクリップする危険があるが、DSFはデジタル乗算器なので拡大しても物理的な破綻がない。安全第一で、拡大が必要な分は極力DSF側に逃がす。
4. **それでも足りない場合はWSFの安全限界まで拡大し、残りをDSFの上限（`max_dsf`）まで積む**: 重みの実際のクリッピング安全限界（`get_weight_scale`のpercentile/sigmaヒューリスティック）までは重みを拡大してでも対応し、それでも残る分は`max_dsf`でキャップしたDSFに押し込む（[to_training.md §5.4.3](to_training.md)で実測した「DSFが`max_dsf=3`にちょうど一致している」ケースがこれに該当する）。

この優先順位——**「離散ノブで近似 → 安全な縮小は重みへ → 危険な拡大はできるだけデジタル乗数へ → それでも足りなければ重みの安全限界まで拡大」**——がMythicの量子化スケール配分の根本的な考え方である。

#### グラフ上での実現

`BreakCompositeScaleIntoFSRAndDigitalScales`が`CompositeScale`の直後に`DigitalScale`ノード（初期値1.0）を新規挿入する。`BreakFSRIntoPFSRAndIFSR`が最終的に:

- `__pFSR`/`__iFSR`属性を**既存のConv/Gemmノード自身**に書き込む（専用ノードは一度も作られない）。値は離散候補リストから、必要な合成スケール(CSF)に最も近い比になる組を選択する。実際のBEVFormer運用ではこの候補リストが各1個に固定されており、実質全モデルで同じ固定値になる（[to_training.md §5.4.3](to_training.md)の実測）。
- `CompositeScale`ノードを削除する。
- 余りを`DigitalScale`ノードのinitializerに掛けて上書きする（DSFの初期値）。

`BreakDigitalScalesIntoFactors`が`DigitalScale`ノードに`__multiplier`/`__shift`属性を追加し、`multiplier/2^shift`というハードウェア実装可能な形に分解する。

### 4.5 ノード統合と対称量子化（§5.5）

`GroupMMAOps`+`ConvertNodesToMythic`が、`DigitalScale`ノードと活性化ノードを既存Conv/Gemmノードへ**吸収**し、`MythicConv2d`等へリネームする。この際`__trainable_dsf`属性（`trainable_dsfs`パラメータ、既定値`_constants.DEFAULT_DSF_PARAMETER_GROUP`、`munc/ops/convert_nodes_to_mythic.py:33-65`）が付与される。

量子化式自体は常に`q = clip(round(x × scale), ...)`という**zero_point項を持たない対称量子化**である（`00_overview.md` §3「非対称ゼロ点なし」）。非負専用の分布（ReLU直後等）はzero_pointシフトではなく、符号なし8bitコード（`hwconfig.signed`フラグ）への切り替えで扱われる。§4.1の`AddInputShifting`はこれとは別の、FP32レベルの再パラメータ化である。

### 4.6 dequantizeのタイミング — オンチップ/オフチップ境界とcancellation/restoration

**前提の補足**: ACE内部の積算（WXをアナログで足し合わせる処理）自体はint8相当ではなく連続値（実質無限精度、§1）で行われる。「8bit相当」が意味を持つのは、ACEへの入力の手前（`input_bits=8`）とADCで読み出した直後（出力側）という**境界だけ**である。この前提の上で、「dequantize（量子化前のスケールに戻す操作）はどのタイミングで起きるか」を整理する。

**基本原則: オンチップのMythicノードからMythicノードへそのまま繋がる場合、dequantizeは発生しない。** これは`ScaleAllNodes`（§4.3）が「あるノードの出力スケールが、次のノードが期待するスケールにあらかじめ一致するよう」計算しているためであり、CSFを1つの数値（WSF×pFSR×DSF/iFSR）に統合する設計（§4.4）の目的そのものである。

dequantize（スケールを戻す操作）が明示的に挿入されるのは、以下の**特定の境界**だけである。

#### ① オンチップ→オフチップの境界（最も明確なdequantize）

`AddOnOffChipTransitionScaling`（`munc/ops/add_on_off_chip_transition_scaling.py`、§5.4で名前のみ既出）が、**on-chip/off-chipの境界エッジを全走査**し、両方向それぞれに専用のMulノード（初期値1.0、`ScaleAllNodes`が後で確定）を挿入する。

```python
def _run(self, nodes):
    for parent_node in list(self.model.get_nodes()):
        ...
        for child_node in ...:
            if parent_offchip_status is True:
                node_type = MYTHICType.ONCHIP_TRANSITION_SCALE    # off-chip→on-chip（quantize相当）
            else:
                node_type = MYTHICType.OFFCHIP_TRANSITION_SCALE   # on-chip→off-chip（dequantize相当）
            node = self.model._insert_node_after_edge(output, ONNXType.MUL, description, node_type, ...)
```

`OFFCHIP_TRANSITION_SCALE`（on-chip→off-chip方向）が、オンチップの量子化済み表現を、オフチップ（ホストCPU、任意精度float32、[to_acm.md §9.2](to_acm.md)で確認済みの「off-chip側は量子化されない」という事実と対応）が期待する実数値に戻す、**まさにdequantizeそのもの**である。逆方向`ONCHIP_TRANSITION_SCALE`はquantize（dequantizeの逆）に相当する。挿入されたノードは`_node_utils.mark_off_chip(node)`で自身も常にoff-chipマークされる（`ConvertNodesToMythic`等の対象から外れるため）。

#### ② Softmax/Sigmoid/Tanhの前後（局所的なdequantize/requantizeのペア）

§4.3のcancellation/restorationノードも同じ役割の局所版である。対象の非線形関数に入る直前のcancellationノードが「入ってきたスケールを一旦打ち消し、元のレンジに戻す」（≈dequantize）、直後のrestorationノードが「出力に再度スケールを掛け直す」（≈requantize）。これは、これらの関数自身が指数関数・除算という、ハードウェアスケールのままでは正しく計算できない演算だからである。

#### ③ ネットワークの最終出力

最後の層がオンチップで終わる場合、損失計算や推論結果として使うには最終的に①と同じ機構（またはモデルの出力ノード自体がoff-chip宣言されていればそこで自動的に処理される）でFP32相当の実数値に戻される。

#### まとめ

| タイミング | 方向 | 役割 |
|---|---|---|
| オンチップ→オンチップ（連続するMythicノード間） | — | **発生しない**（`ScaleAllNodes`が事前にスケールを一致させているため） |
| オンチップ→オフチップ | `OFFCHIP_TRANSITION_SCALE` | **dequantize** |
| オフチップ→オンチップ | `ONCHIP_TRANSITION_SCALE` | quantize（dequantizeの逆） |
| Softmax/Sigmoid/Tanhの前後 | cancellation/restoration | 局所的なdequantize/requantizeのペア |

なお`to_training`/`train`のシミュレーション上は全てfloat32テンソルのまま計算される（§7既述）ため、ここでの「dequantize」は型変換（int8テンソル→float32テンソル）ではなく、**掛けるスケールの値を変えて実効的な数値レンジを元に戻す操作**を指す。

### 4.7 スケーリングが挿入される条件 — 透過性という統一的な視点

§4.1〜4.6で見た個々のスケーリングノードの挿入条件を、**「スケールに対して透過か非透過か」**という一つの視点で整理する。

#### 透過性の定義

ある演算$f$が、正のスケール$s$に対して

```
f(s × x) = s × f(x)
```

を満たすとき「スケールに対して透過的」と呼ぶ。**ハードウェアがその区間にどんな絶対的なスケール値を割り当てても、相対的に正しい結果が得られる**ため、前後でスケールを変える必要がない＝壁（スケーリングノード）が不要になる。

#### スケーリングノードが入る3つのパターン

| パターン | 具体例 | 挿入されるもの | 理由 |
|---|---|---|---|
| ①演算自体が非透過（出力スケール＝入力スケールの"積"、または絶対値依存） | `Mul`(2テンソル), `MatMul` | cancellation/restoration | $mul(s_1 x, s_2 y)=(s_1 s_2)\,mul(x,y)$ — 出力スケールが入力スケールの積になり、単一の共有値として伝播できない |
| 同上 | `Softmax` | cancellation/restoration | $softmax(sx) \ne s\cdot softmax(x)$ — 絶対スケールが出力の"形"自体を変える |
| 同上（別の機構で対処） | `Conv`/`Gemm` | `CompositeScale`→pFSR/iFSR/WSF/DSF（§4.2/4.4） | 数式は`Y=W·X+b`——重み$W$と入力$X$という独立スケールを持つ量の積であり、数学的構造はMatMulと同じ非透過演算。ただし重みは学習済みの静的な量なので、動的なcancellation/restorationペアの代わりに**重み自体を事前補正しておく**という別の解決法を取る |
| ②演算は透過だが、複数入力間でスケールの整合性が必要 | `Add`, `Sum`, `Concat` | 自由スケーリングノード（`ADD_SCALE_INPUT`/`OUTPUT`、§4.3） | $add(sx,sy)=s(x+y)$は**両方が同じ$s$のときのみ**成立。2つの入力が別々の上流から来て異なるスケールを持つ場合、揃える必要がある。既に揃っていれば挿入しない（条件付き、`required=False`） |
| ③表現ドメインの境界 | オンチップ/オフチップ境界 | `AddOnOffChipTransitionScaling`（§4.6） | 量子化済み表現(on-chip)と任意精度表現(off-chip)という**ドメインそのものの切り替え**であり、どんな演算が次に来るかとは無関係に機械的に挿入される |

一方、**透過的な演算（`ReLU`, `hardtanh`/`Clip`, `MaxPool`等）は壁にならず**、スケーリングノードは挿入されない。これらを跨いでedge groupが拡張される（§4.3の`_scan_node`がこれらを素通りさせる）。

#### 実測: BEVFormerのon-chip領域には非透過な活性化が存在しない

実際のモデル（`mythic-1600x900-untrained.onnx`）を調べると、BEVFormerのon-chip（アナログ）側がこの観点でどう振る舞っているかが具体的に確認できる。

```
独立ノードとして残っているSoftmax/Sigmoid/Relu/Clip:
  Softmax: 18件、Sigmoid: 8件、Relu: 47件、Clip: 3件 — 全て off-chip（100%）

Mythicノードに吸収された__activation属性:
  relu: 61件（MythicConv2d 22 + MythicSum 27 + MythicQuantizedMul 12）
  hardtanh: 56件（MythicConv2d 54 + MythicSum 2）
```

独立したSoftmax/Sigmoid（非透過、cancellation/restorationが必要）は**全てoff-chipに退避しており、on-chip側には1件も存在しない**。on-chip側で使われている活性化は`relu`と`hardtanh`のみで、どちらも透過的（`hardtanh`はConv/Sumに明示的な活性化が続かない場合に暗黙的に割り当てられる、ハードウェアレンジへのクリップ。§4.6/[to_training.md](to_training.md)既述）。つまりBEVFormerのon-chip領域には、**活性化関数由来の非透過な壁は実質的に存在しない**——残る非透過な壁はConv/Gemm自身の`CompositeScale`機構（パターン①の3行目）とAdd/Sum/Concatの整合性チェック（パターン②）だけである。これが、§4.3で見た「edge groupが比較的広い範囲に拡張される」現象の実測上の理由である。

---

## 5. `train` — 新規ノードを増やさず、丸め・ノイズ・DSFだけが動く

`train`ステップは`to_training`が確定させたグラフ構造（`MythicConv2d`/`MythicLinear`/`MythicSum`/`MythicQuantizedMul`等）に対して**新規ノードを一切追加しない**。動くのは次の2つだけである。

### 5.1 DSFだけが学習パラメータとして更新される

`MakeDSFsTrainable`（`munc/ops/make_dsfs_trainable.py`）が`MythicSum`/`MythicQuantizedMul`/`MythicMatMul`の乗数initializerに`__trainable`マスクを付与し（§4.5の`__trainable_dsf`とは別の属性、ノード種別も異なる）、TorchNetのoptimizerがこれを学習対象として誤差逆伝播で更新する。WSF・pFSR・iFSR・活性化スケーリングノードの値は`to_training`で確定済みのまま変化しない。

`PinLastOnChipConvDSF`/`PinLastOnChipMul`（`munc/ops/pin_last_on_chip_conv_dsf.py`等）は特定条件（off-chip遷移の直前）のノードから`__trainable_dsf`を除去し、ハードウェア最適値（0.5または1）に固定して学習対象から外す例外処理である。

**実測による裏付け**: 環境上に残っていた学習済みBEVFormerモデル（`bevformer-tiny-1600x900-trained.onnx`、`__type=MythicModel`）と、未学習モデルの同一15ノードを比較したところ（[to_training.md §10.5](to_training.md)）、`__pFSR`/`__iFSR`は学習前後で完全に不変（WSF・pFSR・iFSRが学習対象でないことの直接的な裏付け）だったのに対し、DSFは学習前に全て`max_dsf=3`へクランプされていた14ノードが、学習後は`1.79`〜`4.97`という層ごとに異なる値へ分散しており、複数ノードで`max_dsf=3`を明確に超えていた。これは「学習時のDSF量子化は`max_dsf`ではなく`ds_max_mult`/`ds_max_shift`というハードウェアの表現限界を使う」（§6既述）という理解を実測で裏付けるものである。

### 5.2 round/clip/noiseはグラフに残らず、forward実行時にのみ適用される

重みを整数に丸めるとSTE（Straight-Through Estimator）なしでは勾配が消失する、という量子化訓練の古典的な問題に対し、`munc/_pytorch/layers.py`はカスタム`torch.autograd.Function`で明示的にSTEを実装している。

```python
# munc/_pytorch/layers.py:22-41
class Quantize(torch.autograd.Function):
    """Quantization autograd function."""

    @staticmethod
    def forward(ctx, x, num_fractional_bits=0):
        if num_fractional_bits == 0:
            return x.round()
        else:
            scale = (2 ** num_fractional_bits)
            return (x * scale).round() / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None   # 丸めの勾配を無視し、入力の勾配をそのまま通す

quantize = Quantize.apply
```

forwardでは本物の`round()`を実行するが、backwardでは**roundの数学的な勾配（ほぼ全域で0）を無視し、`grad_output`を無条件に素通しする**。同じパターンが`FloorSTE`（:8-20）、`QuantizeDSF`（:113-129、DSFのmultiplier/shift分解）、`QuantizeFSR`（:132-152、iFSR/pFSRの離散候補へのスナップ）にも適用されている——**「丸め・離散化が入るところは例外なくSTEで勾配を通す」**という一貫した方針である。

一方、`clip`は`torch.clamp`のラッパーに過ぎず、カスタムautogradを使わない（:103-110）。

```python
def clip(x, min_, max_):
    return torch.clamp(x, min_, max_)
```

PyTorch標準の`clamp`は範囲外で本当に勾配0になる。`BoreasWeightModel.forward`のコメントがこの意図を明言する。

```python
# munc/_boreas_ace_model.py:57
# Clip at the beginning to prevent gradients outside of physically possible range
```

つまり**「量子化誤差（round）は学習を妨げないようSTEで無視するが、ハードウェアの表現限界を超えた値（clip）への勾配は意味がないので本当にゼロにする」**という役割分担が意図的に設計されている。

アナログノイズ注入（`munc/_pytorch/noise.py`の`WeightNoise`/`TempShift`/`ADCNonLinearity`/`ADCNoise`）も同様にカスタムautograd.Functionで、乱数は`torch.no_grad()`下で定数として生成し、backwardは入力への勾配をそのまま通すSTEである。

```python
# munc/_pytorch/noise.py:4-55
class WeightNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, additive_noise, mult_sigma):
        weight = weight + normal_like(0, additive_noise, weight) + weight*normal_like(0, mult_sigma, weight)
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
```

`weight_noise()`関数のみ`ste`フラグを持ち、`ste=False`の場合は`.apply()`を使わず`.forward()`を直接呼ぶことで、ノイズの数式そのもの（`weight`依存の乗算ノイズ項を含む）をPyTorchの標準自動微分に辿らせるモードに切り替えられる（Denali側`weight_noise_back_prop`パラメータ、`_denali_ace_separable_model.py:240-242`、Boreas側`back_prop_weight_noise`）。デフォルトはSTE。

これらの適用順序（§4.5のBaseWeightModelを例に）:

```python
# _ace_model.py:148-162（概要）
weight = quantize(weight, hw_config.weight_fractional_bits)  # STEでround
weight = clip(weight, weight_min, weight_max)                # 範囲外は勾配0
```

**この丸め・クリップ・ノイズはONNXグラフに新規ノードとして残らない**。TorchNetレイヤー（`MythicConv2d`等のPyTorch実装）のforward内で毎回計算されるだけであり、`train`完了後にONNXへ書き出す際は、確定した重み（ノイズなしの決定論的な値）とDSFの学習済み数値がそのまま使われる。

---

## 6. `to_acm` — スケール因子が実際の物理表現に変換される（参考）

本ドキュメントの主眼は`to_structural`→`to_training`→`train`だが、「スケール因子が確定しても、まだ実際の丸めは実行されない」（§4.5）という事実の対になる最終段として、[to_acm.md](to_acm.md)の既存解析を参照する。

[to_acm.md §6](to_acm.md)によれば、`HardCodeWeightAndBias`（`munc/ops/hard_code_weight_and_bias.py:5-10`）が「重み・バイアスをinitializerとしてハードコードし、それらを生成していたノードをグラフから削除する」処理を行い、`quantize_weight`（Boreasでは`weight_fractional_bits=0`のため実質整数丸めに退化、Denaliでは8bitの小数部を持つ固定小数点、[to_acm.md §6.3](to_acm.md)）が実際の数値丸めを実行する。iFSR/pFSR自体も連続値ではなく、ハードウェアごとに定義された離散集合から選ばれる（`hw_specs.py`、[to_acm.md §6](to_acm.md)）。

つまり「`to_training`でスケール因子（対応表）が確定する」→「`train`でDSFが学習調整される（丸め自体はforward時のシミュレーションのみ）」→「`to_acm`で初めて重みが実際に最終的な数値へ丸められ、ハードコードされる」という3段階が、量子化誤差の実際の発生タイミングを決めている。

**注意（「重みは8bit量子化」は不正確、Denaliでは実質16bit相当）**: `weight_min`/`weight_max`（`-128`/`127`）はあくまで**レンジ**であり、量子化グリッドの細かさは別パラメータ`weight_fractional_bits`（`hw_specs.py`）で決まる。

| ハードウェア | `weight_fractional_bits` | 量子化グリッド | 実質bit数 |
|---|---|---|---|
| Boreas | 0 | 整数刻み（`round(weight)`） | 真の8bit（256段階） |
| Denali | 8 | `1/256`刻み（`round(weight×256)/256`） | 実質16bit相当（整数部8bit＋小数部8bit） |

本解析の主対象であるBEVFormer（Denali）では、重み・バイアスは`-128〜127`というレンジの中を`1/256`刻みで量子化される。これは`WeightBiasQuantClip`（`munc/ops/weight_bias_quant_clip.py:11-49`、`quantize_weight`→`clip_weight`の順）が実行し、実機artifactでも`weight×256`が誤差0で整数になることが直接確認されている（[to_acm.md §9.2](to_acm.md)、YOLOPX/BEVFormer共通）。**活性化（`input_bits=8`）はこの区別を受けず、両ハードウェアで真の8bitのまま**——この精度差は重み・バイアスに限った話である。

**ただし「実質16bit相当」はデジタル表現の分解能であり、物理的に区別できる実効精度ではない。** 重みの書き込みノイズが支配的な誤差要因になるため、実際の有効bit数はこれより低い。Denaliの重み書き込みノイズは`WEIGHT_ADDITIVE_NOISE_SIGMA=1.92`を`pFSR`で割った値（コード単位、`munc/_denali_ace_separable_model.py:25,231`）であり、実効精度の目安は

```
log2(255 / (1.92 / pFSR)) bit
```

BEVFormer実測の`pFSR=5.0`では`log2(255/0.384) ≈ 9.4`bit——Boreasの真の8bitよりは良いが、デジタル表現が示す16bitには遠く及ばない。この式から**pFSRが大きいほど実効精度が上がる**ことが分かる。これは[to_training.md §5.4.3](to_training.md)で見た`calculate_half_pfsr_ifsr_and_digital_scale`のスコア関数が「同条件ならpFSRが大きい方を優先する」（`half_pFSR_cost=-1`, `munc/_session_tools.py:225`）としていた理由の物理的根拠そのものである。

まとめると、重みの精度は**レンジ（8bit整数相当）→デジタル表現（16bit相当）→実効精度（ノイズ支配で9〜10bit程度）**という3層構造になっており、「重みは8bit/16bitに量子化される」のどちらも単独では不正確である。

---

## 7. 参照ファイルと未解明点

### 抽出ソースの所在

[to_training.md §11](to_training.md#11-参照ファイルと未解明点)の表に準じる。本ドキュメント固有の追加分:

| 分類 | ファイル |
|---|---|
| STE実装（round/floor/DSF/FSR量子化） | `munc/_pytorch/layers.py` |
| アナログノイズのSTE実装 | `munc/_pytorch/noise.py` |
| サインネス補正 | `munc/ops/add_input_shifting.py` |
| off-chip宣言 | `munc/_node_utils.py`（`mark_off_chip`/`is_off_chip`） |
| モデル固有構造変更の例（pythia） | `mythic/model_zoo/pythia/conversion_steps.py` |

### 未解明点

1. `weight_noise_back_prop`/`back_prop_weight_noise`フラグ（STEを使わず本物の勾配を伝播させる設定）が、実際にどのモデル/configで有効化されているかは未調査。本解析で確認した経路は常に既定の`ste=True`。
2. `ADCNonLinearity`/`ADCNoise`/`TempShift`には`WeightNoise`のような`ste`切替フラグが存在しない理由（常にSTE固定である設計意図）は未確認。
3. `to_acm`での重み丸め（`quantize_weight`）とDSFの学習済み値との整合性——`train`完了後にDSFが取りうる任意の実数値を、`to_acm`の`__multiplier`/`__shift`分解（§4.4と同じ`calculate_digital_scale_factors`機構と推定）がどう再量子化するかの詳細は、[to_acm.md](to_acm.md)側の既存未解明点を参照。本ドキュメントでは追跡していない。
4. `to_structural`で`__off_chip`以外にモデル共通で付与される属性が本当に無いか（pythia以外の4モデルについて同様の精査は[to_structural.md](to_structural.md)側で完了済みだが、本ドキュメントは再確認していない）。
5. §5.1で実測した「DSFが学習で`max_dsf`を超えて動く」ことは学習済みBEVFormerとの比較で裏付けられたが、**それによって最終的な推論精度（AP等）がどの程度回復するか**は、学習済みartifactに評価メトリクスが付随していなかったため未検証（[to_training.md §10.5](to_training.md)の注意点を参照）。
