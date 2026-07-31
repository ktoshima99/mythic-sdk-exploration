# 計画: YOLOPX PPA(面積-電力トレードオフ探索)パラメータ探索

状態: **探索完了(SDK v26.05.2)。Phase 0-5 全完了。結論は「48 ACE が最適(72 ACE は面積・電力とも劣位)」。** m2048=9.39ms/0.743W(可行、マージン71.5%)/ m2072=7.69ms/0.829W(可行だが劣位)、24=重み容量超過で不可/32=定義なし。**BEVFormer(72が必須)とは正反対の結論**になる(§5.1)。[PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) で確立した方法論を YOLOPX に適用したもの。

### 進捗サマリ

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | 資料精読(Compiler Optimization Report - YOLOPX v3.0 / Model Summary Report v3.0)→照合基準値の確定 | ✅ 完了(§2) |
| 1 | 環境確認(既存コンテナ `mythic_ppa_explore_2605_2b` 再利用、YOLOPX artifact の配置) | ✅ 完了 |
| 2 | Accuracy simulator の実測値確認 | ⛔ **実施不可**(BDD100K データセット未配置。ドキュメント値を参考値とする。§2.1) |
| 3 | ACE数を変えてコンパイル(m2024/m2032/m2048/m2072) | ✅ 完了。48/72 成功、24=容量超過で真の INFEASIBLE、32=定義なし(§3) |
| 4 | 各 artifact の PPA 推定 | ✅ 完了。m2048 はドキュメント値を 0.3% 以内で再現(§4.1)、m2072 実測(§4.2) |
| 5 | トレードオフ表作成(物理面積 5.278mm²/ACE) | ✅ 完了。§5 —— **Pareto フロンティアは 48 ACE の単一点に退化** |

### 結論サマリ

| `num_aces` | 面積 | Latency | 33ms制約 | Power@30fps | 採否 |
|---|---|---|---|---|---|
| 24 | 127 mm² | — | — | — | 不可(重み容量超過) |
| 32 | 169 mm² | — | — | — | 不可(定義なし) |
| **48** | **253 mm²** | **9.39 ms** | ✅ 71.5%マージン | **0.743 W** | ✅ **推奨** |
| 72 | 380 mm² | 7.69 ms | ✅ 76.7%マージン | 0.829 W | ❌ 面積+50%/電力+11.6% |

**48 ACE が面積・電力の両目的で 72 ACE を Pareto 支配する。** 72 の latency 優位(-18.1%)は 33ms 制約充足には不要(両者とも大幅にクリア)なので、重み付けに関係なく 48 が一意に最適。

**BEVFormer との相違(最大の含意)**: BEVFormer-Tiny は 72 ACE が唯一の可行 SKU(48 では 43.47ms で制約 31% 超過)。**両モデルを同一 SKU に載せるなら BEVFormer が律速して 72 ACE が必要**、YOLOPX 専用 SKU なら 48 ACE で面積 33% 削減可能(§5.1)。

**方法論(BEVFormer 探索からの差分):**

- **入力は単一カメラ**。YOLOPX compiler config の `--input-dims` は `736 1280 3`(コンテナ内で実物確認)。BEVFormer の `6 928 1600 3` と異なり6カメラ合成ではないため、`Combined Analog + Digital NPU Latency` はそのまま1フレームの latency。
- **既定 arch が m2048**。YOLOPX 同梱の compiler config は `yolopx_736x1280x3_m2048_{default,high}_optimization.yaml` の3種で、**m2072 用の config は同梱されていない**。m2024/m2032/m2072 用の YAML は m2048 の high_optimization をコピーして `--amp-arch` のみ書き換えて作成した(§3.1)。
- **`--optimization-effort` は 500**(BEVFormer は 300)。YOLOPX の high_optimization config 既定値をそのまま使う。
- 環境: compiler docker tag `v26.05.2`、コンテナ `mythic_ppa_explore_2605_2b`(BEVFormer 探索で立てたものを再利用)、ワークスペース `/tmp/ppa_workspace_2605_2/`。ツールの使い方は [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md)。

---

## 0. 目的と制約

BEVFormer 探索(§0)と同一の枠組みを使う: `num_aces` を**ダイ面積・製造コストを決める設計変数**として扱い(SKU 選定)、area・power の2目的同時最小化を latency 制約下で行う。

**YOLOPX 固有の状況**: BEVFormer-Tiny では既定の 72 ACE が 33ms 制約に対しマージン 4.3% しかなく、探索は「これ以上削れるか」という下方向のみだった。**YOLOPX はドキュメント値で 9.41ms / 106fps** であり、33ms 制約に対して**マージン 71.5%** と極めて大きい。したがって YOLOPX の探索は:

> **48 ACE(253mm²)から下方向にどこまで削れるか(=より小さい・安い SKU が成立するか)** が主問題になる。

これは BEVFormer とは逆向きの問い(BEVFormer は 72 が下限だったが、YOLOPX は 48 が上限側の余裕点)であり、**両モデルを同一チップで走らせる場合の SKU 選定では BEVFormer が律速する**という含意を持つ(§5)。

| 項目 | 内容 |
|---|---|
| 制約1(area) | `num_aces` に対応する die area ≤ A_max(具体値は未確定。候補空間のカットオフ) |
| 制約2(精度) | ドキュメント値で確認済み(§2.1)。`num_aces` は精度に影響しない(BEVFormer PLAN §3)ため本探索では再検証不要 |
| 制約3(レイテンシ) | 1フレームあたり ≤ 33ms(30fps)。ドキュメント値 9.41ms で大幅にクリア |
| 目的関数 | area・power の2目的同時最小化 |

---

## 1. 前提: SDKが提供する機能

BEVFormer 探索(PLAN §1)と同一。`num_aces` は `mythic-ppa-estimators` 側では変更できず、**`mythic-compiler` の `--compiler-config` YAML 内 `COMPILER_OPTIONS` の `--amp-arch` で指定**する。露出パラメータ一覧・計測ツールの限界(Leakage/Clock Tree/PCIe/D2D 電力未算入 → 相対比較専用)も同じ。

YOLOPX の既定 compiler config(コンテナ内実物):

```yaml
# yolopx_736x1280x3_m2048_high_optimization.yaml
model: "yolopx"
COMPILER_OPTIONS: [
  "--amp-arch m2048",           # Gen 2 with 48 Analog Compute Engines
  "--acm",
  "--input-dims 736 1280 3",    # 単一カメラ
  "--stats terse",
  "--optimization-effort 500",  # Run optimization longer
]
VNN:
  CONTAINER_SCRIPT: yolopx_postprocessing.py
```

**アナログ/デジタル振り分け(compiler-ready artifact から実測)**:

| グラフ | 内容 |
|---|---|
| `..._on_chip_1_bcm.onnx`(アナログ) | 387ノード。`BCMConv2d` 206 / `Slice` 94 / `BCMSum` 41 / `Concat` 27 / `Resize` 9 / `MaxPool` 8 |
| `..._off_chip_0.onnx`(デジタル・前処理) | 2ノード(`Mul`, `Clip`)、入力 `[N,3,736,1280]` |
| `..._off_chip_2.onnx`(デジタル・検出ヘッド末端) | 14ノード(`Conv` 9 / `Concat` 3 / `Cast` 2)。3スケール × {cls, reg, obj} の 1x1 Conv。入力は 192ch の [92,160]/[46,80]/[23,40] |

デジタル側は 3検出ヘッドの最終 Conv のみ(Model Summary Report v3.0 §9 の記述と一致)で、ドキュメント値では **Digital 1.84ms / 1 Digital NPU コア / MAC 89M(全体の 0.03%)**。アナログ側 285.0G MAC が支配的。

---

## 2. 精度・レイテンシの現状(ドキュメント値)

出典: `doc/reports/Model Summary Report.pdf`(v3.0)、`doc/reports/Compiler Optimization Report - YOLOPX.pdf`(v3.0)。いずれも Mythic 提供レポートの引用であり本探索で再計測したものではない。データセットは BDD100K。

**精度**(FP32 → ANA8 Retrained、ドキュメント値):

| 指標 | FP32 | ANA8 Quantized | ANA8 Retrained | ANA8 Retrained 100PPM |
|---|---|---|---|---|
| Driving Area Seg. Accuracy | 0.979 | 0.829 | 0.978 | 0.977 |
| Driving Area Seg. IOU | 0.887 | 0.000 | 0.880 | 0.875 |
| Driving Area Seg. mIOU | 0.931 | 0.415 | 0.927 | 0.924 |
| Lane Line Seg. Accuracy | 0.834 | 0.000 | 0.831 | 0.801 |
| Lane Line Seg. IOU | 0.275 | 0.006 | 0.268 | 0.249 |
| Lane Line Seg. mIOU | 0.631 | 0.455 | 0.627 | 0.617 |
| Detection Precision | 0.047 | 0.001 | 0.048 | 0.043 |
| Detection Recall | 0.890 | 0.016 | 0.864 | 0.851 |
| Detection mAP50 | 0.681 | 0.000 | 0.630 | 0.615 |
| Detection mAP50-95 | 0.334 | 0.000 | 0.290 | 0.272 |

- **retraining の効果が決定的**: ANA8 Quantized(再学習なし)はセグメンテーション IOU / 検出 mAP がほぼ 0 に崩壊するが、ANA8 Retrained では FP32 とほぼ同等に回復する。BEVFormer-Tiny(mAP 0.2293→0.2290)と同様の傾向だが、崩壊幅は YOLOPX の方がはるかに大きい。
- 検出系は retrained でも劣化が残る(mAP50 0.681→0.630 = -7.5%、mAP50-95 0.334→0.290 = -13.2%)。セグメンテーション系はほぼ無劣化(mIOU -0.4%)。
- 100PPM でさらに小幅劣化(mAP50 0.630→0.615)。

**レイテンシ・電力・面積**(単一カメラ 736x1280、m2048 / high_optimization、ドキュメント値):

| 項目 | 値 |
|---|---|
| Mythic NPU構成 | m2048(48 ACEs, 12 ACEタイル, 4MB/タイル, **253 mm²**ダイ面積), 1 Digital NPUコア |
| Analog NPU Processing Time | 7.58 ms |
| Digital Estimated Frame Processing | 1.84 ms |
| **Combined Latency** | **9.41 ms**(33ms制約に対しマージン23.59ms=**71.5%**) |
| Combined Frame Rate | 106.23 fps |
| Power @30fps(target) | **0.679 W**(analog 0.677 / digital 0.002) |
| Power @106fps(max) | 2.238 W(analog 2.230 / digital 0.008) |
| ACE利用率 | 72.55% |
| Weight utilization | 83.42%(49,858,224 / 59,768,832、257 weight blocks) |
| SRAM utilization | 63.77%(32,096,422 / 50,331,648 bytes) |
| Total Executed ACE MACs | 285,021,216,768 |
| L0 IR | Tiles 13 / Buffers 384 / Launchers 629 / ACE Launchers 257 / ACE Weights 49,858,224 |

**注意(ドキュメントの既知の不整合)**: Compiler Optimization Report - YOLOPX の power セクションは `Number of ACEs: 24` と表示しているが、これは旧版 estimator の ACE 数誤表示(BEVFormer 探索でも遭遇した既知の症状)。同レポートの performance セクション・ACE utilization レポートはいずれも **48 ACEs / 12 タイル**と明記しており、実際の構成は m2048。v26.05.2 の estimator は structured output 化され `Number of ACEs` が正表示される([HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §2)ので、本探索の実測値で確認する。

### 2.1 Accuracy simulator の自前実測は実施不可(BDD100K 未配置)

BEVFormer 探索では `convert_model.py steps=eval_trained` を自前実行し、`eval_trained` が run ごとに約 0.7pt 変動することを実測した(BEVFormer PLAN §2.1)。**YOLOPX で同じ実測を行うには BDD100K データセットが必要だが、この環境には存在しない**:

| 探索場所 | 結果 |
|---|---|
| YOLOPX config 既定パス `/data/shared/global/datasets/bdd100k/`(`configs/yolopx/model_setup/yolopx.yaml`) | 存在しない |
| ホスト全体(`find / -iname "*bdd100k*"`) | ヒットなし |
| S3(`s3-srdm` / `mythic-sdk` / `sharebucketlsi` / `retrainig-output` / `s3-srdm-iit-intern-2026` 全バケット) | BDD100K なし(`datasets_sumire/` には COCO2017・CIFAR-100・OpenImages v7・vlmeval のみ) |
| SDK コンテナ内 `mythic-model-zoo/datasets/` | 空 |
| `/mnt/nvme_scratch`(nuScenes/CARLA 用スクラッチ) | nuScenes 系のみ |

**したがって §2 のドキュメント値を精度の参考値として採用する**(ユーザー判断)。`num_aces` は精度シミュレーション結果に影響しない(BEVFormer PLAN §3)ため、この欠落は本探索(area-power トレードオフ)の判断には影響しない——精度は `num_aces` に対して定数である。

BDD100K が将来配置された場合の実行手順は [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §3 と同形式:

```bash
cd $MYTHIC_SDK_ROOT/mythic-model-zoo
source scripts/yolopx/yolopx-720p-m2000.env
DATASET=<bdd100k_dir>/ python3 scripts/common/convert_model.py steps=eval_trained \
    trained_model=../models/training/yolopx/yolopx_trained.onnx
```

evaluator は `mythic.model_zoo.yolopx.evaluate_metrics.evaluate_metrics`(`configs/yolopx/base_config.yaml`)。

---

## 3. コンパイル結果(ACE数を振る)

### 3.1 準備した compiler config

YOLOPX 同梱 config は m2048 のみ(§1)なので、high_optimization をベースに `--amp-arch` だけ書き換えた YAML をコンテナ内に作成した:

```bash
CFG=$MYTHIC_SDK_ROOT/mythic-model-zoo/configs/yolopx/compiler
for A in m2024 m2032 m2072; do
  sed "s/--amp-arch m2048/--amp-arch $A/" \
    $CFG/yolopx_736x1280x3_m2048_high_optimization.yaml \
    > $CFG/yolopx_736x1280x3_${A}_high_optimization.yaml
done
```

入力 artifact は既存の `models/training/yolopx/yolopx_trained.tar.gz`(再学習不要)。**Docker-out-of-Docker のパス一致要件(HOWTO §0.2.1)のため、artifact をホスト/コンテナで同一パスに見える `/tmp/ppa_workspace_2605_2/yolopx_in/` に複製して使用した**(既存コンテナの `models/training` バインドマウント元ディレクトリが空に再作成されていたため)。

### 3.2 コンパイル結果

| `num_aces` | 所要時間 | 結果 |
|---|---|---|
| 24 (`m2024`) | 打ち切り(約2時間で解ゼロ) | **利用不可(容量不足=真の INFEASIBLE)**。§3.3 |
| 32 (`m2032`) | 6.7秒 | **利用不可**(ターゲット定義なし)。§3.4 |
| 48 (`m2048`) | 約112分 | **成功**(exit 0)。ドキュメント値の L0 IR stats を完全再現(§3.5) |
| 72 (`m2072`) | 1回目: 約112分で異常終了 / **2回目: 約109分で成功** | 1回目は `malloc(): invalid size`(ヒープ破壊)でクラッシュ。**同一設定の再試行で成功**(exit 0)。§3.6 |

### 3.3 m2024 が利用不可な根本原因 —— アナログ重み容量の不足(BEVFormer とは別原因)

**BEVFormer の m2024 失敗は CP-SAT のタイムアウト(`status: UNKNOWN`、理論上は入る可能性が残る)だったが、YOLOPX の m2024 は容量そのものが足りない**。

compiler-ready artifact のアナロググラフ(`compiler_ready_artifact_on_chip_1_bcm.onnx`)の `BCMConv2d` 206個の重みを直接集計した:

| 量 | 値 |
|---|---|
| **アナログ重み素数(生・パディングなし・複製なし)** | **32,447,232** |
| バンクパディング概算(行を256、列を8の倍数に切り上げ) | 36,233,216 |

これを各 arch の重み容量(1タイル=4 ACE、容量 4,980,736/タイル。BEVFormer 探索の `weight_utilization.txt` で検証済みの固定値)と比較する:

| 構成 | タイル数 | 重み容量 | 生重み / 容量 |
|---|---|---|---|
| **m2024** | 6 | **29,884,416** | **1.086(=108.6%、超過)** |
| m2048 | 12 | 59,768,832 | 0.543 |
| m2072 | 18 | 89,653,248 | 0.362 |

**生の重み(複製を一切行わない理論下限)だけで m2024 の全容量を 8.6% 超過する。** コンパイラは並列化のため重みを複製する方向にしか動けないので、これは削減の余地がない下限であり、**m2024 は YOLOPX にとって物理的に配置不可能**。

実機挙動もこれと整合した: m2024 のコンパイルは `CpSatClustering` に到達したものの、**解を1つも見つけないまま**(`best:` 行がログに 0 件)停滞した。同時に走らせた m2048/m2072 は数秒で初期解を得て `best:` を更新し続けたため、これは「難しいが解ける問題」ではなく「解が存在しない問題」の挙動である。

ドキュメント値の m2048 実配置 49,858,224 重み(容量の 83.42%)は複製込みの値で、生 32.4M に対し約 1.54 倍の複製が行われている。

**この集計手法を BEVFormer-Tiny で検証(クロスチェック)**: 同じスクリプトを BEVFormer の `compiler_ready_artifact_on_chip_1_bcm.onnx`(`BCMConv2d` 74個)に適用すると **生 24,569,024 / パディング概算 26,394,624** となり、m2024 容量 29,884,416 に対し **82.2% / 88.3%** の充填率になる。これは BEVFormer PLAN §4 / HOWTO §1 で weight blocks 数比から概算していた「約 25M、約 84% 充填」と一致する。**したがって本手法は妥当であり、両モデルの m2024 失敗は原因が異なると確定する**:

| モデル | 生重み | m2024 容量比 | m2024 の失敗原因 |
|---|---|---|---|
| BEVFormer-Tiny | 24.57M(パディング 26.39M) | 82~88%(**収まる**) | 高充填領域で CP-SAT がタイムアウト(`status: UNKNOWN`、122分) |
| **YOLOPX** | **32.45M**(パディング 36.23M) | **109~121%(収まらない)** | **容量超過。真の INFEASIBLE** |

### 3.4 m2032 が利用不可な根本原因 —— ターゲット定義なし(モデル非依存)

m2032 のコンパイルは 6.7 秒で即失敗:

```
target.cpp:38 FATL| ABORT: Unable to build AMP architecture "m2032" without compatibility enabled!
```

**BEVFormer で確認したものと完全に同一のエラー**であり、原因も同じ——`dnn_compiler` バイナリに `m2032` のターゲット定義自体が存在しない(定義済みは m2024/m2048/m2048_ace16mb/m2072 のみ。strings/逆アセンブル調査で確定、[HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §1)。`--boreas-a-compatible` による抜け道も BEVFormer 側で塞がれている(エラーが `without` → `with` に変わるだけ)。**モデルに依存しない SDK 側の制約**なので、YOLOPX でも同じく利用不可であり、再度の抜け道テストは不要。

### 3.5 m2048 サニティチェック(環境の正当性確認)

BEVFormer 探索の Phase 2 と同じ狙いで、ドキュメント値のある m2048 構成を再コンパイルし L0 IR stats を照合した。**全項目が完全一致**し、環境・入力 artifact・config の正当性を確定した:

| L0 IR stats 項目 | 実測 | ドキュメント値(Compiler Opt Report YOLOPX v3.0 §5.1) |
|---|---|---|
| Tiles | 13 | 13 |
| Buffers | 384 | 384 |
| Launchers | 629 | 629 |
| ACE Launchers | 257 | 257 |
| **ACE Weights** | **49,858,224** | 49,858,224 |
| ACE calculations | 1,649,376 | 1,649,376 |
| Max ACE calculations | 151,248 on tile 2_3 | 151,248 on tile 2_3 |

マルチスレッドソルバーの非決定性(§6-2)にもかかわらず配置結果がビット一致した点は、YOLOPX の L0 最適化がこの構成では安定解に収束していることを示す。

### 3.6 m2072 の1回目クラッシュ(コンパイラのヒープ破壊バグ)

m2072 の1回目のコンパイルは、2つ目の `CpSatClustering`(SRAM 割当後のスケジューリング)で解を更新し続けていた途中(93.5秒地点、`best:1822`)に **`dnn_compiler` プロセスが異常終了**した:

```
#7      93.50s best:1822  next:[1750,1821] core fixed_bools:391/9114
malloc(): invalid size (unsorted)
/bin/bash: line 1:    12 Aborted (core dumped) /mythic/dnn_compiler ... --amp-arch m2072 ...
```

`malloc(): invalid size (unsorted)` は glibc がヒープのメタデータ破壊を検出した際のメッセージであり、**リソース不足(OOM)や制約違反ではなく `dnn_compiler` 側のメモリ破壊バグ**。ホストメモリは 248GB に対しピーク使用量が数GB程度で、OOM ではない。直前のログに `The solution hint is complete, but it is infeasible! we will try to repair it.` があるが、これは CP-SAT が hint 修復を試みる通常の情報メッセージで、この時点では解の更新が正常に進んでいた(`best:` を7回更新)。

**マルチスレッドソルバーの非決定性(§6-2)により再実行で結果が変わりうるため、同一設定で再試行した。** BEVFormer 探索では発生しなかった事象であり、YOLOPX 固有(または m2072×YOLOPX の組み合わせ固有)の可能性がある。

**2回目は同一設定・同一入力で成功した(約109分、exit 0)。** 1回目がクラッシュした同じスケジューリング部分問題を通過し(1回目は93秒地点で落ちたが、2回目は同部分問題で216秒以上探索を継続して収束)、artifact 生成まで完走した。**したがって m2072 は YOLOPX でも利用可能であり、1回目のクラッシュは再現性のない一過性のコンパイラバグ**と結論する。実務上の含意: **`malloc(): invalid size` 系のクラッシュに遭遇したら、設定を変えずまず再実行する**(§6-6)。

### 3.7 m2072 のマッピング結果(m2048 との比較)

| L0 IR stats 項目 | m2048 | m2072 | 差 |
|---|---|---|---|
| Tiles | 13 | **19** | +6 |
| Buffers | 384 | 520 | +136 |
| Launchers | 629 | 903 | +274 |
| ACE Launchers | 257 | 311 | +54 |
| **ACE Weights** | 49,858,224 | **58,857,104** | **+18.0%** |
| ACE calculations | 1,649,376 | 1,649,376 | **一致** |
| Max ACE calculations(最繁忙タイル) | 151,248 (tile 2_3) | **104,374** (tile 2_3) | **-31.0%** |

**読み取れること**:
- **ACE calculations は 1,649,376 で完全一致** —— モデルの演算量そのものは `num_aces` に依存しない(当然だが、実測での裏付け)。
- **ACE Weights は +18.0% 増加**(49.86M→58.86M)。ACE が増えた分、コンパイラが重みを複製して並列度を上げている(BEVFormer でも同傾向: 48で31.25M→72で35.76M)。生重み 32.45M(§3.3)に対する複製倍率は m2048 で 1.54倍、m2072 で 1.81倍。
- **最繁忙タイルの負荷が 31% 低下**(151,248→104,374)。これがクリティカルパス短縮=latency 改善の直接の源であり、§4.2 の実測 latency 改善と対応する。

---

## 4. PPA推定結果

実行コマンド(単一カメラなので `--power-inference-rate 30` で1回のみ):

```bash
mythic-ppa-estimators --estimate-performance --estimate-power --power-inference-rate 30 \
    /tmp/ppa_workspace_2605_2/out/yolopx_m2048_high_2605_2.tar.gz
```

### 4.1 m2048(48 ACEs)実測 —— ドキュメント値と一致

**funcsim 所要時間 約49分**(07:22→08:11)。BEVFormer の約4時間に対して大幅に短いのは、入力が単一カメラ(736x1280)であり ACE 演算数が 1.65M(BEVFormer は 8.42M)と少ないため。

| 項目 | m2048 実測 | ドキュメント値(v3.0) | 差 |
|---|---|---|---|
| Analog NPU Processing Time | **7.56 ms** | 7.58 ms | -0.3% |
| Digital Estimated Frame Processing | **1.84 ms** | 1.84 ms | 一致 |
| **Combined Analog + Digital NPU Latency** | **9.39 ms** | 9.41 ms | -0.2% |
| Combined Frame Rate | **106.47 fps** | 106.23 fps | +0.2% |
| ACE Utilization | **72.75%** | 72.55% | +0.2pt |
| Critical Path ACE Latency | 7.56 ms | 7.58 ms | -0.3% |
| Min理論(48 ACE均等並列) | 5.50 ms | 5.50 ms | 一致 |
| Max理論(1 ACE、並列化なし) | 263.90 ms | 263.90 ms | 一致 |
| Total Executed ACE Operations | 1,649,376 | 1,649,376 | 一致 |
| Total Executed ACE MACs | 285,021,216,768 | 285,021,216,768 | 一致 |
| Digital NPU MAC / コア数 / 周波数 | 89,000,000 / 1 / 1,000 MHz | 同 | 一致 |
| Digital NPU MAC Utilization | 75.79% | 75.79% | 一致 |
| **Number of ACEs** | **48(正表示)** | 24(旧版の誤表示、§2注記) | v26.05.2 で修正 |
| **Total Combined Power @30fps** | **0.743 W**(analog 0.740 / digital 0.002) | 0.679 W(analog 0.677 / digital 0.002) | +9.4% |
| └ Functional Unit / Interconnect | 0.591 W / 0.149 W | 0.528 W / 0.149 W | FU のみ +11.9% |

**差の解釈**: latency・ACE 演算数・MAC 数・理論値はすべて 0.3% 以内(ソルバー非決定性による通常の変動幅、§6-2)で一致し、**環境の正当性が確定した**。power は Functional Unit Power のみ +11.9%(0.528→0.591 W)ずれており、Interconnect Power は完全一致。これは v26.05 レポート作成時点から `power_estimator.py` の FU 電力係数が更新された可能性を示すが、**本探索は同一ツール・同一条件での相対比較専用(§6-1)なので、SKU 間比較には影響しない**。

**その他の実測挙動**:
- **fpsクランプは発生しない**: 達成可能 106.47fps ≫ 要求 30fps のため、BEVFormer m2048 で見られた自動クランプ(`Power inference rate of '30' exceeds estimated`)は起きない。**したがって YOLOPX の 48/72 の power は同一 30fps 基準で直接比較できる**(BEVFormer では 48 が 23fps にクランプされ基準が揃わなかった)。
- **die area 行は出力されない**(v26.05.2 の仕様、HOWTO §2)。ドキュメント値 v3.0 には `Estimated Die Area to Achieve Estimated Processing Time: 253.09 mm^2` が載っているが、これは旧版 estimator の出力。面積は §5 の物理傾きで算出する。
- **【判定】m2048 = 9.39ms ≤ 33ms → 制約OK(マージン 23.61ms = 71.5%)。**

### 4.2 m2072(72 ACEs)実測 —— latency は改善するが power は悪化

**funcsim 所要時間 約58分**(09:12→10:07)。

| 項目 | m2048(48 ACEs) | m2072(72 ACEs) | 差 |
|---|---|---|---|
| Analog NPU Processing Time | 7.56 ms | **5.86 ms** | **-22.5%** |
| Digital Estimated Frame Processing | 1.84 ms | 1.84 ms | **一致** |
| **Combined Latency** | **9.39 ms** | **7.69 ms** | **-18.1%** |
| Combined Frame Rate | 106.47 fps | 130.01 fps | +22.1% |
| Critical Path ACE Latency | 7.56 ms | 5.86 ms | -22.5% |
| **ACE Utilization** | **72.75%** | **62.59%** | **-10.2pt(悪化)** |
| Min理論(均等並列) | 5.50 ms(48 ACE) | 3.67 ms(72 ACE) | -33.3% |
| Max理論(1 ACE) | 263.90 ms | 263.90 ms | 一致 |
| Total Executed ACE Operations | 1,649,376 | 1,649,376 | 一致 |
| Total Executed ACE MACs | 285,021,216,768 | 285,021,216,768 | 一致 |
| Max SRAM Read/Write Time | 6.83 ms(12タイル) | 4.99 ms(18タイル) | -26.9% |
| Total SRAM Bytes Read | 4,624,739,160 | **4,871,279,544** | **+5.3%** |
| Total SRAM Bytes Written | 2,401,121,232 | **2,577,301,008** | **+7.3%** |
| Digital 側(MAC/コア/周波数/利用率) | 89M / 1 / 1,000MHz / 75.79% | 同一 | **完全一致** |
| Number of ACEs(正表示) | 48 | 72 | — |
| **Total Combined Power @30fps** | **0.743 W** | **0.829 W** | **+11.6%(悪化)** |
| └ Analog(FU / Interconnect) | 0.740(0.591 / 0.149) | 0.827(0.664 / 0.163) | +11.8%(+12.4% / +9.4%) |
| └ Digital NPU | 0.002 W | 0.002 W | 一致 |

**【最重要】72 ACE は latency を 18.1% 改善するが、power を 11.6% 悪化させる。** 両者とも 30fps 基準で fps クランプなしの比較なので、この power 差は基準の違いによる見かけのものではなく実質的な差である(§4.1 の注記)。

**power が増える機構**(§0 で「`num_aces` を増やせば power が下がる場合がある」と想定していたのと逆になった理由):
- 30fps 固定の推定では、必要な演算量(ACE MACs 285.0G、完全一致)は同じであり、ACE を増やしても「1フレームあたりの仕事量」は減らない。
- 一方で ACE を増やすと重み複製が 49.86M→58.86M(+18.0%、§3.7)に増え、**複製された重みを配るための SRAM トラフィックが増加する**(読み +5.3% / 書き +7.3%)。Interconnect Power が +9.4% 増えているのはこれに対応する。
- Functional Unit Power も +12.4% 増えており、より多くの ACE アレイに同じ演算を分散させる際のオーバーヘッド(複製した重みブロックそれぞれの駆動)が効いている。
- **ACE 利用率は 72.75%→62.59% に悪化**している。これは「72 ACE を使っても均等並列の理論値(3.67ms)には遠く、余った並列度を活かしきれていない」ことを示す。YOLOPX は BEVFormer と違い 48 ACE で既に十分な並列度があり、72 にしても遊ぶ ACE が増えるだけになる。

**【判定】m2072 = 7.69ms ≤ 33ms → 制約OK(マージン 76.7%)。ただし m2048 に対し面積 +50% / power +11.6% で、latency 改善は制約充足に不要。**

---

## 5. トレードオフ表(確定版)

面積は物理傾き 5.278 mm²/ACE(=380/72、BEVFormer PLAN §4.1)で算出。全て単一カメラ 736x1280 の直読値、power は 30fps 基準(両者クランプなし)。

| `num_aces` | 物理ダイ面積 | Combined Latency | 33ms制約 | ACE Util | Total Power @30fps | 採否 |
|---|---|---|---|---|---|---|
| 24 (m2024) | 127 mm² | — | — | — | — | **コンパイル不可**(重み容量 109~121% 超過=真の INFEASIBLE、§3.3) |
| 32 (m2032) | 169 mm² | — | — | — | — | **利用不可**(ターゲット定義なし、§3.4) |
| **48 (m2048)** | **253 mm²** | **9.39 ms** | **OK(マージン 71.5%)** | **72.75%** | **0.743 W** | ✅ **推奨(Pareto最適・唯一)** |
| 72 (m2072) | 380 mm² | 7.69 ms | OK(マージン 76.7%) | 62.59% | 0.829 W | ❌ **劣位**(面積 +50% / power +11.6%) |

### 5.1 結論: 48 ACEs が 72 ACEs を Pareto 支配する

**YOLOPX では area-power のトレードオフが成立しない —— 48 ACE が両目的関数(面積・電力)で 72 ACE より優れており、かつ latency 制約も満たす。**

| 目的変数 | 48 ACEs | 72 ACEs | どちらが良いか |
|---|---|---|---|
| 面積(小さいほど良い) | **253 mm²** | 380 mm² | **48**(-33%) |
| 電力(小さいほど良い) | **0.743 W** | 0.829 W | **48**(-10.4%) |
| latency(制約 ≤33ms) | 9.39 ms ✅ | 7.69 ms ✅ | 両者クリア(制約充足なので優劣なし) |

Pareto フロンティアは**単一点(48 ACEs)に退化する**。72 ACE が唯一勝っている latency は既に 33ms 制約を 71.5% のマージンでクリアしている領域での改善であり、**制約充足に何ら寄与しない**(9.39ms も 7.69ms も「30fps 要件を満たす」という点では同値)。したがってステークホルダーの重み付け(面積優先か電力優先か)に関係なく、**48 ACEs が一意に最適**となる。

**BEVFormer 探索との対比 —— 結論が正反対になる**:

| | BEVFormer-Tiny | YOLOPX |
|---|---|---|
| 入力 | 6カメラ 6x928x1600x3 | 単一カメラ 736x1280x3 |
| ACE 演算数 | 8,421,600 | 1,649,376(**約1/5**) |
| ACE MACs | 955.0 G | 285.0 G(約1/3.4) |
| 生アナログ重み | 24.57M | 32.45M(**より大きい**) |
| 48 ACEs | **43.47 ms → 不可行**(制約 31% 超過) | **9.39 ms → 可行**(マージン 71.5%) |
| 72 ACEs | 31.58 ms → 可行(マージン 4.3%) | 7.69 ms → 可行だが劣位 |
| 24 ACEs | CP-SAT タイムアウト(容量は 82~88% で収まる) | **容量超過で真の INFEASIBLE**(109~121%) |
| **結論** | **72 が必須**(唯一の可行 SKU) | **48 が最適**(72 は面積・電力とも劣位) |

**両モデルを同一チップで動かす場合の含意**: YOLOPX 単体なら 48 ACE(253mm²)で足りるが、BEVFormer-Tiny は 72 ACE(380mm²)を要求する。**したがって両モデルを共通 SKU に載せる前提では BEVFormer が律速し、72 ACE を選ぶことになる**——その場合 YOLOPX 側は 72 ACE 上で 7.69ms / 0.829W で動作する(48 ACE 専用機に比べ電力面で 11.6% 不利だが、動作はする)。逆に YOLOPX 専用 SKU を切るなら 48 ACE で面積を 33% 削減できる。この判断は A_max(§6-4)と、どのモデルを同一シリコンに載せるかという製品構成の決定に依存する。

### 5.2 latency と power の関係(同一モデル内の観察)

`num_aces` を増やしたときの各指標の動き:

| 指標 | 48→72 の変化 | 機構 |
|---|---|---|
| ACE 演算数 / MAC 数 | ±0% | モデル固定なので当然 |
| Digital 側 latency / power | ±0% | `num_aces` はデジタル側に影響しない(BEVFormer でも同結果) |
| Analog latency | -22.5% | 最繁忙タイル負荷 -31.0%(§3.7)によるクリティカルパス短縮 |
| ACE 重み | +18.0% | 並列化のための重み複製 |
| SRAM トラフィック | 読 +5.3% / 書 +7.3% | 複製重みの配布 |
| Analog power | +11.8% | FU +12.4%(ACE アレイ増)/ Interconnect +9.4%(SRAM トラフィック増) |
| ACE 利用率 | -10.2pt | 並列度が余り、遊ぶ ACE が増える |

**30fps 固定の推定では「ACE を増やす = 同じ仕事をより多くのハードで分散する = 単位時間あたりの電力は増える」** という関係になる。§0 で想定していた「ACE を増やすと無理な高稼働率を避けられて power が下がる」というシナリオは、**latency 制約が既に大きなマージンでクリアされている YOLOPX では成立しない**(クロック下げ等の補償手段をツールが自動適用しないため)。この想定が成立しうるのは BEVFormer のように制約ギリギリの領域だが、BEVFormer 側では 48 が不可行だったため検証できていない。

---

## 6. リスク・未検証事項

1. PPA推定値は相対比較専用(Leakage/Clock Tree/PCIe/D2D 電力未算入)。BEVFormer PLAN §6-1 と同じ。
2. マルチスレッドソルバーの非決定性により、同一設定でもコンパイル結果が数%変動する(YOLOPX レポート §2 にも明記)。
3. **BDD100K データセット未配置のため精度の自前実測ができない**(§2.1)。ドキュメント値を参考値として採用。`num_aces` は精度に影響しないため本探索の判断には影響しないが、`eval_trained` の run 間変動(BEVFormer では約0.7pt)が YOLOPX でどの程度かは未確認。
4. A_max(area上限)の具体値が未確定。BEVFormer PLAN §6-3 と同じ。
5. m2032 の抜け道テスト(`--boreas-a-compatible`)は YOLOPX では再実行していない(BEVFormer 側で SDK 側のターゲット定義欠落と確定しており、モデル非依存のため。§3.4)。
6. **`dnn_compiler` に一過性のヒープ破壊バグがある**(§3.6)。m2072 の1回目が CP-SAT スケジューリング中に `malloc(): invalid size (unsorted)` で core dump した。**同一設定の再試行で成功したため回避策は「そのまま再実行」**だが、長時間コンパイル(約110分)を捨てることになるため、複数構成をまとめて回す場合は失敗を検出して自動再試行する仕組みが望ましい。発生頻度は不明(観測 1/3 回、m2048/m2024 では未発生)。
7. **power の絶対値がドキュメント値と乖離する**(m2048 実測 0.743W vs レポート 0.679W、+9.4%)。Interconnect Power は完全一致し Functional Unit Power のみずれるため、v26.05.2 で FU 電力係数が変更された可能性が高い(§4.1)。**同一 SDK 版内での相対比較(本探索の 48 vs 72)には影響しない**が、レポート値と実測値を混在させた比較は不可。
8. **m2048 と m2072 の 2 点しか比較できていない**。24 は容量超過、32 はターゲット定義なしで、SDK が露出する `--amp-arch` の粒度がこの 2 点に限られるため。「48 未満で YOLOPX を動かす」選択肢の評価には SDK 外の手段(タイル数の直接指定など、未確認)が必要。

---

## 7. 次のアクション

1. **製品構成の決定を待つ**: 48 ACE 専用 SKU か、BEVFormer と共通の 72 ACE SKU か。前者なら面積 33% 削減、後者なら YOLOPX は 7.69ms / 0.829W で動作(§5.1)。この判断が A_max(§6-4)とモデル構成の決定に依存する。
2. **A_max の確定**: 面積上限が 253mm² 未満なら、SDK が露出する範囲では YOLOPX を載せられない(§6-8)。
3. **精度実測(データセット入手後)**: BDD100K を配置し `convert_model.py steps=eval_trained` でドキュメント値(§2)を再現確認。`num_aces` は精度に影響しないため本探索の結論は変わらないが、量子化後精度の run 間変動幅の把握に必要(§6-3)。
4. **`--optimization-effort` の感度確認(任意)**: 本探索は全構成で 500(high_optimization 既定)に固定した。48 ACE の 9.39ms には 71.5% のマージンがあるため、effort を下げてコンパイル時間(約110分)を短縮できるかは未検証。

---

## 参照

- [PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) — 本探索の方法論の元。BEVFormer-Tiny での同一探索
- [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) — `mythic-compiler` / `mythic-ppa-estimators` の使い方
- `doc/reports/Compiler Optimization Report - YOLOPX.pdf`(v3.0) — m2048 構成の照合基準値の出典
- `doc/reports/Model Summary Report.pdf`(v3.0) §9 — YOLOPX の精度・PPA サマリの出典
- `doc/user-guides/YOLOPX Retraining Guide for Mythic AMP.pdf` — 再学習手順
- [02_ppa_estimation.md](02_ppa_estimation.md) — PPA推定式・定数・未算入項目の解析
- [03_accuracy_simulation.md](03_accuracy_simulation.md) — 精度シミュレーションの仕組み
