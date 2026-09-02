# 06. 実ハイブリッド構成のデジタル側ボトルネックとモデル構造解析

状態: 調査完了。すべてブラックボックス手法(既存コンテナ`mythic_digital_ppa`での`vnnmap`直接実行・`[sys]`cfgスイープ・ONNXグラフの静的解析)による。逆アセンブル・gdbは使用していない。

対象の問い:

1. BEVFormer実ハイブリッド構成(72 ACE、アナログ=ResNet-50バックボーン、デジタル=Transformer部分)におけるデジタル側のレイテンシ・電力のボトルネックは何か。[05_all_digital_ppa.md](05_all_digital_ppa.md)の「全デジタル実行」の結論(メモリ帯域が支配的)は、実際にデジタル実行される範囲(Transformer部分のみ)にもそのまま当てはまるか。
2. BEVFormer/YOLOPXのACEクロスバー充填率の差(32.6% vs 49.6%、[07_ppa_improvement_challenges.md](07_ppa_improvement_challenges.md) §4.3)を、実際のモデル構造(backbone種別・層ごとのチャネル数)から説明できるか。

結論の要約:

| 問い | 結論 |
|---|---|
| 実ハイブリッドのデジタル側ボトルネック(レイテンシ) | exposed DMAではない(11.7%止まり)。MAC利用率9.69%という低効率——Attention/GridSample/Reshapeの低計算密度——が支配的(§2.2) |
| 実ハイブリッドのデジタル側ボトルネック(電力) | DDRアクセスではない(18.4%止まり)。DMEM/IMEM(ローカルメモリアクセス、34.7%)とNon-MAC演算(28.3%)が支配的(§2.3) |
| YOLOPXのbackboneアーキテクチャ | ResNet系ではなくYOLOv7系E-ELAN。ヘッドはYOLOX式decoupled anchor-free head(§3.2) |
| クロスバー充填率差(32.6% vs 49.6%)の構造的説明 | 部分的に説明可能(BEVFormerのResNet-50は272で割り切れない大チャネル層が多い)。ただしMAC総量の集計に未解決の不一致があり、確証ではなく有力な仮説にとどまる(§3.5, §3.6) |

---

## 目次

- [1. 手法](#1-手法)
- [2. 実ハイブリッド構成のデジタル側ボトルネック(BEVFormer Transformer部分)](#2-実ハイブリッド構成のデジタル側ボトルネックbevformer-transformer部分)
- [3. モデル構造のONNX解析(BEVFormer/YOLOPX)](#3-モデル構造のonnx解析bevformeryolopx)
- [4. 留保事項](#4-留保事項)
- [5. 参照](#5-参照)

---

## 1. 手法

本ドキュメントの実測はすべて以下のいずれかで得ており、逆アセンブル・gdbは一切使用していない。

- **§2(デジタル側)**: 既存コンテナ`mythic_digital_ppa`(v26.05.2の`compilerd-bin`イメージ、[05_all_digital_ppa.md](05_all_digital_ppa.md) §3.1で構築済み)内で、出荷スクリプトの既定構成(`BevformerTiny.TRANSFORMER_PART_ONLY=True`)から生成済みの`.vidir`(`/work/out_tf_ctl/BevformerTiny.vidir`)に対し、`vnnmap`バイナリを`--explore --edma`で直接実行し、stdoutの`Cycles per inference:`/`Inference performance:`ブロックをそのまま読む。電力の成分分解は[05_all_digital_ppa.md](05_all_digital_ppa.md) §5.1で確認された「`pow*Pj`係数は`[sys]`のcfgキーとして上書き可能」という性質を利用し、1グループの係数だけを0にしたcfgで再実行し、`Power@eff. fps`の差分を見る手法による。
- **§3(構造解析)**: `onnx`(v1.22.0)でYOLOPX/BEVFormerのONNXファイルを直接ロードし、全Convノードの入出力チャネル数・カーネルサイズ・group数を集計する静的解析。学習コード等は参照せず、ONNXグラフのノード名・チャネル推移から構造を判定した。

---

## 2. 実ハイブリッド構成のデジタル側ボトルネック(BEVFormer Transformer部分)

### 2.1 前提: 全デジタル評価(05)との違い

[05_all_digital_ppa.md](05_all_digital_ppa.md)が測定した「全デジタル実行」は、ResNet-50バックボーンを含むフルグラフをすべてデジタル(v-MP)で実行した場合の**仮想シナリオ**である(MACs 904.354 bn、latency 58.87 ms、Power@30fps 25.58 W)。

実際のハイブリッド構成(BEVFormer m2072、72 ACE)では、ResNet-50バックボーンはアナログACEで処理され、**デジタル側が担うのはTransformer部分のみ**である。この部分の規模は全デジタル評価とは大きく異なる(出典: `bevformer_tiny.py::TRANSFORMER_PART_ONLY=True`の既定構成、[05_all_digital_ppa.md](05_all_digital_ppa.md) §4.1の「transformer のみ」列と同一の実行):

| 項目 | 全デジタル(backbone含む) | **実ハイブリッドのデジタル部分(Transformerのみ)** |
|---|---|---|
| MACs | 904.354 bn | **16.529 bn**(54.7分の1) |
| Model size | 54.611 MB | **13.622 MB** |
| Total cycles | 117,605,925 | **10,116,764** |
| fps / latency | 16.99 / 58.87 ms | **216.15 / 4.63 ms** |
| MAC利用率 | 41.67% | **9.69%** |
| Power@30fps | 25.582 W | **1.245 W** |
| Max DDR | 439,778 kB | **23,908.22 kB** |

[07_ppa_improvement_challenges.md](07_ppa_improvement_challenges.md) §3-3で言及されている「デジタル側の固定フロア4.63ms」はこの実ハイブリッド値であり、全デジタル評価の58.87msとは無関係である。しかし同§3-4が明記する「デジタル実行のボトルネックはメモリ帯域である」という結論は全デジタル評価から得られたものであり、**実ハイブリッドのデジタル部分(Transformerのみ)にそのまま適用できるかは別途検証が必要**だった。以下がその検証結果である。

### 2.2 サイクル内訳の実測(レイテンシのボトルネック)

既定cfg(`system_configs/bevformer.cfg`: nMPs=288, frequency=2GHz, xTile=2)で`vnnmap`を直接実行し、stdoutの`Cycles per inference:`ブロックを読んだ結果:

```
MAC:            7,193,800  (71.1%)
non MAC:        1,742,748  (17.2%)
exposed DMA:    1,180,216  (11.7%)
Total:         10,116,764
```

```
eff. fps: 216.15   eff. latency: 4.63 ms   efficiency: 9.69%
```

**exposed DMAは11.7%に過ぎず、全デジタル評価(40.8%)のように支配的ではない。** Transformer部分の重み・活性化サイズが小さく(Max DDR 23.9 MB、対して全デジタル評価は439.8 MB)、DDR往復自体が大きな負荷にならないためである。

支配的なのは**MAC cycles(71.1%)でありながらMAC利用率が9.69%しかない**という組み合わせである。すなわち「計算に割り当てられているサイクル時間」は長いが、そのうち実質的な乗算加算に使われている割合はごく一部である。

**3つの要因をサイクル数の重みで順位付けすると、以下のようになる。**

1. **(最大の要因、71.1%)MAC分類サイクルの低充填。** `MAC cycles`はサイクル数として最大の内訳(71.1%)だが、その内部の利用率はわずか9.69%であり、大半が空転している。Deformable Attentionの少数サンプリング点による小さな行列積は、サイクル配分式`getMacCycles(n, nMACs, x) = (n/nMACs) × x`(§2.3)に投入する次元(n)自体が小さいため、64 MAC/cycle/MPの並列レーンの大部分を空けたままサイクルを消費する。**サイクル数で見た場合、ボトルネックの主要因はこれであり、「non-MAC演算のサイクルが多い」わけではない。**
2. **(第2の要因、17.2%)non-MAC演算(Softmax/Reshape等)。** これらはMAC演算をゼロしか生まないままサイクルを消費するが、比率は1.より小さい。
3. **(最小の要因、11.7%)exposed DMA。** §2.1で述べた通り、全デジタル評価(40.8%)と対照的にこの規模のモデルでは支配的でない。

**因果の向きに関する注意**: `efficiency % = MACs_bn × 1e9 × 100 / ((procCycles+exposedDmaCycles) × 64 × nMPs)`(`05_all_digital_ppa.md` §2.2)という定義から、利用率(efficiency%)はサイクル数とMAC量から逆算される**派生指標**であり、それ自体が独立にサイクル数を増加させる原因ではない。「利用率が低いためサイクル数が増える」ではなく、**Attentionの小さな行列積という演算構造が、実質MAC量に対して不釣り合いに多くのMAC分類サイクルを要求しており、その結果として利用率が低く観測される**、という因果関係が正確である。

これに加えて、[02_ppa_estimation.md](02_ppa_estimation.md)/[05_all_digital_ppa.md](05_all_digital_ppa.md) §2.4が指摘する「1層あたり理想MAC時間の110%を下限とするフロア」も、計算密度の低い層で頻繁に発動し要因1のサイクルをさらに押し上げている可能性がある**[推測]**——この因果はcfgスイープでは直接検証できておらず、層別のフロア発動状況を確認するには`<model>_flow.csv`の解析が必要(次の調査候補、§4)。`MAC cycles(71.1%)`の内部が具体的にAttentionの行列積・FFN(Linear層)・検出ヘッドのConvのどの割合で構成されているかも、同様に層別解析なしには分解できていない。

**含意**: 実ハイブリッド構成のデジタル側レイテンシ改善において、OCRAM拡張のようなメモリ帯域対策([05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3の主要レバー)は主要な効果を持たない。有効な対策は、Attention/GridSample/Reshapeをより計算密度の高い表現に変換するか、これらの演算自体の発行回数を減らす構造変更である。

### 2.3 電力内訳の実測(`pow*Pj`係数の分離)

[05_all_digital_ppa.md](05_all_digital_ppa.md) §5.1が明らかにした「`pow*Pj`係数は`[sys]`のcfgキーとして上書き可能」という性質を利用し、係数グループを1つずつ0にしたcfgで再実行し、`Power@eff. fps`(既定8968.07 mW、eff. fps=216.15での電力)からの減少分を各成分の寄与とみなした。

**単位についての注意**: 本節の基準値8968.07 mWは`Power@eff. fps`(このTransformer部分が単独で出す実効fps=216.15での電力)であり、[05_all_digital_ppa.md](05_all_digital_ppa.md) §5.2で定義される`Power@30fps`(1244.71 mW=1.245 W。§2.1・§3-3・§6で「デジタル側電力」として引用している値)とは異なる基準である。両者は同一の1推論あたりエネルギー(41.49 mJ、`Power÷fps`で一致)を異なるfpsへ線形外挿した値の関係にあり、`Power@30fps = Power@eff.fps × (30/216.15)`という単純なスカラー倍で変換できる。この変換は全成分に同一の係数を掛けるだけなので、**各成分の比率(%)はfps基準に依存せず同一であり、以下の結論は`Power@30fps`(1.245 W)基準でもそのまま成立する**(絶対値はmW列を30/216.15倍すればよい)。

| 成分 | ゼロ化後の`Power@eff.fps` | 寄与(差分、`Power@eff.fps`基準) | 寄与(`Power@30fps`基準に換算) | 比率 |
|---|---|---|---|---|
| **DMEM/IMEM(ローカルスクラッチパッド)** | 5852.88 mW | 3115.19 mW | 432.4 mW | **34.7%** |
| **Non-MAC unit(elementwise/正規化等)** | 6429.79 mW | 2538.28 mW | 352.3 mW | **28.3%** |
| DDR(`powDdrReadPj`/`WritePj`) | 7314.24 mW | 1653.83 mW | 229.5 mW | 18.4% |
| MAC unit(`powMacUnitPjPerCycle`) | 7833.23 mW | 1134.84 mW | 157.5 mW | 12.7% |
| OCRAM(M/C read/write) | 8442.14 mW | 525.93 mW | 73.0 mW | 5.9% |
| Bus/NoC(`powBusMatrix*`/`powNoc*`) | 8968.07 mW(変化なし) | 0 mW | 0 mW | 0.0% |
| **合計** | — | 8968.07 mW | **1244.7 mW(≈1.245 W)** | 100.0% |

(比率の合計が100.0%に一致することから、電力モデルが§5.1の記述通り5項の単純な加算であることが再確認できる。`Power@30fps`基準の合計1244.7 mWが、§2.1で引用した既知値1244.71 mWと一致することも確認済み。)

**DDRアクセスは18.4%に過ぎず、全デジタル評価で確認された「OCRAM拡張で28.6%削減」(§4.3, [05_all_digital_ppa.md](05_all_digital_ppa.md))のような支配的コストではない。** Bus/NoCがゼロなのは、この規模のモデルでは`mCluster=1`かつタイル間通信がほとんど発生しないためと考えられる**[推測]**。

支配的なのは**DMEM/IMEM(34.7%)とNon-MAC unit(28.3%)を合わせた63.0%**である。これは§2.2のレイテンシ側の知見(MAC cyclesが多いが利用率が低い/non-MAC cyclesが17.2%)と整合的で、Attention系演算が「ローカルメモリへの読み書きを伴うelementwise/整形処理」を多く含み、これがレイテンシだけでなく電力側でも支配的コストになっていることを示す。

**含意**: 実ハイブリッド構成のデジタル側電力改善において、DDR/OCRAM容量の調整([05_all_digital_ppa.md](05_all_digital_ppa.md)の主要レバー)は限定的な効果しかない。有効な対策はレイテンシ側と同じく、Attention系演算の構造そのものを見直すことである。

### 2.4 ms単位への変換とその限界

§2.2のサイクル比率を、ツールが実際に報告する`eff. latency`(4.63 ms)に適用すると、以下の近似的なms内訳が得られる:

| 内訳 | サイクル比率 | 推定ms(4.63 ms × 比率) |
|---|---|---|
| MAC(計算時間) | 71.1% | 約3.29 ms |
| Non-MAC | 17.2% | 約0.80 ms |
| Exposed DMA | 11.7% | 約0.54 ms |

この近似には留保が必要である。単純に`Total Cycles ÷ frequency`(10,116,764 ÷ 2 GHz = 5.058 ms)を計算すると、ツールが報告する`eff. latency`(4.63 ms)より**約8.5%大きい値**になる。[05_all_digital_ppa.md](05_all_digital_ppa.md) §2.4が指摘した「stdoutの`Total`と実際にfps/latencyの分母になる内部集計は別系統」という問題が、全デジタル評価(backbone中心、ズレ0.125%)よりもTransformer部分(ズレ約8.5%)で大きく現れている。`xTile`を2→1にすると、このズレは8.5%→4.3%に縮小したが解消はしなかった(§2.5)。この残差の正確な内訳を特定するには、[05_all_digital_ppa.md](05_all_digital_ppa.md) §2.4-2.5と同水準のgdbプロービングが必要であり、本調査(ブラックボックス限定)の範囲では確定できない。したがって上表のms内訳は**比率としては信頼できるが、絶対値は最大10%程度の誤差を持つ近似値**として扱うべきである。

### 2.5 `xTile`感度実験

```
[sys] ... xTile=1 (既定2から変更、他は既定のまま)
```

| 項目 | xTile=2(既定) | xTile=1 |
|---|---|---|
| MAC cycles | 7,193,800 | 6,435,463 |
| non-MAC cycles | 1,742,748 | 1,764,195 |
| exposed DMA cycles | 1,180,216 | **2,536,511** |
| Total cycles | 10,116,764 | 10,736,169 |
| eff. fps / latency | 216.15 / 4.63 ms | 194.50 / 5.14 ms |
| efficiency | 9.69% | 8.72% |
| DDR Read / Write | 123.5 / 17.1 MB | 67.7 / 4.9 MB |
| Power@eff.fps / @30fps | 8968.07 / 1244.71 mW | 6579.25 / 1014.82 mW |

`xTile`を1に減らすとDDRトラフィックは減る(123.5→67.7 MB)が、**exposed DMA cyclesは倍以上に増加し(1,180,216→2,536,511)、結果としてレイテンシは悪化する(4.63→5.14 ms)**。これは[05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3が確認した「`xTile`を4に増やすと悪化する」という観察とは逆方向の変化点で、タイル数が少ないほどDMA隠蔽の機会(タイル間のパイプライン重複)も減ることを示唆する**[推測]**。`xTile`の最適値は2付近にあると考えられるが、既定値(2)以外の値(3以上)は本調査では未確認。

---

## 3. モデル構造のONNX解析(BEVFormer/YOLOPX)

### 3.1 使用ファイルと手法

| モデル | ファイル | 内容 |
|---|---|---|
| YOLOPX | `mythic_sdk/v26.05.2/archive/models/training/yolopx/multiclass_yolopx_fp32.onnx` | 標準Conv、162層。BCM変換前のフルグラフ |
| YOLOPX | `.../yolopx/yolopx_trained.onnx` | `MythicConv2d`カスタムop、206層。BCM変換後(チャネルを8の倍数へパディング済み) |
| BEVFormer | `mythic_sdk/v26.05.2/archive/models/training/bevformer/bevformer-tiny-fp32-1600x900.onnx` | 標準Conv、55層 |
| BEVFormer | `.../bevformer/bevformer-tiny-1600x900-trained.onnx` | `MythicConv2d`、74層(= [07_ppa_improvement_challenges.md](07_ppa_improvement_challenges.md) §4.3の`BCMConv2d`数74と一致) |

全Convノードについて入出力チャネル数・カーネルサイズ・group数を抽出し、`MACs = out_H × out_W × out_C × in_C × kH × kW / groups`で層別MAC数を算出した。PyTorch学習コードはYOLOPX側にSDK内に見つからず(post-processingスクリプトのみ)、構造判定はONNXノード名とチャネル推移から行った。

### 3.2 YOLOPXのbackboneアーキテクチャの同定

ノード名(`layer_1`〜`layer_5`、各ブロックが`cv1/cv2/cv3/cv4`+`Concat`、ダウンサンプルは`MaxPool`+`cv1/cv2`の2経路)とチャネル推移(3→32→64→128→256→512→1024)から、**YOLOv7系のE-ELAN(Extended-ELAN)backbone**と判定できる。

- **Neck**: `SPPF`+`head_elan_1〜4`+`mp1/mp2`+`Resize`という典型的なYOLOv7 PAFPNネック。
- **Head**: `stems`+`cls_convs`/`reg_convs`+`cls_preds`/`reg_preds`/`obj_preds`という**YOLOX式decoupled anchor-freeヘッド**(off-chip)。
- `model.5`以降の3ブランチは`drive_seg`/`ll_seg`の2つのセグメンテーションデコーダ。

「YOLOP+X(YOLOX)」というモデル名は、この「YOLOv7系E-ELAN backbone + YOLOX式decoupled head」という構成をそのまま反映している。depthwise/grouped convは**両モデルとも未使用**(全Conv層でgroup=1)。

### 3.3 チャネル分布比較

| | stem | 中間層代表 | 終盤層代表 | backbone最大out_C |
|---|---|---|---|---|
| YOLOPX(E-ELAN) | 3→32→64 | layer_3 cv1: 256→128 | layer_5 cv1: 1024→512 | 1024 |
| BEVFormer(ResNet-50) | 3→64(7x7, stride2) | layer2 conv3: 128→512 | layer4 conv3: 512→2048 | 2048(+FPN 2048→256) |

### 3.4 クロスバー次元(1280入力×272出力)との適合度

「in_C<1280 かつ out_C<272」(クロスバー1回のドット積に対して小さすぎる層)の比率:

| モデル | 比率(fp32) | 比率(trained) |
|---|---|---|
| YOLOPX | 89.5% | 90.3% |
| BEVFormer | 58.2% | 55.4% |

「out_C>272」(列方向の分割が必要になる層)の比率:

| モデル | 比率 |
|---|---|
| YOLOPX | 9.9% |
| **BEVFormer** | **40.0%** |

BEVFormerのResNet-50はBottleneck構造の1x1 conv(256/512/1024/2048ch出力)が多く、out_C>272の層が全体の40%を占める。YOLOPXはE-ELAN構造で大チャネル層が少なく(9.9%)、分割自体が発生しにくい。

「in_C>1280」(行方向の分割が必要になる層)は、BEVFormerに3層(FPN側方接続含む)、YOLOPXにSPPF concat後の1層のみ。

### 3.5 MAC総量の整合性チェック(未解決の不一致)

ONNXから計算した層別MAC合計を、既知の実測値([07_ppa_improvement_challenges.md](07_ppa_improvement_challenges.md) §4.3のACE MACs)と比較した:

| モデル | ONNX計算(fp32) | ONNX計算(trained) | 既知の実測値(ACE MACs) | 比率 |
|---|---|---|---|---|
| YOLOPX | 171,341,153,280 | 184,123,883,520 | 285,021,216,768 | **60.1〜64.6%** |
| BEVFormer(backbone、6カメラ換算) | 735,367,987,200 | 735,367,987,200 | 955,023,360,000 | **77.0%** |

(BEVFormerのbackbone単カメラMACは122,561,331,200で、fp32/trainedとも完全一致。6カメラ分は単純に×6で換算。)

**両モデルとも既知の実測値に対して23〜40%不足しており、一致していない。** この差は、コンパイラのタイル分割時の境界オーバーラップ再計算やクロスバー内チャネルパディングなど、ONNXの静的グラフからは見えないハードウェア実行時の追加コストが原因である可能性が高いが、本調査(ONNX静的解析のみ)では特定できていない。**以下§3.6の議論は、この未解決の不一致を踏まえ、確証ではなく有力な仮説として扱う。**

### 3.6 クロスバー充填率差(32.6% vs 49.6%)との関係

[07_ppa_improvement_challenges.md](07_ppa_improvement_challenges.md) §4.3が示すクロスバー充填率(平均MAC/回)は、BEVFormer 113,402 MAC/回(32.6%)、YOLOPX 172,805 MAC/回(49.6%)である。

§3.4の結果は、直感(小チャネル層が多いほど充填率が下がる)とは**逆の傾向**を示す。YOLOPXは「小さすぎる層」の比率がBEVFormerより高い(89.5–90.3% vs 55.4–58.2%)にもかかわらず、充填率はYOLOPXの方が高い。

見つかった構造的な傾向は逆で、**BEVFormerのResNet-50はout_C>272の層が40%を占め、その多く(1024ch, 2048ch)が272の倍数でない**ため、列方向にクロスバー分割したときに割り切れない「余りブロック」が生じ、その余りブロックが不完全充填になる、という機構が有力に見える。YOLOPXは大チャネル層が少なく(9.9%)分割自体がほぼ発生しないため、この種の「分割余りロス」が構造的に起きにくい**[推測]**。

ただし、この仮説は平均MAC/回の差(113,402 vs 172,805、約1.52倍)を数値的に再現するものではなく、§3.5のMAC総量不一致も未解決のままである。**この構造的傾向は部分的な説明にとどまり、確証ではない。** より確実な検証には、コンパイラの`auto_partition`が実際にどうタイル分割・列分割を行っているかのログ(コンパイル時の中間出力)の解析が必要で、これは本調査の範囲外である。

---

## 4. 留保事項

| 留保 | 内容 |
|---|---|
| ms内訳は近似値 | §2.4。`Total Cycles ÷ frequency`とツール報告値の間に4〜8.5%のズレがあり、その原因はブラックボックスでは特定できていない |
| フロア発動の因果は未検証 | §2.2。「Attention系の低密度層でフロアが頻発する」という説明は[推測]であり、層別`flow.csv`の解析で検証していない |
| Bus/NoCゼロの一般性は未検証 | §2.3。今回のモデル規模・cfg(`mCluster=1`)固有の結果である可能性がある |
| MAC総量の不一致は未解決 | §3.5。ONNX計算値が既知の実測値より23〜40%小さい。原因(タイル分割オーバーヘッド等)は仮説のみ |
| 充填率差の説明は部分的 | §3.6。272で割り切れない大チャネル層という構造的傾向は見つかったが、平均MAC/回の差を定量的に再現できていない |
| `xTile`のスイープは2値のみ | §2.5。1と2のみを確認。3以上の挙動は未確認 |
| 逆アセンブル不使用 | 本ドキュメント全体。内部の正確な計算式が必要な場合は[05_all_digital_ppa.md](05_all_digital_ppa.md) §2.2-2.5, §5.1-5.3を参照 |

---

## 5. 参照

- [05_all_digital_ppa.md](05_all_digital_ppa.md) — 全デジタル実行の実測・内部式の逆アセンブル解析。本ドキュメント§2はこのドキュメントの手法(`vnnmap --explore`、`[sys]`cfgスイープ、`pow*Pj`分離)を実ハイブリッド構成に適用したもの
- [07_ppa_improvement_challenges.md](07_ppa_improvement_challenges.md) §3-1, §3-3, §3-4, §4.3 — 本ドキュメントが検証対象とした既存の結論・[推測]タグ
- [02_ppa_estimation.md](02_ppa_estimation.md) §3.4, §3.9 — アナログ側のACE/SRAM時間式、クロスバー次元(1280×272)の定義
- `mythic_sdk/v26.05.0/_extracted_compiler/vnnsdk_scripts/bevformer/bevformer_tiny.py` — `TRANSFORMER_PART_ONLY`既定構成
- コンテナ`mythic_digital_ppa`(`gcr.io/mythic-devops/compilerd-bin:v26.05.2`)、`/work/out_tf_ctl/BevformerTiny.vidir` — §2の実測に使用した既存アーティファクト(再利用可能)
