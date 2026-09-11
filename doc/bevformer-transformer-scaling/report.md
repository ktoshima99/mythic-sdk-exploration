# BEVFormer-Tiny Transformer構成スケーリング実測レポート(内部版)

**社内限定。** 本ドキュメントは`doc/reverse-engineering/`配下の逆アセンブル・`strings`調査結果を引用している箇所を含む。外部公表用には、それらを一切含まない[report_external.md](report_external.md)を参照すること。

[doc/reverse-engineering/07_ppa_improvement_challenges.md](../reverse-engineering/07_ppa_improvement_challenges.md) §3-3・§5.2は、BEVFormer-Tiny Transformerの既存PPA実測(4.63 ms / 1.245 W@30fps、MAC利用率9.69% — [05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §4.1、[06_hybrid_digital_and_structural_analysis.md](../reverse-engineering/06_hybrid_digital_and_structural_analysis.md) §2.1-2.3)が**たった1点のみ**であり、encoder/decoder層数や1層あたりのテンソルサイズ(`bev_h_`/`bev_w_`/`embed_dims`)を変えたスケーリング実測が行われていないことを明記している。本レポートは、そのために新規実装したスイープツール(`tools/digital_ppa/transformer_config_sweep/`)を用いて実際に9点の実測を行った結果と、そこから得られた知見をまとめる。

数値の一次出典は全て本レポート付属のスイープ結果(`tools/digital_ppa/transformer_config_sweep/results/sweep_transformer_config.json`)であり、既存ドキュメントの数値は基準点との一致検証(§3)にのみ引用する。外挿・未検証の記述には**[推測]**を付す。

---

## 目次

- [1. 位置づけとスコープ](#1-位置づけとスコープ)
- [2. 計測手法](#2-計測手法)
- [3. 検証結果(基準点の再現)](#3-検証結果基準点の再現)
- [4. スイープ結果](#4-スイープ結果)
- [5. 分析](#5-分析)
- [6. 制約・スコープ外](#6-制約スコープ外)
- [7. 再現方法](#7-再現方法)
- [8. 参照](#8-参照)

---

## 1. 位置づけとスコープ

対象はBEVFormer-Tinyの**Transformer単体グラフ**(`TRANSFORMER_PART_ONLY=True`、ResNet-50バックボーンを含まない)。理由:

- この構成が既存の基準実測値(4.63 ms/1.245 W@30fps)を生成した構成そのものである。
- バックボーン込みのフルグラフ(`tools/digital_ppa/run_full_digital.py`が生成する構成)はバックボーン(ResNet-50)支配的なコストがTransformerのスケーリング信号を薄めてしまう上、1点あたりの計測コストも大きい。

スイープ対象パラメータは2軸、それぞれ独立な1次元スイープ(基準点を中心に他パラメータは固定):

- **層数軸**: `encoder.num_layers`(既定3)、`decoder.num_layers`(既定6)
- **入力サイズ軸**: `bev_h_`/`bev_w_`(BEVグリッド、既定50×50)、`embed_dims`(隠れ次元、既定256)

基準点(enc=3, dec=6, bev=50×50, C=256)を含め、合計9点を実測した。

---

## 2. 計測手法

### 2.1 ツール構成

新規実装したツールは `tools/digital_ppa/transformer_config_sweep/` に配置し、既存の`tools/digital_ppa/run_full_digital.py`・`sweep_system_config.py`とSDK本体は変更していない(読んでイディオムを模倣したのみ)。

| スクリプト | 役割 |
|---|---|
| `dynamic_transformer_builders.py` | encoder/decoderの層数Nを可変にしたソース生成型ビルダー |
| `power_component_breakdown.py` | 電力のコンポーネント別内訳(`pow*Pj`係数ゼロ化差分法)の自動化 |
| `run_transformer_config_point.py` | 1点分のパイプライン(config上書き→ビルド→量子化→explore→電力内訳→JSON出力) |
| `sweep_transformer_config.py` | 外側オーケストレータ(点ごとにsubprocess起動) |
| `verify_dynamic_layers.py` | 動的ビルダーが手書きビルダーと一致することの検証 |

### 2.2 層数可変ビルダーの実装方針

`bevformer.modeling.encoder.build_bevformer_tiny_encoder`(N=3)と`bevformer.modeling.decoder.build_bevformer_tiny_decoder`(N=6)は、config値`num_layers`を一切読まずPython側で層をハードコード展開している。層数を可変にするため、`@script`デコレータ付きグラフ関数のPythonソースを文字列として動的生成し、`exec()`する方式を採用した(ネイティブの`for`ループでeagerなsub-layer呼び出し可能オブジェクトを回す方式は採用していない — 各`layer_i_*`は独自の乱数重みを持つ別個の関数オブジェクトであり、ONNXの`Loop`ノードでは「イテレーションごとに異なる重みの関数を呼ぶ」ことを表現できないため)。

実装上、2点の技術的な工夫が必要だった:

1. onnxscriptの`@script`デコレータは`inspect.getsource()`でソースを取得するため、`exec()`で生成した関数はそのままでは動かない(`OSError: could not get source code`)。生成ソースを`linecache.cache`に登録することで解決した。
2. `@script`デコレータの名前解決(`inspect.getmodule(f).__dict__` + `inspect.getclosurevars(f).nonlocals`)は、`layer_i_j`等のsub-layer呼び出し可能オブジェクトが**真のPythonクロージャ変数**であることを前提にしている。そのため、生成した`@script`関数を`_factory(layer_0_0, layer_0_1, ...)`という外側関数の中にネストして定義し、実際のsub-layerオブジェクトを引数として渡して呼び出す構成にした。これにより、手書き版(`build_bevformer_tiny_encoder(config)`内のローカル変数としての`layer_0_0`等)と同じクロージャ構造になる。

### 2.3 電力コンポーネント別内訳の自動化

[05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §5・[06_hybrid_digital_and_structural_analysis.md](../reverse-engineering/06_hybrid_digital_and_structural_analysis.md) §2.3は、`[sys]` cfgの`pow*Pj`係数群を手動でグループごとにゼロ化し、`Power@eff. fps`の差分を読むという手法を手作業で行っていた。`power_component_breakdown.py`はこれをスクリプト化し、以下6コンポーネントについて自動計測する: `mac_unit`・`non_mac_unit`・`dmem_imem`・`ocram`・`ddr`・`bus_noc`。任意の`.vidir`に対して独立に実行可能なCLIとしても提供する。

### 2.4 実行環境

`gcr.io/mythic-devops/compilerd-bin:v26.05.2`コンテナ(`mythic_digital_ppa`)内で、`/mythic/pyvnnsdk-env/bin/python`を用いて実行した。モデルの重みは全てランダム初期化(`initialize_weight()` → `np.random.random(shape)`)であり、量子化も1サンプルのダミー校正(`QuantizationConfig(calibration_dataset_size=1)`)である — 本レポートで測定しているのはグラフ形状に基づく静的なPPA特性であり、モデルの精度や実際の学習済み重みには依存しない。

各config点はそれぞれ独立したPythonプロセス(subprocess)として実行した。理由は、`initialize_onnx()`がプロセス全体で共有される`onnxscript.values.Opset.cache`を走査し、キャッシュ済みの関数定義を次にビルドするモデルに全て追加してしまうため、1プロセス内で複数の異なるconfig点を続けてビルドすると古い/形状の異なるキャッシュ済み関数定義が別の点に漏れ込むリスクがあるためである。

---

## 3. 検証結果(基準点の再現)

新しいパラメータ化パイプライン(動的ビルダーを組み込んだ状態、N=3/6は既存のハードコード値と同じ)で基準点を実行し、既知の基準実測値と完全一致することを確認した:

| 指標 | 既知の基準値 | 本パイプラインでの再現値 |
|---|---|---|
| MACサイクル | 7,193,800 | 7,193,800 |
| non-MACサイクル | 1,742,748 | 1,742,748 |
| exposed DMAサイクル | 1,180,216 | 1,180,216 |
| 総サイクル | 10,116,764 | 10,116,764 |
| eff. fps | 216.15 | 216.15 |
| eff. latency | 4.63 ms | 4.63 ms |
| MAC利用率 | 9.69% | 9.69% |
| Power@eff.fps | 8968.07 mW | 8968.07 mW |
| Power@30fps | 1244.71 mW | 1244.71 mW |

電力コンポーネント別内訳も[06_hybrid_digital_and_structural_analysis.md](../reverse-engineering/06_hybrid_digital_and_structural_analysis.md) §2.3の公開表と一致した(DMEM/IMEM 34.7%・non-MAC unit 28.3%・DDR 18.4%・MAC unit 12.7%・OCRAM 5.9%・Bus/NoC ~0%)。

さらに、動的生成した層数N=3(encoder)/N=6(decoder)のビルダーが手書きビルダーと**ノード列・接続・イニシャライザまで完全一致**する`ModelProto`を生成することを、`verify_dynamic_layers.py`で個別に確認した(固定シードによる乱数重み一致トリックを使用)。

以上3点により、以降のスイープ結果は信頼できるものとして扱う。

---

## 4. スイープ結果

基準点(enc=3, dec=6, bev=50×50, C=256)を中心に、層数軸・入力サイズ軸それぞれ独立に振った9点の実測結果(生データは`tools/digital_ppa/transformer_config_sweep/results/sweep_transformer_config.json`):

| tag | enc | dec | bev | C | MACサイクル | non-MAC | exposed DMA | 総サイクル | レイテンシ | MAC利用率 | Power@30fps | MACs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 6 | 50×50 | 256 | 7,193,800 | 1,742,748 | 1,180,216 | 10,116,764 | 4.63 ms | 9.69% | 1244.71 mW | 16.529 bn |
| bev_25 | 3 | 6 | 25×25 | 256 | 4,370,989 | 1,439,121 | 340,908 | 6,151,018 | 3.14 ms | 9.09% | 656.50 mW | 10.511 bn |
| bev_75 | 3 | 6 | 75×75 | 256 | 11,603,821 | 2,248,788 | **40,078,099** | 53,930,708 | **21.38 ms** | 3.37% | **4871.14 mW** | 26.559 bn |
| embed_128 | 3 | 6 | 50×50 | 128 | 4,152,394 | 1,558,390 | 269,676 | 5,980,460 | 3.02 ms | 5.00% | 542.11 mW | 5.576 bn |
| embed_384 | 3 | 6 | 50×50 | 384 | 6,753,684 | 1,927,156 | 2,190,542 | 10,871,382 | 5.53 ms | 16.11% | 2260.76 mW | 32.860 bn |
| enc_1 | 1 | 6 | 50×50 | 256 | 4,683,872 | 1,472,116 | 199,260 | 6,355,248 | 3.24 ms | 8.96% | 668.89 mW | 10.695 bn |
| enc_5 | 5 | 6 | 50×50 | 256 | 9,703,728 | 2,013,380 | **12,869,424** | 24,586,532 | **10.02 ms** | 6.05% | 2256.74 mW | 22.363 bn |
| dec_2 | 3 | 2 | 50×50 | 256 | 5,028,748 | 856,288 | 186,720 | 6,071,756 | 3.06 ms | 10.20% | 888.29 mW | 11.506 bn |
| dec_10 | 3 | 10 | 50×50 | 256 | 9,358,852 | 2,629,208 | 1,606,000 | 13,594,060 | 6.19 ms | 9.44% | 1572.08 mW | 21.552 bn |

電力コンポーネント別内訳(最大寄与のコンポーネント):

| tag | 最大寄与コンポーネント | 割合 |
|---|---|---|
| baseline | dmem_imem | 34.7% |
| bev_25 | dmem_imem | 38.4% |
| bev_75 | **ddr** | **67.1%** |
| embed_128 | non_mac_unit | 43.6% |
| embed_384 | ddr | 39.5% |
| enc_1 | dmem_imem | 38.4% |
| enc_5 | **ddr** | **38.6%** |
| dec_2 | dmem_imem | 36.5% |
| dec_10 | dmem_imem | 34.4% |

---

## 5. 分析

### 5.1 MACs(演算量)は層数に対して厳密に線形

encoder層数を1→3→5と振ったMACs(10.695 / 16.529 / 22.363 bn)は、ΔN=2ごとに**厳密に+5.834 bn**(1層あたり2.917 bn)増加している。decoder層数を2→6→10と振ったMACs(11.506 / 16.529 / 21.552 bn)も、ΔN=4ごとに**厳密に+5.023 bn**(1層あたり1.256 bn)増加している。これは各層が固定サイズの重み(embed_dims=256のC×C型行列)を持つモデル構造から理論的に予想される通りであり、ツールの正しさの追加的な裏付けにもなっている。

embed_dims(隠れ次元C)については、C=128/256/384での`MACs/C`比が0.0436/0.0645/0.0856 bn、`MACs/C²`比が0.000340/0.000252/0.000223 bnと、`/C`比よりも`/C²`比の方がばらつきが小さい。多くの行列演算の重みがC×C型であることと整合的であり、embed_dimsに対するMACsの増加は層数軸よりも急である**[推測、3点のみからの定性的傾向]**。

### 5.2 レイテンシ・電力はMACsに比例せず、DDRスピルの有無で不連続に跳躍する

MACsが滑らかに増加する一方、レイテンシと電力は**そうではない**。特に以下の3点で`exposed DMAサイクル`が急増し、レイテンシ・電力が跳躍している:

- `bev_75`(BEVグリッド75×75、基準の2.25倍のセル数): exposed DMA 1,180,216 → 40,078,099(**34倍**)、レイテンシ 4.63 → 21.38 ms、Power@30fps 1244.71 → 4871.14 mW
- `enc_5`(encoder層数5、基準の1.67倍): exposed DMA 1,180,216 → 12,869,424(**11倍**)、レイテンシ 4.63 → 10.02 ms
- `embed_384`(隠れ次元384、基準の1.5倍): exposed DMA 1,180,216 → 2,190,542(1.9倍)、レイテンシ 4.63 → 5.53 ms

この3点はいずれも`DDR Write (MB)`が**0より大きい**(bev_75: 436.5 MB、enc_5: 168.5 MB、embed_384: 36.6 MB)。一方、跳躍が起きていない点(`bev_25`・`embed_128`・`enc_1`・`dec_2`)は`DDR Write (MB)`が**厳密に0.0**である:

| tag | maxOCR (kB) | maxDDR (kB) | DDR Read (MB) | DDR Write (MB) | exposed DMA / 総サイクル |
|---|---|---|---|---|---|
| baseline | 1022.00 | 23,908.22 | 123.5 | 17.1 | 11.7% |
| bev_25 | 998.41 | 10,925.52 | 27.1 | **0.0** | 5.5% |
| bev_75 | 972.56 | 80,016.80 | 1347.5 | **436.5** | **74.3%** |
| embed_128 | 966.50 | 13,633.22 | 15.1 | **0.0** | 4.5% |
| embed_384 | 987.17 | 58,585.53 | 527.5 | **36.6** | 20.1% |
| enc_1 | 1004.88 | 8,156.84 | 18.9 | **0.0** | 3.1% |
| enc_5 | 1022.00 | 51,933.63 | 267.4 | **168.5** | **52.3%** |
| dec_2 | 1022.00 | 21,408.22 | 106.4 | **0.0** | 3.1% |
| dec_10 | 1022.00 | 31,042.38 | 137.0 | 24.4 | 11.8% |

`maxOCR`は全点でほぼ一定(966〜1022 kB、`OCRAM1`の容量1024 kBに近い値)であるのに対し、`DDR Write`は小さい点(0.0〜17.1 MB: `bev_25`・`embed_128`・`enc_1`・`dec_2`・`baseline`)と大きい点(24.4〜436.5 MB: `dec_10`・`embed_384`・`enc_5`・`bev_75`)の間に明瞭な差がある(厳密な二値ではなく、baseline・dec_10のように小さいが非ゼロの書き込みを伴う点も存在する — 正確な境界は§5.4で追加実測により特定する)。これは、**活性化(および重み、§5.4参照)のワーキングセットがオンチップ容量に収まる限りは中間結果をDDRに書き出す必要がなく`exposed DMA`は小さいままだが、収まらなくなると中間結果のDDR往復(スピル)が発生し、`exposed DMA`サイクルが跳躍する**という閾値的な挙動を強く示唆する。

[07_ppa_improvement_challenges.md](../reverse-engineering/07_ppa_improvement_challenges.md) §4.2はAnalog(ACE)側のSRAM/ACE境界比という同種の律速要因切り替えの枠組みを導出している。本実測は、Digital側でも同様の「オンチップ容量境界」による律速要因の不連続な切り替わりが実際に起きることを、Transformer層数・入力サイズという新しい軸で確認したものである。

### 5.3 MAC利用率はどの軸でも一貫して低いままである

MAC利用率(`efficiency_pct`)は3.4%〜16.1%の範囲に収まり、どの構成でも[05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md)が指摘する低MAC利用率(基準点9.69%)から大きく改善しない。層数や入力サイズを変えても、Deformable Attentionの少数サンプリング点に起因する小さな行列積という構造自体は変わらないため、利用率の改善は本質的に構造変更(サンプリング点数やヘッド数の再設計)が必要と考えられる**[推測]**。

### 5.4 補足調査: 重みのDDR退避が発生する容量境界の推定

§5.2で見た`exposed DMAサイクル`の跳躍は、`Max DDR Weights (kB)`(重みのうちDDR常駐が必要になった分)が0からゼロでない値に切り替わる境界と対応している可能性が高いという仮説を検証するため、以下の追加実測を行った(生データ: `tools/digital_ppa/transformer_config_sweep/results/ocram_threshold_bisection.json`):

- `bev_h_`/`bev_w_`を55・60・65・70で細分化(基準50と75の間)
- `encoder.num_layers=4`(基準3と5の間)
- `decoder.num_layers=8`(基準6と10の間)
- `embed_dims=320`(基準256と384の間)

このために`run_transformer_config_point.py`の出力に`Max DDR Weights (kB)`・`Max DDR IO (kB)`・`Max OCR WEIGHTS (kB)`・`Max OCR IO (kB)`の内訳を追加した(vnnmapの`maxDDR`/`maxOCR`行にもとから含まれる`wgt/bas/sft`(重み)/`inp/out`(活性化)の内訳を取得するのみで、追加のvnnmap実行は不要)。

**まず確認された事実(ドキュメント調査による)**: `bevformer.cfg`は`OCRAM0=33,554,432`バイト(32MB)・`OCRAM1=1,048,576`バイト(1MB)・`pCluster=12`を指定している。[05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §4.3は、フルグラフ(バックボーン込み)構成で`OCRAM0`を32MB→512MBに増やすとDDR Write/Readがほぼゼロになることを実測しており、`OCRAM0`がオンチップに載るかどうかを支配する主要な容量パラメータであることを示している。一方、`maxOCR`/`maxDDR`メトリクスの計算ロジック自体(重みと活性化のどちらがどの条件でDDRに退避するかを決める正確な判定式)は、既存ドキュメント中で逆アセンブルされていない未解明事項として明記されている([01_compilation.md](../reverse-engineering/01_compilation.md) §3.4.3の`[推測(strings根拠)]`注記、[05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §9.2の未解決課題リスト)。判明している唯一の手がかりは、コンパイラ内の文字列リテラル`"Not enough OCR in cluster %u switch to DDR in layer %u/%u"`・`"Cut before layer %2u/%2u => weight %7.1f kB (OCR %7.1f kB)"`であり、この判定が**クラスタ単位・レイヤー単位**で行われることを示唆している(`pCluster=12`と整合)。

**そのため、汎用的な解析式は既存資料からは得られない。以下は本レポートで実測により初めてブラケットした値である。**

全16点(元の9点+追加7点)を`Model Size (MB)`(モデル全体の静的重みサイズ)でソートすると、4つの軸(層数・BEVサイズ・embed_dims)を横断して極めて明瞭な境界が現れる:

| Model Size (MB) | Max DDR Weights (kB) | tag |
|---|---|---|
| 5.708 | 0.00 | embed_128 |
| 7.842 | 0.00 | dec_2 |
| 10.495 | 0.00 | enc_1 |
| 11.562 | 0.00 | bev_25 |
| **13.622** | **0.00** | **baseline** |
| **14.199** | **6485.50** | **bev_55** |
| 14.830 | 6951.07 | bev_60 |
| 15.185 | 4295.60 | enc_4 |
| 15.517 | 6893.50 | bev_65 |
| 16.259 | 6981.50 | bev_70 |
| 16.512 | 2868.50 | dec_8 |
| 16.748 | 8198.59 | enc_5 |
| 17.055 | 6852.50 | bev_75 |
| 18.751 | 10763.75 | embed_320 |
| 19.402 | 5883.66 | dec_10 |
| 24.661 | 15227.25 | embed_384 |

**16点全てが、モデル全体の重みサイズ13.622 MB(baseline)と14.199 MB(bev_55)の間の1点の境界できれいに二分される** — これ未満では常に`Max DDR Weights=0`(重みが完全にオンチップに載る)、これ以上では常に`Max DDR Weights>0`(境界を超えた途端に2.9〜15.2 MBという大きな塊が一度にDDRへ退避する)。この「境界を超えると数MB単位で一気に退避する」という不連続な挙動は、上記の文字列リテラルが示す**レイヤー単位の判定**(1レイヤー分の重み丸ごとがOCRに収まるかDDRに切り替えるかの二択)と整合する。

ただし2点、精度の限界として明記する:
1. `Max DDR Weights`/`Max OCR Weights`はモデル全体の重み合計(13〜25 MB)よりずっと小さい値(数百kB〜十数MB)であり、これは「推論全体を通じたある瞬間のピーク値」であって「モデル全体が同時にどこかに存在する」という意味ではない。したがって「モデル全体の重みサイズが13.622〜14.199 MBの境界を超えるとDDR退避が起きる」という関係は、本実測で振った4軸(層数・BEVサイズ・embed_dims)においては**厳密に成り立つ実用上の予測式**だが、モデル全体の重みサイズそのものが直接比較される容量値である保証はない**[推測]**――より正確には、境界のすぐ内側にある特定の1レイヤーの重みサイズが、その時点でOCRに残っている空き容量(活性化に使われている分を除いた残り)を超えるかどうかで決まっている可能性が高い。
2. 活性化側(`exposed DMAサイクル`・`Max DDR IO`)は同じ境界に対して滑らかには増加しない。`bev_55`(N_q=3,025)の`exposed DMA`(442,758)は`baseline`(N_q=2,500、1,180,216)より**小さく**、`bev_65`(2,467,958)は`bev_60`(3,023,655)より小さいなど、単調ではない。これは活性化側の配置判定もレイヤー・タイル単位の離散的な決定であり、`bev_h_×bev_w_`の総セル数のような単一の連続量では説明できないことを示している。

この境界の正確な発動条件(どのレイヤーの、どの容量チェックが、正確にどの閾値バイト数で発動するか)を確定するには、[05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §2で`Cycles per inference`ブロックに対して行ったのと同様の`gdb`によるディスアセンブル(`tools/digital_ppa/probe_vnnmap_cycles.py`の手法を`maxOCR`/`maxDDR`の出力箇所に適用する)が必要であり、これは本レポート・既存資料のいずれでも未着手である。

**重要な注記: `bevformer.cfg`のOCRAM0=32MB/OCRAM1=1MBは、本レポートが探索用に選んだ値ではなく、SDKの公式パイプラインがBEVFormer-TinyのPPA数値を算出する際に実際に使っている、ハードコードされた値である。** `mythic-compiler`が生成する`bevformer_postprocessing.py`は、

```python
model = BevformerTiny(result_directory)
run_vnn_flow(model, result_directory,
    system_config=Path(__file__).parent / "system_configs" / "bevformer.cfg",
    skip_validation=True, advanced=True)
```

という形でこのcfgへのパスをコンパイル時に埋め込み、`vnnmap --explore --edma`を実行してデジタルNPU側のPPA(JSON)をコンパイル済みアーティファクトに焼き込む。`mythic-ppa-estimators`はこのJSONをそのまま読むだけで、OCRAM0/OCRAM1をユーザーが変更できるフラグは一切持たない([05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §4.1で報告されている"Digital Estimated Frame Processing: 4.63 ms"が、本レポートの基準点実測4.63msと完全一致するのはこのため)。したがって、本節で示した13.622〜14.199 MBの境界は、BEVFormer-Tinyについて**SDKが実際に公式PPA数値を計算する際の境界**である。

ただし、この32MB/1MBという数値自体が、実際のM2000チップの物理的なSRAM容量と一致しているかどうかは、依然として未確認である:

- [05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §6・§9.2は、デジタル側の面積(v-MPコア面積・SRAMマクロ面積/MB)が「SDK外の情報でしか見積もれない」「全デジタル評価の最大の未確定要素」と明記し、OCRAM0の現実的上限の確定を未解決課題として挙げている。
- 実データシート`Mythic_M2000_NPU_Datasheet_v0.3.pdf`は"Shared SRAM"/"on-chip SRAM"を定性的に記述するのみで、容量の数値は一切記載していない。
- 公式にサポートされるPPA Estimator(`Mythic_PPA_Estimator_Datasheet_v0.4.pdf` §5)がユーザー向けに提供する固定の「virtual platform」(例: 24Ace6Tile・48Ace12Tile)は、Analog側(ACEアレイ・タイル構成)を指すものであり、Digital側のこの`[sys]` cfg(OCRAM0/OCRAM1/nMPs)とは別の仕組みである。`bevformer.cfg`はSDK開発者がモデルごとにハードコードした値であって、同データシートが言う「ユーザーが自由設定できないハードウェア構成」の対象そのものではない――つまりデータシートはこの値の現実性について何も述べていない。

したがって、「32MB/1MBという値がSDKの公式PPA計算で実際に使われている」ことは確定したが、「32MB/1MBが実チップの真の容量と一致する」ことは依然として本レポート・既存資料のいずれからも確認できない**[未確認]**。実チップの容量がこれと異なる場合、SDKが現在報告しているPPA数値自体が実チップとズレている可能性がある。

---

## 6. 制約・スコープ外

- スイープは各軸3〜4点の小規模スイープであり、層数×入力サイズのクロス項(交互作用)は測定していない。
- `bev_h_`と`bev_w_`は常に正方形(`bev_h_ == bev_w_`)として振っており、非正方形グリッドは未測定。
- DDRスピルの閾値(ワーキングセットのどの時点でOCRAM容量を超えるか)を明示的に定式化してはいない。§5.2の観察は9点の実測データからの相関的な読み取りであり、閾値の解析的な予測式は導出していない。
- `Efficiency (%)`はハードウェア`[sys]`config(`nMACs`等)を一切変えていないため、実効MAC並列度に対する感度は本レポートの対象外([05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §8/§9.2で指摘される固定64 MAC/cycle/MP定数の限界も参照)。
- Analog(ACE)側のバックボーンは対象外(§1参照)。

---

## 7. 再現方法

```bash
# コンテナ起動(既存のdigital-PPAワークフローと同じイメージ)
docker run -d --name mythic_digital_ppa --memory=200g \
    -v <repo>/tools/digital_ppa/transformer_config_sweep:/work \
    gcr.io/mythic-devops/compilerd-bin:v26.05.2 sleep infinity

# 検証(ゲート1): 動的ビルダーが手書きビルダーと一致するか
docker exec mythic_digital_ppa /mythic/pyvnnsdk-env/bin/python /work/verify_dynamic_layers.py

# スイープ本体
docker exec mythic_digital_ppa /mythic/pyvnnsdk-env/bin/python /work/sweep_transformer_config.py /work/sweep_out
```

結果は`<repo>/tools/digital_ppa/transformer_config_sweep/results/sweep_transformer_config.json`に格納されている。

---

## 8. 参照

- [doc/reverse-engineering/07_ppa_improvement_challenges.md](../reverse-engineering/07_ppa_improvement_challenges.md) §3-3・§4.2・§5.2 — 本レポートが実測で埋めた未測定事項の出典、SRAM/ACE境界比の枠組み
- [doc/reverse-engineering/05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §4.1・§5・§8/§9.2 — 基準実測値、電力コンポーネント別内訳の手法、efficiency%の制約、OCRAM0拡大によるDDR trafficの実測(§4.3)
- [doc/reverse-engineering/06_hybrid_digital_and_structural_analysis.md](../reverse-engineering/06_hybrid_digital_and_structural_analysis.md) §2.1-2.3 — 基準実測値の再確認、電力コンポーネント別内訳の公開表
- [doc/reverse-engineering/01_compilation.md](../reverse-engineering/01_compilation.md) §3.4.1・§3.4.3 — OCRAM0/OCRAM1のcfgキー定義、OCR→DDR切り替えを示す文字列リテラル(`Cut before layer...`等)、`nMPs∈{1,4,8}`という実コンパイルパスの制約と`explore_model`の無制約性の対比
- `mythic_sdk/v26.05.2/doc/datasheets/Mythic_M2000_NPU_Datasheet_v0.3.pdf` — 実データシート。SRAMは定性的記述のみで容量数値は非公開
- `mythic_sdk/v26.05.2/doc/datasheets/Mythic_PPA_Estimator_Datasheet_v0.4.pdf` §5 — 公式PPA Estimatorが固定virtual platform単位でのみハードウェア構成を提供し、SRAM容量の自由設定は未リリース機能である旨の記述
- `tools/digital_ppa/run_full_digital.py`・`tools/digital_ppa/sweep_system_config.py`・`tools/digital_ppa/probe_vnnmap_cycles.py` — 本レポートのツールが模倣したイディオム、および§5.4で今後必要とされるディスアセンブル手法の参照元
- `tools/digital_ppa/transformer_config_sweep/` — 本レポートの実測に使用した新規ツール一式
- `tools/digital_ppa/transformer_config_sweep/results/sweep_transformer_config.json` — §3・§4の数値の一次データ(9点)
- `tools/digital_ppa/transformer_config_sweep/results/ocram_threshold_bisection.json` — §5.4の容量境界推定に使用した追加7点の一次データ
