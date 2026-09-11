# BEVFormer-Tiny Transformer構成スケーリング実測レポート

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

`maxOCR`は全点でほぼ一定(966〜1022 kB、OCRAM1の容量1024 kBに近い値)であるのに対し、`DDR Write`はゼロかゼロでないかの二値的な切り替わりを示している。これは、**活性化のワーキングセットがOCRAM(オンチップスクラッチパッド、1MB)に収まる限りは中間結果をDDRに書き出す必要がなく`exposed DMA`は小さいままだが、収まらなくなった瞬間に中間結果のDDR往復(スピル)が発生し、`exposed DMA`サイクルが一気に跳躍する**という閾値的な挙動を強く示唆する。

[07_ppa_improvement_challenges.md](../reverse-engineering/07_ppa_improvement_challenges.md) §4.2はAnalog(ACE)側のSRAM/ACE境界比という同種の律速要因切り替えの枠組みを導出している。本実測は、Digital側でも同様の「オンチップ容量境界」による律速要因の不連続な切り替わりが実際に起きることを、Transformer層数・入力サイズという新しい軸で確認したものである。

### 5.3 MAC利用率はどの軸でも一貫して低いままである

MAC利用率(`efficiency_pct`)は3.4%〜16.1%の範囲に収まり、どの構成でも[05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md)が指摘する低MAC利用率(基準点9.69%)から大きく改善しない。層数や入力サイズを変えても、Deformable Attentionの少数サンプリング点に起因する小さな行列積という構造自体は変わらないため、利用率の改善は本質的に構造変更(サンプリング点数やヘッド数の再設計)が必要と考えられる**[推測]**。

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
- [doc/reverse-engineering/05_all_digital_ppa.md](../reverse-engineering/05_all_digital_ppa.md) §4.1・§5・§8/§9.2 — 基準実測値、電力コンポーネント別内訳の手法、efficiency%の制約
- [doc/reverse-engineering/06_hybrid_digital_and_structural_analysis.md](../reverse-engineering/06_hybrid_digital_and_structural_analysis.md) §2.1-2.3 — 基準実測値の再確認、電力コンポーネント別内訳の公開表
- `tools/digital_ppa/run_full_digital.py`・`tools/digital_ppa/sweep_system_config.py` — 本レポートのツールが模倣したイディオムの参照元
- `tools/digital_ppa/transformer_config_sweep/` — 本レポートの実測に使用した新規ツール一式
- `tools/digital_ppa/transformer_config_sweep/results/sweep_transformer_config.json` — 本レポートの数値の一次データ
