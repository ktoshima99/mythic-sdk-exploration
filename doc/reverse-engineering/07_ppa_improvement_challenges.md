# 07. PPA改善の課題整理 — 既存探索の横断統合

Mythic M2000 SDK上でのAIモデルPPA(Power/Performance/Area)改善に必要な事項と主要な課題を、既存の実測結果を横断的に整理したもの。対象は主に BEVFormer-Tiny([PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md))と YOLOPX([PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md))の2モデルのSKU探索結果、および全デジタル実行の実測([05_all_digital_ppa.md](05_all_digital_ppa.md))。

本ドキュメントは**新規のコード調査・実機実行を行わず**、既存ドキュメント(`00_overview.md`〜`05_all_digital_ppa.md`, `conversion_steps/to_structural.md`, `conversion_steps/to_training.md`, `PLAN_*`, `HOWTO_*`, `FUTURE_*`)の記述・実測値を再整理・外挿したものである。数値の一次出典はすべて各節に明記する。外挿・未検証の記述には**[推測]**を付す。**例外として§3-10は、既存の`_ppa_*.tar.gz`(コンパイル成果物。再コンパイル・funcsim再実行は不要)に対し`power_estimator.py`を新規に実行し、電力のコンポーネント別内訳を実測している(手法・スクリプトは`tools/power_breakdown/power_breakdown.py`、§3-10冒頭参照)。§3-11・§4.3の一部・§6は、[06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md)で実施した新規のブラックボックス実測(既存コンテナでの`vnnmap`直接実行・ONNXグラフ静的解析。逆アセンブル不使用)を要約したものであり、詳細な手法・生データは同ドキュメントを参照。**

---

## 目次

- [1. 位置づけ](#1-位置づけ)
- [2. SDKが露出するPPAレバーの総括](#2-sdkが露出するppaレバーの総括)
- [3. 主要課題](#3-主要課題)
- [4. 律速要因別に見たモデル構造の指針](#4-律速要因別に見たモデル構造の指針)
- [5. LLM/VLA等、将来の多層Transformerモデルへの課題](#5-llmvla等将来の多層transformerモデルへの課題)
- [6. モデル別ボトルネック総括(BEVFormer/YOLOPX)](#6-モデル別ボトルネック総括bevformeryolopx)
- [7. まとめ表](#7-まとめ表)
- [8. 参照](#8-参照)

---

## 1. 位置づけ

BEVFormer-TinyとYOLOPXという2モデルのSKU探索([PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md), [PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md))は、それぞれ独立に「`num_aces`をどこに設定すべきか」という問いに答える形で完了している。両探索は互いに正反対の結論(BEVFormerは72 ACEが必須、YOLOPXは48 ACEが最適)に至っており、この対比自体がPPA改善の課題の多くを含んでいる。本ドキュメントはこの2つの探索結果と、SRAM分解手法([02_ppa_estimation.md](02_ppa_estimation.md) §3.9)、全デジタル実行の実測([05_all_digital_ppa.md](05_all_digital_ppa.md))、コンパイラのグラフ書き換え規則([01_compilation.md](01_compilation.md))、on/off-chipマーキング機構([to_structural.md](conversion_steps/to_structural.md))を統合し、以下の3点に答える形で整理する:

1. これらのAIモデルのPPA改善に何が必要か。主要な課題は何か(§3)。
2. レイテンシ改善に適したモデル構造とは何か——SRAM-boundなモデルではSRAMアクセスを減らす構造、ACE-boundなモデルではACE演算数(クロスバー利用率)を減らす構造という、律速要因ごとに異なる2つの問いに分けて答える(§4)。
3. 今後、より多層のTransformer/LLM/VLAモデルを検討する際、デジタル側で処理せざるを得ないTransformerブロックの計算量を削減すべきか、そしてそれ以外にどのような課題が立つか(§5)。

スコープ外: 本ドキュメントは既存探索の統合であり、pythia(GPT-NeoX系)やLLM/VLA相当モデルについての新規PPA実測は行っていない。§5の記述の多くは既存の2モデル(いずれも畳み込み中心のバックボーンを持つ)の実測からの外挿であり、層数の多いTransformerデコーダそのものを対象にした実測ではない点に注意。

---

## 2. SDKが露出するPPAレバーの総括

出典: `PLAN_bevformer_ppa_exploration.md` §1.1、[02_ppa_estimation.md](02_ppa_estimation.md) §6。

| レバー | 場所 | 値域 | 効き先 | 既知の限界 |
|---|---|---|---|---|
| `num_aces`(アナログACE数) | `--amp-arch`(コンパイラYAML `COMPILER_OPTIONS`) | SDK同梱`funcsim`は24/32/48/72の4値に対応するが、実際にコンパイルが通るのは**48/72のみ**(24はCP-SATタイムアウトまたは重み容量超過、32はターゲット定義自体が存在しない) | die area(単調)、ACEクリティカルパス、SRAMトラフィック(複製経由) | area/powerの結合(§3-2)、モデル依存で最適値が正反対になる(BEVFormer=72必須、YOLOPX=48最適) |
| `n_mps` | `vnnmap/compilation.py` | `{1,4,8}`のみ(デジタル側) | タイル分割・並列度 | BEVFormer既定YAML内に対応キーが見当たらず未確認(`HOWTO_ppa_exploration_tools.md` §4) |
| `nMPs`(全デジタル実行時の`[sys]`キー) | `vnnmap`の`system_config` | 既定288(BEVFormer全デジタル構成) | デジタルMACアレイ数 | 増やすと**悪化**する(§3-4、`05_all_digital_ppa.md` §4.3) |
| OCRAM0/OCRAM1 / DDR | `vnnmap`の`system_config`(`[sys]`) | 既定32MB/1MB/可変 | DDR↔OCRAM昇格、exposed DMAサイクル削減 | 効果が非現実的な容量でしか出ない場合がある(512MBで33ms達成、§3-4) |
| `frequency` | `system_config` | CLIフラグなし、直書きが必要 | レイテンシ全体に線形 | サイクルモデルには入らず、DDRレイテンシもcycle単位のため高周波側は楽観的に振れる(`05_all_digital_ppa.md` §2.2) |
| `QuantizationConfig.tensor_n_bits` | `vnnort/quantizer/quantization_config.py` | **8か16のみ**(4/6bitは`ValueError`) | 精度⇄PPA | 既存2モデルは既定8bitで精度制約を満たすため未探索(§3-5) |
| コンパイラeffortフラグ(`--target-frame-rate`/`--parallelism-multiplier`/`--auto-parallelize`/`--opt`) | `dnn_compiler --help` | 既定300fps/1.0/0.0(オフ)/default | 同一area内でのlatency-powerトレードオフ | 本探索群では未着手(`PLAN_bevformer_ppa_exploration.md` §7) |
| 内部ソルバーフラグ(`--relative-objective-target`等) | `dnn_compiler`バイナリ内部 | [社内資料由来・未検証] | ACE利用率の目標値経由でlatency短縮 | フラグ名・既定値がコード上で直接検証できていない(`PLAN_bevformer_ppa_exploration.md` §1.3) |

**露出されていないもの**: ADC/DACビット深度(8bit固定)、sparsity、BitSpreadingMode(クローズドソースの`dnn_compiler`/`vnnmap`内部で決定)。デジタル側のarea係数(v-MPコア単体の面積、SRAMマクロ面積/MB)もSDK外(§3-7)。

### 2.1 P/P/A軸別の整理——各パラメータがどの軸に効くか

上表のレバーをArea/Performance(latency)/Powerの3軸に分けて再整理する。

**Area(A)に影響するパラメータ**

| パラメータ | 効き方 |
|---|---|
| `num_aces` | **唯一の主要面積パラメータ**。物理傾き5.278 mm²/ACEで単調増加(24→158mm², 48→253mm², 72→380mm²、`PLAN_bevformer_ppa_exploration.md` §4.1) |

デジタル側(v-MPコア)の面積係数はSDKから一切露出されない(§3-7)。`nMPs`(MACアレイ数)・`pCluster`は構成として振れるが、それを面積へ変換する係数がSDK内に存在しないため、デジタル側のAは原理上パラメータ化できない。

**Performance(latency)に影響するパラメータ**

公表レイテンシは`max(ACEクリティカルパス, SRAM時間, SIMD時間)`(アナログ)+デジタル側latency(直列加算)という構造のため(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.4, §3.8)、律速項ごとに効くパラメータが変わる。

| パラメータ | 効く経路 |
|---|---|
| `num_aces` | ACEクリティカルパスを短縮(並列化)。SRAM-boundなら効かない(§3-1) |
| モデル構造(層形状・充填率・重み再利用度) | ACE演算回数・SRAM往復量そのものを規定(§4.2, §4.3) |
| `n_mps` | タイル分割・並列度(BEVFormerでは対応YAMLキー未確認) |
| OCRAM0/OCRAM1/DDR | DMA隠蔽量(デジタル側)。512MBまで拡張して初めて33ms達成という非現実的な例(§3-4) |
| `frequency` | デジタル側latencyに線形。DMAモデルがcycle単位のため高周波側は楽観的に振れる |
| `nMPs`(全デジタル実行時) | 一見効きそうだが**増やすと悪化**(exposed DMA支配、§3-4)——直感に反する逆効果レバー |
| `xTile` | デジタル側タイル数。増やすとDMAサイクル爆発(xTile=4で10.9倍、§3-4) |
| `dmaThreshold`/`readDiv*`/`readLatency*` | デジタル側DMA隠蔽判定の閾値・速度(cfgから上書き可能) |
| コンパイラeffortフラグ | 同一area内でのlatency-powerトレードオフ(本探索群では未着手) |

**Power(power)に影響するパラメータ**

| パラメータ | 効く経路 |
|---|---|
| `num_aces` | 重み複製→SRAMトラフィック増→power増(固定fps時、§3-2)。「ACEを増やせばpowerが下がる」という直感が成立しない主因 |
| 推論レート(fps) | `power = 1推論あたりエネルギー × 推論レート`で直接線形([02_ppa_estimation.md](02_ppa_estimation.md) §4) |
| `frequency` | デジタル側`Power@30fps`は per-inference エネルギーの外挿値のため実は**不変**(§3-6、[05_all_digital_ppa.md](05_all_digital_ppa.md) §5.2)——高周波でも変わらないという非直感的挙動 |
| `nMPs`/OCRAM(デジタル側) | DDRトラフィック経由でpowerに影響(nMPs増→悪化、OCRAM増→改善、§3-4) |
| `tensor_n_bits` | 量子化ビット幅(8/16のみ)。ビット幅が上がるほどSRAMトラフィック・演算コストが増える方向(4/6bitは選択肢自体がない、§3-5) |
| `pow*Pj`係数群(デジタル)/`ENERGY_TABLE`(アナログ) | プロセスノード依存(28/12/5nm)のアクセス単価そのもの。通常は変更対象ではない固定値 |

**3軸を横断して結合するパラメータ(トレードオフの核)**

最も重要な点は、**`num_aces`だけがArea・Performance・Powerの3軸すべてに同時に効く**ことである(§3-2)。これが「単純な1軸の探索にできない」理由の核心で、他のパラメータ(frequency・OCRAM・`n_mps`・コンパイラeffortフラグ等)は基本的にareaを固定したままlatencyとpower(主にデジタル側)にのみ効く。

| パラメータ | Area | Performance | Power |
|---|---|---|---|
| `num_aces` | ●(唯一の主要因) | ● | ● |
| モデル構造(重み再利用度・充填率) | ○(重み容量経由で間接) | ● | ● |
| OCRAM/DDR/`n_mps`/`frequency`(デジタル側) | ✕(露出なし) | ● | ● |
| `tensor_n_bits` | ✕ | ○ | ● |
| コンパイラeffortフラグ | ✕(area固定のまま) | ● | ● |

`num_aces`が3軸を同時に動かす一方、他のレバーは基本的に「areaを固定したままP/Pだけを動かす」性質を持つ。これは`PLAN_bevformer_ppa_exploration.md` §7が挙げる「同area内でのlatency-powerトレードオフ軸」という次の探索候補の位置づけとも整合する。

---

## 3. 主要課題

以下、実測に基づく課題を優先度順(実効性・影響範囲の大きい順)に列挙する。各項目は「事実→含意」の形式。

### 3-1. 律速項がモデルごとに異なる(ACE-bound vs SRAM-bound)

公表レイテンシは`max(ACEクリティカルパス, SRAM時間, SIMD時間)`で決まる(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.4)。この3項の大小関係はモデルによって入れ替わる:

| モデル(SKU) | ACEクリティカルパス | SRAM時間(アクセス法) | 律速 |
|---|---|---|---|
| YOLOPX m2048 | 7.56 ms | 6.83 ms | **ACE-bound** |
| BEVFormer-Tiny m2072 | 23.37 ms | **26.95 ms** | **SRAM-bound** |

出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.9。

**含意**: ACE-boundなモデル(YOLOPX)ではACE増設がレイテンシ短縮に直接効くが、SRAM-boundなモデル(BEVFormer)ではACEを増やしてもSRAM項が頭打ちを続けるため改善しにくい([02_ppa_estimation.md](02_ppa_estimation.md) §3.9.1)。ACE利用率(YOLOPX 72.75%、BEVFormer 69.43%)だけを見ていては両者を区別できない——律速項の判定は「そのモデルにACE増設が効くか」を決める前提条件であり、モデル構造の変更やSKU選定の前に必ず行うべき最初のステップになる。判定は既存の`_ppa_*.tar.gz`から`perf_breakdown.sh`で約9秒で可能(funcsim再実行不要、[02_ppa_estimation.md](02_ppa_estimation.md) §3.9(4))。

#### 3-1.1 ACE数を増やしても律速項自体は反転しない(実測範囲内)[推測]

既存2モデルはいずれも48/72 ACEの2点で実測されており、律速項がACE数を変えることで反転するかを確認できる:

| モデル | ACE数 | ACEクリティカルパス | SRAM時間(=`Analog NPU Processing Time`。両モデルともこちらが`max`) | 律速 | SRAM/ACE比 |
|---|---|---|---|---|---|
| YOLOPX | 48 | 7.56 ms | 6.83 ms | ACE-bound | 0.904 |
| YOLOPX | 72 | 5.86 ms | 4.99 ms | ACE-bound(変化なし) | 0.852 |
| BEVFormer-Tiny | 48 | 35.22 ms | **38.85 ms** | SRAM-bound | 1.103 |
| BEVFormer-Tiny | 72 | 23.37 ms | 26.95 ms | SRAM-bound(変化なし) | 1.153 |

出典: `PLAN_yolopx_ppa_exploration.md` §4.1-4.2、`PLAN_bevformer_ppa_exploration.md` Phase 2/3 実測表。BEVFormerの`Analog NPU Processing Time`(=`maximum_bottleneck_ns`、[02_ppa_estimation.md](02_ppa_estimation.md) §3.4)はSRAM-boundである以上、定義上そのままSRAM時間の値と一致する(ACEクリティカルパスからの残差計算ではない)。

**両モデルとも48→72で律速項は反転せず、むしろ支配側がより支配的になっている**(YOLOPXはACE側の比率が下がり=ACE優位が拡大、BEVFormerはSRAM側の比率が上がり=SRAM優位が拡大)。メカニズム[推測]: ACE数を増やすとACEクリティカルパスは並列化により直接短縮される。SRAM時間も「最も忙しい1タイルあたりの負荷」がタイル分散で下がる効果を受けるが(YOLOPX実測: 最繁忙タイル負荷-31.0%、SRAM時間-26.9%、§3-2参照)、同時に並列化のための重み複製で総SRAMトラフィックは増える(同+5.3%/+7.3%)。この2つの相反する力の綱引きの結果、今回の2モデルではACE項の方が相対的に速く下がったため、支配項の比率が広がる方向に動いた。

この綱引きの結果は決定論的ではなく、**ACE項とSRAM項がほぼ等しい(比率が1に近い)モデルであれば、ACE数を増やした際にどちらが先に下がるかで律速項が反転する可能性は原理的にある**。今回の2モデルはいずれも比率が0.85〜1.15程度で、たまたま反転するほどの拮抗点ではなかっただけであり、「ACE数を増やしても律速項は反転しない」という一般法則としては確立していない。

### 3-2. 並列化のための重み複製がarea・SRAMトラフィック・powerを結合する

コンパイラはACE数が多いほど並列化のために重みを複製する。YOLOPXの48→72 ACE実測:

| 項目 | 48 ACE | 72 ACE | 差 |
|---|---|---|---|
| ACE Weights | 49,858,224 | 58,857,104 | **+18.0%** |
| SRAM Bytes Read | 4,624,739,160 | 4,871,279,544 | +5.3% |
| SRAM Bytes Written | 2,401,121,232 | 2,577,301,008 | +7.3% |
| Total Combined Power@30fps | 0.743 W | 0.829 W | **+11.6%(悪化)** |
| ACE利用率 | 72.75% | 62.59% | -10.2pt(悪化) |

出典: `PLAN_yolopx_ppa_exploration.md` §3.7, §4.2。

**含意**: 30fps固定の推定では、演算量(ACE MACs)が同一のままACE数だけを増やすと、同じ仕事をより多くのハードウェアに分散するために重み複製が増え、SRAMトラフィックとpowerがともに増える。「ACEを増やせば稼働率が下がってpowerが下がる」という直感(`PLAN_bevformer_ppa_exploration.md` §0で当初想定されていたシナリオ)は、latency制約に大きな余裕があるYOLOPXでは成立しない(`PLAN_yolopx_ppa_exploration.md` §5.2)。BEVFormer側はこのシナリオを検証する前提(48 ACEが可行)が成立しなかったため未検証。area・SRAMトラフィック・powerが独立に動かせない変数として結合している点が、SKU選定を単純な1軸の探索にできない理由である。

### 3-3. デジタル側レイテンシは`num_aces`に依存しない固定フロア

BEVFormer-Tinyの Digital Estimated Frame Processing は48 ACE・72 ACEで完全に一致する(4.63 ms、`PLAN_bevformer_ppa_exploration.md` §4)。アナログ側処理時間がSKUで26.95→38.85msと変化する一方、デジタル側は不変。

**含意**: デジタル側の演算(Transformerエンコーダ/デコーダ、検出ヘッド)はMAC比率では1.7%だが処理時間比率では14.5%を占める(出典: [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md))。この4.63msはSKU選定やアナログ側の最適化では一切削れない固定コストであり、全体レイテンシ予算(33ms)のうち常に一定の割合を占有する。Transformer比重がモデル設計上で増えるほど、この固定フロアの絶対値そのものが増えると推測される([推測]。実測はBEVFormer-Tiny 1点のみで、Transformer層数を変えたスケーリング実測は行っていない)。

この4.63ms自体の内部ボトルネック(サイクル内訳・電力内訳)は§3-11で扱う。[05_all_digital_ppa.md](05_all_digital_ppa.md) §3-4が明らかにした「デジタル実行はメモリ帯域が支配的」という結論は、backboneを含む全デジタル評価から得られたものであり、この4.63ms(Transformer部分のみ)にはそのまま当てはまらない点に注意。

### 3-4. デジタル側はメモリ帯域ボトルネックであり、MAC数を増やすと悪化しうる

**本節はResNet-50バックボーンを含むフルグラフを全デジタル実行した場合の結論であり、実際のハイブリッド構成でデジタル実行される範囲(Transformer部分のみ)には別のボトルネックが確認されている(§3-11)。**

BEVFormerフルグラフを全デジタル実行した`nMPs`スイープ(2GHz固定):

| nMPs | fps | latency | MAC利用率 | Power@30fps |
|---|---|---|---|---|
| 288 | 16.99 | 58.87 ms | 41.67% | 25.58 W |
| 576 | 5.99 | 167.03 ms | 7.34% | 27.54 W |
| 1152 | 3.75 | 266.33 ms | 2.30% | 33.46 W |
| 2304 | 4.02 | 248.65 ms | 1.23% | 37.27 W |

出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3。

**含意**: MACアレイを増やすとexposed DMA(隠蔽できないDMAサイクル)が支配的になり、fpsが下がりDDRトラフィックが増えてpowerも悪化する。「デジタル実行のボトルネックはMAC数ではなくメモリ帯域である」([05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3の明記)。効くレバーはOCRAM0(オンチップSRAM)のみで、512MBまで拡張すると初めて33.19ms/18.27Wに達するが、この容量は面積・コストの観点で非現実的([05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3, §6)。この観察は§5(LLM/VLAへの課題)で直接引用する。

#### 3-4.1 SDKが洗練されても「メモリ帯域が支配的」という質的結論は変わりにくいと推測される[推測]

この結論は`vnnmap`の推定の粗さ(§3-7.1)から生じた見かけの結果ではなく、アーキテクチャレベルの構造的事実に根ざしていると考えられる: MACアレイ数(`nMPs`)を増やしてもそれに供給するデータを運ぶメモリ帯域が同時に増えない限りデータ待ちのサイクルが増えるという一般的なroofline制約、オンチップSRAM(OCRAM0)拡張がDRAMラウンドトリップを消すことによる劇的な改善(§3-4本文)、Transformer部分のMAC利用率が9.69%しかないのはAttention/Reshape/GridSampleが密な行列積ではないという演算の性質の問題であること(§5.1)——いずれもSDKの推定精度とは独立にモデル・ハードウェアの構成から出てくる帰結であり、より精緻なシミュレータで再現しても同じ方向の結論になると推測される。

一方、「どの程度」メモリ帯域が支配的かという**定量値**は、現状のSDKに存在する2つの反対方向のバイアスの影響を受けている:

- **DMA隠蔽モデルが二値的な閾値判定である点**(`dmaThreshold`既定765未満は完全隠蔽、超過分のみ露出。出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §2.3)は、実チップがダブルバッファリング等でより滑らかにDMAと計算を重ねられる場合、現在の推定は露出DMAを**過大に見積もっている**(=メモリバウンドを過大評価している)可能性がある。この点が洗練されればメモリバウンド評価は緩和される方向に動くと推測される。
- **DDRレイテンシがサイクル単位でモデル化されている点**([05_all_digital_ppa.md](05_all_digital_ppa.md) §2.3, §8)は、周波数を上げるとDDRアクセス時間も比例して縮む扱いになる楽観的バイアスであり、これを物理的に正しいns基準へ修正すると、逆に**メモリバウンド評価はより深刻になる**方向に動くはずである。

この2つのバイアスは逆方向に働くため、どちらが優越するかは既存ドキュメント群の記述だけでは判断できず、本調査群では実測・検証していない未解決事項である。したがって「メモリ帯域が支配的」という**質的結論**は構造的な理由により反転しにくいと推測されるが、**支配の度合いを表す具体的な数値**はSDKの洗練度に応じて変動しうる。

### 3-5. 量子化ビット幅は8/16bit固定で、精度制約に対して既に絞る余地がない

`QuantizationConfig.tensor_n_bits`は8か16のみ許可され、4/6bitは`ValueError`で拒否される(出典: `PLAN_bevformer_ppa_exploration.md` §1.1)。既存2モデルはいずれも既定8bitで精度制約(FP32同等)を満たしている(`PLAN_bevformer_ppa_exploration.md` §2、`PLAN_yolopx_ppa_exploration.md` §2)。

**含意**: 精度を犠牲にPPAを改善する方向の量子化ポリシー探索は、そもそもSDKが露出する範囲では選択肢がない(下位ビット幅が使えないため)。PPA改善の主戦場は量子化ではなく、SKU選定(`num_aces`)とモデル構造そのものになる。

### 3-6. 電力モデルはLeakage/ClockTree/PCIe/D2D未算入で相対比較専用

アナログ側`power_estimator.py`の`total_power`には Functional Unit と Interconnect のみが算入され、Leakage/Clock Tree/PCIe/Die-to-Dieはいずれもコメントアウトされ未算入(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §4.11)。デジタル側`Power@30fps`も、実際に出るfps(`eff. fps`)でのエネルギーを30fpsへ単純に線形外挿した値であり、「30fpsで動作した場合の電力の物理モデル」ではない(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §5.2)。デジタル側も同様にダイナミック電力のみで、leakage/clock tree/PCIe/D2Dは非含有([05_all_digital_ppa.md](05_all_digital_ppa.md) §5.3)。

**含意**: 本探索群で得られる数値はすべて**同一条件下での相対比較(A/B比較)専用**であり、絶対値のサインオフには使えない。異なるfps基準(BEVFormer m2048の23fpsクランプ vs m2072の30fps)や異なるSDKバージョン間(YOLOPX実測0.743W vs レポート値0.679W、+9.4%)の値を混在させて比較してはならない。

### 3-7. デジタル側の面積はSDKから推定できない

`vnnmap`バイナリの文字列にarea/mm²/die相当のシンボルは存在しない(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §6)。アナログ側は物理傾き5.278 mm²/ACEで算出可能だが、デジタル側はv-MPコア1個あたりの面積・SRAMマクロ面積/MBのいずれもSDK外の情報が必要。

**含意**: ハイブリッド構成(アナログ+デジタル)とオール・デジタル構成をarea軸で公平に比較することは、現状のSDKだけでは不可能。全デジタル案の実行可能性を評価するには、latency/powerだけでなくarea見積り(SDK外のデータシート等)を別途取得する必要がある(既にlatency/powerだけでも全デジタルはハイブリッドに対し1.9倍/5.7倍劣位であることが判明しているため、area面での劣位が確定すれば全デジタル案は完全に棄却できる)。

#### 3-7.1 デジタル側PPA推定はアナログ側に比べ成熟度が低い[推測]

面積の欠落(3-7)は単独の欠陥ではなく、デジタル側PPA推定経路(`vnnmap --explore`)全体がアナログ側(`perf_analysis.py`/`power_estimator.py`)に比べ簡易的な段階にとどまっていることの一部と見るのが妥当と考えられる。根拠となる具体的な指標:

- **キャリブレーションがランダム1サンプル。** デジタル経路の量子化は`QuantizationConfig(calibration_dataset_size=1)`(ランダム1サンプルのダミー量子化)と`skip_validation=True`(数値等価チェックのスキップ)で構成され(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §2)、得られる値は「cycles/powerの見積りにのみ使える」と留保されている(同§8)。
- **効率(利用率)計算が実際の構成を見ない。** `efficiency %`は`64 MAC/cycle/MP`固定で計算され、`nMACs`(実際のMACアレイ構成)を参照しない(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §2.2)。既定値(nMACs=32)では偶然一致するが、`nMACs=64`にすると表示52.52%に対し真の利用率は約26%——構成を変えると表示自体が不正確になる。対照的にアナログ側のACE利用率は`total_ace_ops`と`num_aces`から都度計算される([02_ppa_estimation.md](02_ppa_estimation.md) §3.6)。
- **サイクルモデルに物理的根拠のないヒューリスティックが残る。** MACオーバーヘッド一律+11.1%(`n/9`)、1層あたり「理想MAC時間の110%」を下限とするフロア、DDRレイテンシがサイクル単位でモデル化されているため周波数を上げるとDDRアクセス時間も比例して短くなる扱いになる(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §2.3、§8の留保事項)。アナログ側の電力モデルはADC/AIDACの電流値(50°C typical、電圧×電流の物理式)まで踏み込んだ回路レベルの推定を行っており([02_ppa_estimation.md](02_ppa_estimation.md) §4.4)、この点で対照的。
- **集計系統が2つに分岐し食い違う。** stdout最終ブロックの`Total`と、実際にfps/latencyの分母になる`vSummaryProfile`は別系統の集計で、0.125%の差が生じる(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §2.4)。逆アセンブル調査なしにはこの食い違いは把握できなかった。
- **フルグラフ実行に一次サポートがない。** 出荷スクリプトの既定`TRANSFORMER_PART_ONLY=True`はTransformer部分のみを対象とし、フルグラフを通すには2つのSDK側の問題(ONNX local function重複、Cap'n Protoのblob上限超過)へのモンキーパッチが必要(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §7)。アナログ側の`mythic-ppa-estimators`のような一次CLIや、`perf_breakdown.sh`のような専用デバッグツールに相当するものがデジタル側には存在しない。

一方で、デジタル側の電力モデル自体(DDR/OCRAM/DMEM/IMEM/MAC/NoCの各アクセス種別ごとの`pow*Pj`係数、出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §5.1)やDMAモデル(`readDiv`/`readLatency`/`dmaThreshold`によるDMA隠蔽判定、同§2.3)は、アナログ側の`ENERGY_TABLE`と同程度の粒度を持つ。**「デジタル側に電力モデルが存在しない」わけではなく、面積の欠落・キャリブレーションの粗さ・効率計算の構成非依存性・フルグラフ実行の非一次サポートという複数の点でアナログ側に比べ未成熟、という言い方が正確である。**

### 3-8. コンパイラ/ソルバーの非決定性・既知バグ

- BEVFormer m2024: CP-SAT `status: UNKNOWN`(122分後にタイムアウト。INFEASIBLE証明ではない。出典: `HOWTO_ppa_exploration_tools.md` §1)。
- YOLOPX m2024: 重み容量108.6〜121%超過による**真のINFEASIBLE**(出典: `PLAN_yolopx_ppa_exploration.md` §3.3)。
- YOLOPX m2072: 1回目コンパイルが`malloc(): invalid size (unsorted)`のヒープ破壊で異常終了、同一設定の再試行で成功(出典: `PLAN_yolopx_ppa_exploration.md` §3.6)。
- マルチスレッドソルバーの非決定性により、同一設定でもコンパイル結果が数%変動する(出典: `PLAN_yolopx_ppa_exploration.md` §6-2)。

**含意**: SKU探索の1点あたりコンパイル約90〜120分・PPA推定約1〜4時間という長いイテレーション時間の中に、タイムアウト・INFEASIBLE・一過性バグが混在する。探索の実務コストとして、単純な「コンパイルが通るか」の判定にすら複数回の再試行や重み容量の事前計算(§4参照)が必要になる。

### 3-9. A_max(面積上限)が事業側で未確定

BEVFormerの探索は「A_maxが380mm²以上でなければ、現行SDK・現行モデルでは可行SKUが存在しない」という結論に達しているが、A_max自体の値はビジネス側(コスト・パッケージング制約)から未確定(出典: `PLAN_bevformer_ppa_exploration.md` §0, §6, §7)。

**含意**: 「72 ACEが必須」という技術的結論の実効性は、A_maxが確定するまで保留状態にある。A_maxが380mm²未満であれば、モデル側の軽量化(アナログ側処理時間を約24%短縮して48 ACEを救済する)かA_max自体の見直しのいずれかが必須になる(`PLAN_bevformer_ppa_exploration.md` §4結論)。この軽量化の具体的な着手方法は§4で述べる。

### 3-10. 電力のコンポーネント別内訳(新規実測)——電力側の律速もSRAM-bound/ACE-boundの分岐を引き継ぐ

§3-1でレイテンシの律速項がモデルごとにACE-bound/SRAM-boundに分岐することを見た。**同じ分岐が電力側にも表れるか**を、実際に`power_estimator.py`を内部関数レベルで叩いて確認した。

**手法**: `power_estimator.py`の`OpEnergy`は1演算のエネルギーを6成分(`ace_active`/`ace_sleep`/`sram`/`accessor`/`control`/`noc`。出典: [02_ppa_estimation.md](02_ppa_estimation.md) §4.1)に分解して内部で保持しているが、CLIの出力経路(`calc_power()`, [pow:544-602])は演算タイプ(ACE/COPY/SIMD/PAD/INFEED/OUTFEED)ごとの`.total`(J/W)と、最終的な Functional Unit / Interconnect の2値しか表に出さず、6成分の内訳そのものは一切表示されない。そこで`M2000_Power`の`calc_energy_ace()`/`calc_energy_copy()`/`calc_energy_simd()`/`calc_energy_pad()`/`calc_energy_infeed()`/`calc_energy_outfeed()`/`calc_energy_interconnect()`を`calc_power()`を経由せず直接呼び出し、演算タイプ×6成分のクロス集計を取得した(スクリプト: `tools/power_breakdown/power_breakdown.py`)。既存の`_ppa_*.tar.gz`から`final.l0.pb`・`packet_log.json`・`event_log.json`(いずれも`mythic-ppa-estimators --estimate-power`実行時に`artifacts/`配下へ既に保存済み)を取り出すだけで済み、**funcsim・コンパイラいずれの再実行も不要**(コンパイル済みプロトバフの静的パースのみで、1モデルあたり1秒未満)という点は[02_ppa_estimation.md](02_ppa_estimation.md) §3.9(4)の`perf_breakdown.sh`と同じ考え方である。

**Accessor/Control/NOCの定義(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §4.1-4.3, §4.6-4.7)**:

- **Accessor**: データそのもの(SRAM成分)ではなく、データ転送を発行・管理する周辺ロジックのコスト。136バイトの転送ディスクリプタ処理+バイト数比例の処理コストからなる(`calc_accessor_energy`, [pow:181-191])。
- **Control**: 演算の順序制御・同期のコスト。データフロー実行モデルの依存関係通知トークン(8バイト×read-modify-write)の更新と、反復カーネルの進行を追跡するオペレーションカウンタ(96/24バイト)の読み書きからなる(`calc_operation_energy`, [pow:170-178])。この成分の`n_iterations`は`get_num_op_control_iterations = ceil(iter/4)`という**コード自身が「コンパイラに未実装」と明記する暫定式**([pow:161])で計算されており、他成分より不確実性が高い。
- **NOC**: **名前が同じ2つの別物が存在する点に注意。** `OpEnergy`の6成分の1つとしての`noc`(演算内NOC)は、ACE/COPY/SIMD/PAD/INFEED/OUTFEEDの全`calc_energy_*`関数で**常に0**(`# TODO - ADD NOC ENERGY`という未実装コメントが残っている、[pow:343]等)。以下の表で報告する「Interconnect(NOC)」は、これとは全く別の`calc_energy_interconnect()`([pow:441-482])が、演算ではなくpacket log(タイル間通信の実行トレースJSON)のバイト転送量から独立に計算する値であり、packet log未提供時は無条件に0を返す。

**副次的に確認したバグ**: `calc_energy_interconnect()`([pow:441-482])は`packet_log_path`を渡さない(`self.packet_log=None`のまま)と`for key, value in self.packet_log.items()`で`AttributeError: 'NoneType' object has no attribute 'items'`を投げてクラッシュする実行時エラーを確認した。[02_ppa_estimation.md](02_ppa_estimation.md) §4.7で「推測」としていた`hasattr(self,"packet_log")`が常に`True`になる(dataclassフィールドのため未提供時もNoneとして存在する)という不具合は、憶測ではなく実際に起きる不具合であることが確定した。回避策は`packet_log_path`/`event_log_path`を必ず渡すことで、`mythic-ppa-estimators`の通常実行では両ファイルとも`artifacts/ppa/`に既に生成されているため実務上の支障はない。

**BEVFormer-Tiny(m2072, 72 ACE, 30fps)の内訳**(出典tar.gz: `bevformer_m2072_high_2605_2_ppa_2026_07_27_12_43_57.tar.gz`。合計3.2873Wは`PLAN_bevformer_ppa_exploration.md`記載の公表値analog 3.287Wと一致し、手法の正当性を確認済み):

| 演算タイプ | ACE active | ACE sleep | SRAM | Accessor | Control | NOC | Total |
|---|---|---|---|---|---|---|---|
| ACE(mma_dot) | 1.52226 W | 0.08371 W | 0.09829 W | 0.18604 W | 0.03293 W | 0 | 1.92323 W |
| COPY | 0 | 0 | 0.20310 W | 0.27261 W | 0.02281 W | 0 | 0.49852 W |
| INFEED/OUTFEED/SIMD/PAD | 0 | 0 | 0.00051 W | 0.00064 W | 0.00006 W | 0 | 0.00122 W |
| **合計(Functional Unit)** | **1.52226** | **0.08371** | **0.30191** | **0.45929** | **0.05580** | 0 | **2.42297 W** |
| Interconnect(NOC、packet log由来) | — | — | — | — | — | 0.8643 W | 0.8643 W |
| **総電力** | | | | | | | **3.2873 W** |

比率で見ると: ACE(active+sleep)48.9%、Interconnect(NOC)**26.3%**、Accessor 14.0%、SRAM 9.2%、Control 1.7%。**Interconnect単体がSRAM+Accessor+Control(24.9%)を上回り、ACEに次ぐ第2位の電力消費源になっている**。また非ACE演算である**COPY**単体(0.499W)は、ACE演算に伴う非ACE(digital)成分の合計(SRAM+Accessor+Control+ACE sleep=0.402W)より大きく、クロスバー計算そのものではなくデータの複写・整形が無視できない電力を占めることを示している。

**YOLOPX(48 ACE / 72 ACE, 30fps)との対比**(出典tar.gz: `yolopx_m2048_high_..._2026_07_31_08_11_04.tar.gz` / `yolopx_m2072_high_..._2026_07_31_10_07_06.tar.gz`。合計0.7404W/0.8270Wは`PLAN_yolopx_ppa_exploration.md` §4.1-4.2記載の公表値0.740W/0.827Wと一致):

| SKU | ACE% | SRAM% | Accessor% | Control% | Interconnect(NOC)% |
|---|---|---|---|---|---|
| YOLOPX m2048(ACE-bound) | **61.7%** | 6.2% | 10.4% | 1.6% | 20.1% |
| YOLOPX m2072(ACE-bound) | **62.9%** | 6.0% | 9.9% | 1.5% | 19.7% |
| BEVFormer m2072(SRAM-bound) | **48.9%** | **9.2%** | **14.0%** | 1.7% | **26.3%** |

**含意**: §3-1で確認したレイテンシ側の律速の分岐(YOLOPX=ACE-bound、BEVFormer=SRAM-bound)は、電力側にも同じ方向の非対称として表れる。ACE-boundなYOLOPXは電力の6割超がACEそのものに集中する一方、SRAM-boundなBEVFormerはACE比率が5割を切り、SRAM・Accessor・Controlという「データ移動+制御」側3成分の合計(24.9%)がYOLOPXの2モデル(18.2%/17.4%)より明確に大きい。Interconnect(NOC)まで含めれば差はさらに拡大する(非ACE成分の合計はBEVFormer 51.1% vs YOLOPX 38.3%/37.1%)。レイテンシ側の律速判定(`perf_breakdown.sh`、§4.1)は、電力側の主要な改善対象(ACE電流そのものか、データ移動経路か)を判定する上でもそのまま使える指標になっている。

#### 3-10.1 §3-2の補足: num_aces増加によるpower悪化の主因はSRAM複製ではなくACEスリープ電力(実測で判明)

§3-2はYOLOPXの48→72 ACE実測から「重み複製→SRAMトラフィック増→power増」という機構を示した。今回の内訳分解で、この機構がpower増加(+11.6%、0.743→0.829W)にどの程度寄与しているかを定量化できる。

YOLOPX 48→72での演算タイプ別差分:

| 成分 | 48 ACE | 72 ACE | 差分 | 全体差分(+0.0724W)に対する比率 |
|---|---|---|---|---|
| ACE active | 0.35031 W | 0.35031 W | **±0(完全一致)** | 0% |
| ACE sleep | 0.10626 W | 0.16988 W | **+0.0636 W** | **87.8%** |
| SRAM(ACE+COPY等合計) | 0.04595 W | 0.05001 W | +0.0041 W | 5.6% |
| Accessor(同上) | 0.07708 W | 0.08155 W | +0.0045 W | 6.2% |
| Control(同上) | 0.01184 W | 0.01204 W | +0.0002 W | 0.3% |
| **Functional Unit合計** | 0.5914 W | 0.6638 W | **+0.0724 W** | 100% |

**ACE activeエネルギーは48→72 ACEで完全に一致する**(総ACE演算回数1,649,376・総MAC数285,021,216,768が両SKUで一致するため、1回あたりの活性化エネルギーも変わらない)。増加分の**88%はACE sleepが占め、SRAM/Accessor/Controlの増加(§3-2が説明した重み複製由来のSRAMトラフィック増)は合計で12%にとどまる**。

メカニズム: `calc_energy_ace()`内の`needed_sleep_time = num_aces / inf_rate - total_time`([pow:346])で、`total_time`(実際にACE演算に使われた時間)はACE演算回数×160ns×`n_iterations`で決まりnum_acesに依存しない。一方`num_aces / inf_rate`(1推論周期あたりの「全ACEの延べ利用可能時間」)はnum_acesに比例して増える。したがって**num_acesを増やすほど「稼働していないACEを維持する時間」がほぼ線形に増加し、そのアイドル時間をsnooze/sleep電流(§2の`i_adc_core_inactive`等)で満たす分がFunctional Unit電力増加の大部分を占める**。これは§3-2が説明した重み複製によるSRAM/Accessorトラフィック増とは独立した、第2の(かつYOLOPXの実測では支配的な)power増加メカニズムである。

**含意**: 「num_aces増設→area↔SRAM↔powerが結合する」という§3-2の結論自体は覆らないが、**SKU選定時にpowerコストを見積もる際は、SRAM/Accessorトラフィックよりもむしろ「使われないACEの待機電力」の方が支配的な項になりうる**点を踏まえる必要がある。この待機電力は演算タイプ別に見ればモデル構造に依存しない(`i_adc_core_inactive`等はモデル非依存の物理定数、[02_ppa_estimation.md](02_ppa_estimation.md) §4.4)ため、「num_acesを1増やすごとに一定のアイドルタックスが乗る」という関係は他モデルにも一般化できると推測される**[推測]**(実測はYOLOPX 48→72の1点のみ)。BEVFormer側は48 ACEが不可行(§3-9)のため、同型の48→72比較は取得できていない。

### 3-11. 実ハイブリッド構成のデジタル側ボトルネックは、全デジタル評価(§3-4)の結論と異なる(新規実測)

§3-4は「デジタル実行はメモリ帯域が支配的」と結論したが、これはResNet-50バックボーンを含む**全デジタル仮想シナリオ**からの結論である。実際のハイブリッド構成(BEVFormer m2072)でデジタル実行されるのはTransformer部分のみ(MACs 16.5 bn、モデルサイズ13.6 MB——全デジタル評価の54.7分の1の規模)であり、このボトルネックが同じかどうかを既存コンテナでの`vnnmap`直接実行(ブラックボックス、逆アセンブル不使用)で確認した。詳細な手法・生データは[06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md) §2参照。

**レイテンシ側**: サイクル内訳はMAC 71.1%・non-MAC 17.2%・exposed DMA 11.7%(既定cfg実測)。**exposed DMAは支配的ではない**(全デジタル評価の40.8%と対照的)。支配的なのは「MAC cyclesが71.1%を占めながらMAC利用率が9.69%しかない」という組み合わせで、Attention/GridSample/Reshapeのような計算密度の低い演算が、実際の仕事量に対して不釣り合いに多くのサイクルを消費していると考えられる(§5.1のMAC利用率9.69%という既知の事実と同一の現象)。

**電力側**: `pow*Pj`係数を1グループずつゼロにする分離実測(手法は§5.1で述れた「1係数だけを変えて差分を見る」)により、DDRの寄与は18.4%に過ぎず**支配的ではない**ことを確認した(全デジタル評価ではOCRAM拡張で28.6%の電力削減が確認されている、§3-4)。支配的なのはDMEM/IMEM(ローカルメモリアクセス、34.7%)とNon-MAC unit(28.3%)で、合わせて63.0%を占める。

**含意**: §3-4・§4.2で「デジタル側の改善にはメモリ帯域(DDR/OCRAM)対策が効く」という結論を、実ハイブリッド構成のデジタル部分(4.63ms/1.245 W、§3-3)にそのまま適用してはならない。この部分の改善に有効なのは、§5.1で述べたAttention系演算の計算密度改善(計算密度の低い演算の発行回数・ローカルメモリアクセス回数を減らす構造変更)であり、OCRAM/DDR容量の調整ではない。全デジタル評価(§3-4)と実ハイブリッドのデジタル部分(本節)で、同じ「デジタル実行」という括りの中でもボトルネックの性質が完全に異なる、という点が本節の主要な発見である。

---

## 4. 律速要因別に見たモデル構造の指針

質問「レイテンシ改善にはSRAMアクセスを減らすことが必要だと思うが、適したモデル構造は何か」への回答。

律速項がSRAM-bound(BEVFormer)かACE-bound(YOLOPX)かでモデルが2分される以上(§3-1)、「SRAMアクセスを減らすモデル構造は何か」だけでは片方のケースにしか答えられない。**ACE-boundなモデルには、対になる問い「ACE演算数(クロスバー利用率)を減らすモデル構造は何か」が別途必要**になる。以下、まず両ケースに共通する前提(4.1)、次にSRAM-bound向け(4.2)、ACE-bound向け(4.3)の指針を分けて述べる。

### 4.1 まず律速判定を行う(構造変更より先にすべきこと)

どちらの指針が有効かは、そのモデルの律速項がSRAMかACEかで決まる(§3-1)。SRAM-boundなモデルに対してACE演算削減を目的に構造を変えてもSRAM項が頭打ちのままで改善せず、逆にACE-boundなモデルにSRAM削減を適用してもACEクリティカルパスで頭打ちのままで改善しない。既存の`_ppa_*.tar.gz`から`tools/perf_breakdown/perf_breakdown.sh`で約9秒で判定できる(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.9(4))ため、モデル構造の変更に着手する前に必ず行うべき最初のステップである。

### 4.2 SRAM-boundなモデル向け——SRAMアクセスを減らす構造

SRAM負荷の主因は「並列化のための重み複製」である。§3-2で示した通り、SRAMトラフィックの増加はACE数を増やした際の重み複製に強く結びついている(YOLOPX 48→72で重み+18.0%、SRAM読+5.3%/書+7.3%)。したがって「SRAM削減に適したモデル構造」は、言い換えれば**「同じACE数でも重み複製が少なくて済む構造、またはSRAM往復あたりの計算密度が高い構造」**である。

具体的な指針(既存実測からの含意):

- **重み再利用度の高い畳み込みはSRAM往復あたりの計算密度が高く、アナログMMAに適する。** BEVFormerのResNet-50バックボーン(出典: [06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md) §3.2、ONNXグラフのノード命名から確認)はMAC比率98.3%を占めながらアナログ側で69.4%という比較的高いACE利用率を達成している(出典: [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)、`PLAN_bevformer_ppa_exploration.md` §2)。1回の重みロードで多数の出力位置を生成する構造(通常の畳み込み)はSRAM再アクセスの相対コストが低い。

- **Depthwise Convは強制的にデジタル(SALU)に落ちる。** `MarkDepthwiseConvsAsDigital`が`group == out_channels`かつ`in_channels/group == 1`の条件を満たすConvに`__digital_onchip`属性を付与し、アナログMMAではなくSALU(デジタル)で処理される(出典: [to_structural.md](conversion_steps/to_structural.md) §8.1)。省パラメータ設計としてよく使われるDepthwise Convは、この意味でオンチップのアナログ処理密度を下げる方向に働く。on-chipでの計算密度を優先するなら、depthwise比率を絞るか、デジタル側で処理される前提でレイテンシ予算を確保する必要がある。

- **多視点・多カメラ入力は特徴マップサイズに比例してSRAM負荷を増やす。** BEVFormer(6カメラ、SRAM-bound、26.95ms)とYOLOPX(単一カメラ、ACE-bound、6.83ms)の対比がこれを裏付ける(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.9)。カメラ数・解像度・特徴マップ解像度の増加は、SRAM-bound化のリスクを直接高める。

  **メカニズム[推測]**(既存の式・実測値の組み合わせによる再構成。単一箇所の直接引用ではない): ACEクリティカルパスとSRAM時間は感度が非対称である。`total_ace_duration_ns = len(timesteps) × 160ns`(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.4)は演算回数のみに依存し、クロスバー1回あたりに実際に流れるバイト量には反応しない(§4.3で見た充填率の低さがそのまま許容されるのはこのため)。一方SRAM時間は最も忙しいタイルの累積バイト/アクセス数を帯域で割った値であり(同§3.4)、実際に読み書きされたデータ量に直接比例する。

  M2000はアナログ重みをクロスバーに常駐させる(weight-stationary)構成のため、推論中にSRAMへ読み書きされるのは主に活性化(入出力の特徴マップ)である。多カメラ入力は同一の重み(共有バックボーン)を用いた推論をカメラ視点数だけ繰り返す構造になるので、SRAM側の累積バイト量は「カメラ数 × 1視点あたりの特徴マップの活性化バイト量」という積で増える。ACE側の演算回数は空間位置と繰り返し数に応じて増えるものの、1回あたりに動くバイト量そのものには鈍感なため、同じ「カメラ数×特徴マップサイズ」の増加に対してSRAM時間の方が先に押し上げられやすい。BEVFormerがYOLOPXよりクロスバー充填率が低い(32.6% vs 49.6%、後述§4.3)という実測は、この非対称性の帰結として整合的に説明できる。

- **Attention/Transformer部分はoff-chip一括処理になりやすい。** BEVFormerの`to_structural`実装はtransformer側サブグラフの全ノードを一括off-chip化する(`everything_off_chip`、出典: [to_structural.md](conversion_steps/to_structural.md) §7.1)。コンパイラ側には`AttentionDetr`等、Attentionを`group=8`のグループ畳み込みへ変換する最適化パスが存在するが(出典: [01_compilation.md](01_compilation.md) §3.1.2(G))、これは計算のマッピング先(アナログMMAへ載せられる表現への変換)を変えるものであり、SRAM往復の絶対量そのものを減らす保証はない**[推測]**——この変換パスがSRAMトラフィックに与える効果は本調査群では実測していない。

- **重み容量の事前計算がSKU選定を高速化する。** YOLOPXのm2024失敗は、コンパイル前に`BCMConv2d`の重み総量を集計するだけで(生重み32.45M vs 容量29.88M=108.6%)、90分超のコンパイル試行なしに予見できた(出典: `PLAN_yolopx_ppa_exploration.md` §3.3)。モデル構造を変更する際は、このパディング込みの重み容量比を先に計算し、目標SKUの容量に収まるかを確認することで、無駄なコンパイル試行を避けられる。

### 4.3 ACE-boundなモデル向け——ACE演算数(クロスバー利用率)を減らす構造

ACEクリティカルパスは`len(timesteps) × 160ns`で決まり(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.4)、`timesteps`はACEがドット積を発行した回数に対応する。したがってACE-boundなモデルのレイテンシを縮めるには、**総MAC数そのものではなく「ACE演算1回あたりに何MAC分の仕事をさせられているか」(クロスバー利用率)**を上げるか、ACE演算の発行回数自体を減らす構造が必要になる。1 ACEクロスバーは最大1280入力×272出力=348,160 MACを1回のドット積(160ns)で処理できる(出典: [00_overview.md](00_overview.md) ハードウェア定数表)。

既存2モデルの実測からこの利用率を比較できる(ACE MACs / ACE Operations、出典: `PLAN_bevformer_ppa_exploration.md` §2、`PLAN_yolopx_ppa_exploration.md` §2):

| モデル | ACE MACs | ACE Operations(=timesteps相当) | 平均MAC/回 | クロスバー充填率 |
|---|---|---|---|---|
| BEVFormer-Tiny(m2072) | 955,023,360,000 | 8,421,600 | 113,402 | 32.6% |
| YOLOPX(m2048) | 285,021,216,768 | 1,649,376 | 172,805 | **49.6%** |

BEVFormer側の充填率がYOLOPXより低いことは、SRAM-boundであるためACE増設のメリットを引き出せていない(§3-1)ことと整合する——ACE側に既に遊びがあるにもかかわらず律速していない。ACE-bound側(YOLOPX)でこの充填率をさらに上げられれば、`timesteps`数(=ACEクリティカルパス)を直接減らせる余地がある。

具体的な指針(既存実測・SDK仕様からの含意):

- **1回のドット積で1280入力×272出力を余さず使う層形状を優先する。** チャネル数がACEクロスバーの入出力次元(1280/272)に対して極端に少ない層(例: 初期層の低チャネルConv、depthwise系の1入力チャネル畳み込み)は、クロスバーの大部分が空いたままACE演算1回を消費するため充填率を下げる**[推測]**。チャネル数をクロスバー次元の倍数に近づける、または層を融合してより大きな行列積にまとめる設計が、同じMAC数でもACE演算回数(=timesteps)を減らす方向に働くと推測される。

  **ONNXグラフの実解析による補足**(出典: [06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md) §3、逆アセンブル不使用): 「in_C<1280かつout_C<272」の小さすぎる層の比率は、YOLOPX(E-ELAN backbone、後述)89.5–90.3%に対しBEVFormer(ResNet-50 backbone)55.4–58.2%で、**小チャネル層の比率自体は直感(充填率が低いYOLOPXの方が高いはず)と逆**である。一方「out_C>272」(列方向の分割が必要になる層)の比率はBEVFormer 40.0%・YOLOPX 9.9%で、BEVFormerのResNet-50 BottleneckがもつC=1024/2048等(いずれも272の倍数ではない)のような大チャネル1x1 convが、列分割時の余りブロックによる不完全充填を通じて充填率を下げている可能性が高いと考えられる**[推測、部分的な説明にとどまる]**。ONNX計算のMAC総量が既知の実測値(ACE MACs)より23〜40%小さく一致しないという未解決の不整合があるため、この説明は確証ではない。詳細は[06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md) §3.5-3.6参照。

- **クロスバー充填率はレイテンシだけでなくACE active電力にも直接効く。** `ace_op_energy_time`([pow:198-252]、[02_ppa_estimation.md](02_ppa_estimation.md) §4.4)によれば、ACE active電力は「1回の演算ごとに必ず発生する固定項(グローバルバイアス電流`i_adc_global`/`i_aidac_global_*`)」+「使用レーン数(`n_inputs_active`/`n_outputs_active`)に比例する可変項」の合計であり、ADC変換時間は充填率に関わらず常に160nsで一定である。したがって充填率が低いほど、固定項のコストを少数のMACにしか償却できず、MACあたりのエネルギーが割高になる。式の定数から試算すると、BEVFormerの平均充填率(32.6%)では、クロスバーを最大まで満たした場合と比べて**MACあたり約1.8倍のエネルギーを要する計算になる**([推測・簡易試算]、入出力レーンが充填率に応じて均等に使われると仮定した近似)。**ACE active電力(BEVFormer 46.3%・YOLOPX 61.7%前後、§3-10)は「本当に必要な計算そのもの」に近いコストであり、num_aces(SKU選択)では変えられない**——§3-10.1が示す通り48→72 ACEでACE active energyは完全に一致し、総ACE演算回数・総MAC数がSKUに依存しないためである。ACE active電力を削減する手段は、モデルの総MAC数を減らすこと(精度とのトレードオフ)と、この充填率を上げることの2つに限られ、それ以外(160nsのADC変換時間、グローバルバイアス電流)はプロセスノード・回路設計に紐づく固定値でSDKから変更できない。

- **タイル分割・パーティションの粒度がACE演算回数に影響する。** `auto_partition`(コンパイラ内部)はSRAMサイズ制約に応じて演算を複数パーティションに分割する(出典: [01_compilation.md](01_compilation.md) §3.3.1、[00_overview.md](00_overview.md) §3.5)。1層が過剰に分割されるほどACE演算(ドット積発行)の回数が増え、1回あたりの充填率が下がりうる**[推測]**。層の重み・活性化サイズをタイル容量(1タイル=4,980,736重み)に対して過度に大きくしない設計が、不要な分割を避ける方向に働く。

- **総MAC数の削減は充填率が保たれる場合にのみACEクリティカルパス短縮に直結する。** ACE-boundなモデルでMAC数を減らしても、1回あたりの充填率が同時に下がれば`timesteps`数(=ACE演算回数)は減らない。したがって層のチャネル数・カーネル形状を削減する際は、削減後もクロスバー充填率が維持されるか(充填率がすでに低い層を優先して削るか、単純に全層を均等に縮小するか)を区別する必要がある**[推測]**。本調査群ではMAC数削減が充填率に与える影響そのものは実測していない。

- **SRAM-boundなモデルへの適用時は優先度が下がる。** BEVFormerのように既にSRAM側が律速している場合、ACE充填率を上げてもレイテンシは改善しない(§3-1)。この指針はACE-boundなモデル、またはSKU変更(ACE増設)でSRAM-boundからACE-boundに転換する場合に限って有効になる。

- **参考: 両モデルのbackboneアーキテクチャの実体(ONNXグラフから確認)。** BEVFormer-Tinyのbackboneは**ResNet-50**(Bottleneckブロックのノード命名から確認)、YOLOPXのbackboneは**YOLOv7系のE-ELAN(Extended-ELAN)**+**YOLOX式decoupled anchor-freeヘッド**である(モデル名「YOLOP+X」はこの構成をそのまま反映している)。両モデルともdepthwise/grouped convは未使用(全Conv層でgroup=1)。出典・詳細な層別チャネル分布は[06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md) §3.2-3.3。

---

## 5. LLM/VLA等、将来の多層Transformerモデルへの課題

質問「デジタルで処理しなければならないTransformerブロックの計算量も削減したほうがいいのか」への回答と、それに付随する広い課題整理。

### 5.1 「デジタル計算量(MAC数)を減らすべきか」への直接回答

**MAC数の削減だけでは効果が薄いと推測される。** デジタル側はすでにメモリ帯域ボトルネックであり、MACアレイを増やすとむしろ悪化する逆説的な結果が実測されている(`nMPs` 288→2304でfps 16.99→4.02、Power@30fps 25.58→37.27W、§3-4)。BEVFormerのTransformer部分単体を全デジタル実行した場合、MAC利用率はわずか**9.69%**であり(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §4.1)、Attention/Reshape/GridSampleといった計算密度の低い演算がMACアレイを埋められないことがボトルネックの主因になっている。backboneのConvがMAC利用率を41.67%まで押し上げているのとは対照的である。

したがって、削減すべき対象は**MAC数そのものではなく、メモリトラフィック(活性化・中間テンソル・将来的にはKVキャッシュの読み書き)**である。具体的には:

- MAC数を保ったまま計算密度の高い演算(Conv、行列積)の比率を上げ、計算密度の低い演算(Reshape/GridSample/Gather等のメモリバウンドな整形操作)を減らす方向の構造変更がPPA改善に直結すると推測される**[推測]**。
- MAC数だけを削減する軽量化(例: 層数を減らす、隠れ次元を減らす)は、メモリバウンドな区間がそのまま残る場合、レイテンシ改善に寄与しない可能性がある**[推測]**。

### 5.2 LLM/VLAで新たに立つと予想される課題(未検証事項を含む)

既存2モデル(BEVFormer, YOLOPX)はいずれも畳み込み中心のバックボーンを持ち、Transformer部分は比較的小規模(BEVFormerのdigital latencyは4.63msで全体の14.5%)である。LLM/VLAのような多層Transformerデコーダを主体とするモデルでは、以下の課題が新たに、あるいはより強く立つと推測される:

1. **KVキャッシュ増大によるSRAM/DDRトラフィック増加。** BEVFormerが6カメラ入力でSRAM-bound化した構造(§3-1, §4.2)と同型のリスクが、シーケンス長の増大でも起こりうる**[推測]**。層数・シーケンス長が増えるほど、デジタル側のメモリトラフィックがボトルネックになる可能性が高い。

2. **動的シーケンス長・自己回帰デコードとの整合。** `vnnmap`はdynamic batch非対応で、batch=-1を1に固定する(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §8)。pythia(GPT-NeoX系、SDK内で唯一のLLM相当モデル)の`to_structural`実装は、`use_kv_cache=True`の場合に`fix_sequence_length`/`simplify_inputs`をスキップする分岐を既に持っており(出典: [to_structural.md](conversion_steps/to_structural.md) §7.3)、SDK側にKVキャッシュ運用を想定した仕組みは存在する。しかし自己回帰デコード時にシーケンス長が変化する場合の静的shape要求との整合性、およびそのPPAへの影響は本調査群では未検証。

3. **既存コンパイラ変換パスの層数スケーラビリティは未実測。** Gemm/MatMul→Conv統一、Attention→`group=8`グループConv、`vidRope`によるRoPE表現、`vidMultiQueryExpand`によるGQA/MQA KV展開(出典: [01_compilation.md](01_compilation.md) §3.1.2(A)(G), §3.1.3)は、コード上は層数に依存しない汎用的な書き換え規則として実装されているが、層数が増えた場合のコンパイル時間・CP-SAT収束性(§3-8で見た通り既存2モデルでも90〜120分・タイムアウトが発生している)がどう変化するかは未検証。層数が数十〜数百に達するLLMでは、コンパイル時間そのものがボトルネックになる可能性がある**[推測]**。

4. **GQA/MQA・RoPEを採用するモデル設計はコンパイラの最適化パスと整合している。** `vidRope`(LLaMA系RoPE)と`vidMultiQueryExpand`(GQA/MQAのKV展開)という専用の書き換えパスが既に存在する(出典: [01_compilation.md](01_compilation.md) §3.1.2(D)(C))ことから、これらの技術を採用するLLM設計は、SDKが既にアナログMMA向け表現への変換手段を持つという意味で相性が良い。逆に、これらのパスが対応しないAttention変種を使う場合、Transformer部分が丸ごとoff-chip(標準ONNX opのまま量子化・ノイズなしで実行)に落ちる可能性がある([03_accuracy_simulation.md](03_accuracy_simulation.md) §8.2の「Attention分解/LayerNormに対応するグラフ書き換えは精度シミュレーション側に存在しない」という記述と符合する)。

5. **VLA固有の非対称拡大リスク。** VLA(視覚+言語+行動)は畳み込みバックボーン(アナログ向き)とLLMデコーダ(デジタル・シーケンシャル向き)の結合になりやすい。BEVFormerで確認された「アナログMAC比率98.3%だが処理時間比率ではデジタルが14.5%」という非対称(出典: [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md))が、デコーダの層数増加に伴ってさらに極端化する可能性が高い**[推測]**——デコーダの計算量がバックボーンに対して相対的に小さくても、処理時間支配率は計算量比率よりずっと高くなりうる。

### 5.3 本ドキュメント作成時点で未検証の事項(次の実測候補)

- pythia(GPT-NeoX系、LLM相当)のPPA実測は、BEVFormer/YOLOPXと同形式のSKU探索(`PLAN_*`)としては未実施。
- KVキャッシュを使った自己回帰デコードの`funcsim`/`vnnmap`実測は未実施。
- 層数を実際に増やした場合のコンパイラ変換パスのコンパイル時間・CP-SAT収束性は未検証。
- Attention部分をoff-chip一括処理からアナログMMAへの変換に載せ替えた場合のSRAMトラフィック変化(§4.2で[推測]とした点)の実測。
- ACE-boundなモデルにおける層形状・タイル分割とクロスバー充填率の関係(§4.3で[推測]とした点)の実測。

---

## 6. モデル別ボトルネック総括(BEVFormer/YOLOPX)

本節は§3-§5の実測を、モデル×領域(レイテンシ/電力)×実行系統(アナログ/デジタル)で整理し、ボトルネックの構造的理由と改善指針を集約する。デジタル側の内訳は[06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md)の新規実測(逆アセンブル不使用)による。

### 6.1 レイテンシ

**BEVFormer-Tiny(m2072, 72 ACE)**

| 系統 | 実測値 | 内訳 | ボトルネック | 構造的理由 |
|---|---|---|---|---|
| アナログ | 26.95 ms(総latencyの支配項) | ACEクリティカルパス23.37 ms / SRAM時間26.95 ms(max、§3-1) | **SRAM-bound** | 6カメラ入力によるSRAM累積負荷(§4.2)。ResNet-50 backboneのクロスバー充填率32.6%(§4.3)。out_C>272の層が40%を占め列分割の余りロスが生じている可能性(06§3.6、**[推測]・部分説明**) |
| デジタル(Transformer部分) | 4.63 ms(固定フロア、§3-3) | MAC cycles 71.1% / non-MAC 17.2% / exposed DMA 11.7%(§3-11、06§2.2) | **MAC利用率9.69%という低効率**(exposed DMAではない) | Attention/Deformable Attention/GridSample/Reshapeの低計算密度演算——softmaxは非MAC、GridSampleはgather中心、reshapeはMACゼロ、少数サンプリング点の行列積は64 MAC/cycle/MPの並列レーンを満たせない(§5.1) |
| **合計** | **31.58 ms**(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §4.2) | — | — | デジタル比率14.7%(4.63/31.58) |

**YOLOPX(m2048, 48 ACE)**

| 系統 | 実測値 | 内訳 | ボトルネック | 構造的理由 |
|---|---|---|---|---|
| アナログ | 7.56 ms(§3-1) | ACEクリティカルパス7.56 ms / SRAM時間6.83 ms(max) | **ACE-bound** | 単眼カメラ入力。E-ELAN backboneのクロスバー充填率49.6%(§4.3)。out_C>272の層は9.9%のみで列分割ロスが少ないと推測される(06§3.6、**[推測]**) |
| デジタル | 未確定。既存の予備値ではmacs_bn=0.089・544.84 fps相当(latency換算約1.84 ms、出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §9.2項目3) | 未取得。BEVFormerのようなサイクル内訳・電力分離実測は本調査群では未実施 | 未確認**[推測、次の調査候補]** | YOLOPXのdecoupled head(cls/reg/obj_preds)がoff-chip処理される可能性が高いが、上記予備値が実際の出荷ハイブリッド構成のデジタル部分を代表するかは未検証 |

### 6.2 消費電力

**BEVFormer-Tiny(m2072, 72 ACE)**

| 系統 | 実測値 | 内訳 | ボトルネック | 構造的理由 |
|---|---|---|---|---|
| アナログ | 3.2873 W(§3-10) | ACE(active+sleep)48.9% / Interconnect(NOC)26.3% / Accessor14.0% / SRAM9.2% / Control1.7% | **データ移動系(SRAM+Accessor+NOC+Control=51.1%)がACE単体(48.9%)を上回る** | SRAM-bound構造(§3-1)の帰結。重み複製によるSRAM/Accessor増(§3-2)。72 ACE固定のため待機電力を絞る余地がない(48 ACEが不可行、§3-9) |
| デジタル(Transformer部分) | 1.245 W(Power@30fps、§3-11、06§2.3) | DMEM/IMEM 34.7% / Non-MAC unit 28.3% / DDR 18.4% / MAC unit 12.7% / OCRAM 5.9% / Bus・NoC ~0% | **DMEM/IMEM+Non-MAC=63.0%**(DDRアクセスではない) | レイテンシ側と同一の構造的原因(Attention系演算の低計算密度) |
| **合計** | **4.505 W**(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §4.2) | — | — | デジタル比率27.6%(1.245/4.505) |

**YOLOPX(m2048, 48 ACE)**

| 系統 | 実測値 | 内訳 | ボトルネック | 構造的理由 |
|---|---|---|---|---|
| アナログ | 0.7404 W(§3-10、PLAN文書記載値0.740Wと同一) | ACE61.7% / Interconnect(NOC)20.1% / Accessor10.4% / SRAM6.2% / Control1.6% | **ACE単体が6割超**(データ移動系はBEVFormerの約半分) | ACE-bound構造(§3-1)の帰結。48→72 ACEでの電力増(+11.6%)はACE sleepが88%を占め、重み複製由来のSRAM/Accessor増は12%のみ(§3-10.1) |
| デジタル | 2.15 mW相当(§9.2予備値、出典同上)——アナログ側の0.3%未満 | 未取得 | 無視できる規模**[推測]** | YOLOPXの総電力はアナログ側が実質支配的で、デジタル側の構造改善は優先度が低いと推測される |
| **合計** | **0.7404 W** | — | — | デジタル寄与は上記予備値が正しければ無視できる規模 |

### 6.3 モデル構造的な改善指針(総括)

| モデル | 領域 | 改善指針 | 出典 |
|---|---|---|---|
| BEVFormer(SRAM-bound) | アナログ・レイテンシ/電力 | 重み再利用度の高い畳み込みを優先、depthwise比率抑制、多視点入力の特徴マップサイズを抑える、Attention/Transformer部分のSRAM往復量削減(grouped conv変換の効果は未実測) | §4.2 |
| BEVFormer | デジタル・レイテンシ/電力 | Attention/Deformable Attention/GridSample/Reshapeの計算密度改善(MACアレイに適した表現への変換、または発行回数の削減)。OCRAM/DDR容量調整は効果が薄い | §3-11, §5.1 |
| BEVFormer | 面積・SKU | 48 ACEを可行にするモデル軽量化(アナログ処理時間を約24%短縮)で、72 ACE固定によるACE sleep待機電力コストを削減できる可能性 | §3-9, §3-10.1 |
| YOLOPX(ACE-bound) | アナログ・レイテンシ | クロスバー充填率の維持・向上(1280×272に適合するチャネル形状、過剰なタイル分割の回避)。MAC数削減は充填率維持が前提 | §4.3 |
| YOLOPX | アナログ・電力 | SKU選定時はSRAM/Accessorトラフィックよりも「ACE待機電力」を優先して評価する | §3-10.1 |
| YOLOPX | デジタル | 総電力の0.3%未満と推定され、優先度は低い**[推測]** | §9.2予備値 |

**含意**: BEVFormerとYOLOPXは、レイテンシ・電力の両軸で「データ移動系(SRAM/Accessor/Interconnect/DMEM/IMEM)が支配的か、演算系(ACE/MAC)が支配的か」という同一の非対称性を、アナログ・デジタルの両方の実行系統で共有している。BEVFormer(SRAM-bound)はアナログでもデジタルでもデータ移動・低密度演算が支配的コストであり、改善指針も両系統で「データ移動量・低密度演算の削減」に一貫して収束する。YOLOPX(ACE-bound)はアナログ側がACEそのものに支配され、デジタル側は総電力に対してほぼ無視できる規模にとどまる。**この対称性は、律速要因の判定(§4.1)が電力側だけでなくデジタル側の改善対象判定にも使える、という§3-10の結論を裏付けている。**

---

## 7. まとめ表

| 課題(§3) | モデル構造への含意(§4, §5) | 対応する既存レバー(§2) |
|---|---|---|
| 律速項がACE-bound/SRAM-boundで分岐(3-1) | 構造変更の前に律速判定を行う(4.1) | `perf_breakdown.sh`による分解(コンパイル不要) |
| 重み複製がarea↔SRAM↔powerを結合(3-2) | 重み再利用度の高い演算を優先(4.2) | `num_aces` |
| デジタル側は固定レイテンシフロア(3-3) | Transformer比重の増加に伴い絶対値も増加と推測 | (SDK外。モデル設計側の判断) |
| デジタル側はメモリ帯域ボトルネック(3-4) | MAC数でなくメモリトラフィックを削減対象にする(5.1) | OCRAM0/OCRAM1、`nMPs`(逆効果に注意) |
| 量子化ビット幅は8/16固定(3-5) | 量子化以外の軸(SKU・構造)で改善する | `tensor_n_bits` |
| 電力モデルは相対比較専用(3-6) | 絶対値のサインオフには使わない | — |
| デジタル側面積が推定不可(3-7) | 全デジタル案の面積劣位は別途確認が必要 | — |
| デジタル側PPA推定は全体的に未成熟(3-7.1) | キャリブレーション・効率計算・フルグラフ実行の粗さを踏まえ数値を額面通りに使わない | — |
| コンパイラ非決定性・既知バグ(3-8) | 重み容量の事前計算でコンパイル試行を減らす(4.2) | — |
| A_max未確定(3-9) | モデル軽量化(約24%短縮目標)かA_max見直しの二択 | — |
| 電力もSRAM-bound/ACE-boundの分岐を引き継ぐ(3-10) | 律速判定(4.1)が電力側の改善対象判定にも使える | `tools/power_breakdown/power_breakdown.py` |
| num_aces↑によるpower悪化はACEスリープが主因(3-10.1) | SKU選定時のpower見積りはACE待機電力を優先して評価 | — |
| 実ハイブリッドのデジタル側ボトルネックは全デジタル評価と異なる(3-11) | Attention系の計算密度改善が有効。OCRAM/DDR調整は効果薄(§6.1, §6.2) | `pow*Pj`分離実測(§5.1手法の適用) |
| depthwise Convは強制デジタル化 | on-chip密度を優先する場合は比率を絞る(4.2) | `MarkDepthwiseConvsAsDigital` |
| 多カメラ・多視点入力はSRAM負荷増(4.2) | 特徴マップ解像度・カメラ数に注意 | — |
| Attentionはoff-chip一括処理になりやすい(4.2) | grouped Conv変換の効果は未実測 | `AttentionDetr`等のRewriteRule |
| ACE-boundなモデルはクロスバー充填率が低い(4.3) | 層形状・タイル分割で充填率を上げる余地(BEVFormer 32.6% vs YOLOPX 49.6%) | `auto_partition`、タイル容量4,980,736重み/タイル |
| LLM/VLAはKVキャッシュ・動的長で新課題(5.2) | メモリトラフィック主導の設計指針を継承 | `use_kv_cache`分岐、`vidRope`/`vidMultiQueryExpand` |

---

## 8. 参照

- [00_overview.md](00_overview.md) — SDK全体構成、レベルA/Bのアナログ/デジタル分割
- [01_compilation.md](01_compilation.md) — コンパイルフロー、RewriteRule分類、量子化ポリシー
- [02_ppa_estimation.md](02_ppa_estimation.md) — PPA推定式・SRAM/ACE分解手法・電力未算入項目
- [03_accuracy_simulation.md](03_accuracy_simulation.md) — 精度シミュレーションとBCM忠実度モデル
- [05_all_digital_ppa.md](05_all_digital_ppa.md) — 全デジタル実行の実測、`nMPs`/OCRAMスイープ
- [06_hybrid_digital_and_structural_analysis.md](06_hybrid_digital_and_structural_analysis.md) — 実ハイブリッド構成のデジタル側ボトルネック実測(§3-11)、ONNXグラフに基づくYOLOPX/BEVFormerの構造解析(§4.3)
- [to_structural.md](conversion_steps/to_structural.md) — on/off-chipマーキング機構、depthwise conv扱い
- [PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) — BEVFormer-Tiny SKU探索(72 ACEが唯一の可行点)
- [PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md) — YOLOPX SKU探索(48 ACEが最適点)
- [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) — `mythic-compiler`/`mythic-ppa-estimators`の使い方、既知バグ
- `tools/power_breakdown/power_breakdown.py` — §3-10で使用した電力コンポーネント別内訳の抽出スクリプト(`_ppa_*.tar.gz`のみで再実行可能、funcsim不要)
- [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) — アナログ/デジタル演算振り分けの実測(MAC比率 vs 処理時間比率)
