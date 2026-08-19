# 07. PPA改善の課題整理 — 既存探索の横断統合

Mythic M2000 SDK上でのAIモデルPPA(Power/Performance/Area)改善に必要な事項と主要な課題を、既存の実測結果を横断的に整理したもの。対象は主に BEVFormer-Tiny([PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md))と YOLOPX([PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md))の2モデルのSKU探索結果、および全デジタル実行の実測([05_all_digital_ppa.md](05_all_digital_ppa.md))。

本ドキュメントは**新規のコード調査・実機実行を行わず**、既存ドキュメント(`00_overview.md`〜`06_to_structural.md`, `PLAN_*`, `HOWTO_*`, `FUTURE_*`)の記述・実測値を再整理・外挿したものである。数値の一次出典はすべて各節に明記する。外挿・未検証の記述には**[推測]**を付す。

---

## 目次

- [1. 位置づけ](#1-位置づけ)
- [2. SDKが露出するPPAレバーの総括](#2-sdkが露出するppaレバーの総括)
- [3. 主要課題](#3-主要課題)
- [4. 律速要因別に見たモデル構造の指針](#4-律速要因別に見たモデル構造の指針)
- [5. LLM/VLA等、将来の多層Transformerモデルへの課題](#5-llmvla等将来の多層transformerモデルへの課題)
- [6. まとめ表](#6-まとめ表)
- [7. 参照](#7-参照)

---

## 1. 位置づけ

BEVFormer-TinyとYOLOPXという2モデルのSKU探索([PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md), [PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md))は、それぞれ独立に「`num_aces`をどこに設定すべきか」という問いに答える形で完了している。両探索は互いに正反対の結論(BEVFormerは72 ACEが必須、YOLOPXは48 ACEが最適)に至っており、この対比自体がPPA改善の課題の多くを含んでいる。本ドキュメントはこの2つの探索結果と、SRAM分解手法([02_ppa_estimation.md](02_ppa_estimation.md) §3.9)、全デジタル実行の実測([05_all_digital_ppa.md](05_all_digital_ppa.md))、コンパイラのグラフ書き換え規則([01_compilation.md](01_compilation.md))、on/off-chipマーキング機構([06_to_structural.md](06_to_structural.md))を統合し、以下の3点に答える形で整理する:

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

### 3-4. デジタル側はメモリ帯域ボトルネックであり、MAC数を増やすと悪化しうる

BEVFormerフルグラフを全デジタル実行した`nMPs`スイープ(2GHz固定):

| nMPs | fps | latency | MAC利用率 | Power@30fps |
|---|---|---|---|---|
| 288 | 16.99 | 58.87 ms | 41.67% | 25.58 W |
| 576 | 5.99 | 167.03 ms | 7.34% | 27.54 W |
| 1152 | 3.75 | 266.33 ms | 2.30% | 33.46 W |
| 2304 | 4.02 | 248.65 ms | 1.23% | 37.27 W |

出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3。

**含意**: MACアレイを増やすとexposed DMA(隠蔽できないDMAサイクル)が支配的になり、fpsが下がりDDRトラフィックが増えてpowerも悪化する。「デジタル実行のボトルネックはMAC数ではなくメモリ帯域である」([05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3の明記)。効くレバーはOCRAM0(オンチップSRAM)のみで、512MBまで拡張すると初めて33.19ms/18.27Wに達するが、この容量は面積・コストの観点で非現実的([05_all_digital_ppa.md](05_all_digital_ppa.md) §4.3, §6)。この観察は§5(LLM/VLAへの課題)で直接引用する。

### 3-5. 量子化ビット幅は8/16bit固定で、精度制約に対して既に絞る余地がない

`QuantizationConfig.tensor_n_bits`は8か16のみ許可され、4/6bitは`ValueError`で拒否される(出典: `PLAN_bevformer_ppa_exploration.md` §1.1)。既存2モデルはいずれも既定8bitで精度制約(FP32同等)を満たしている(`PLAN_bevformer_ppa_exploration.md` §2、`PLAN_yolopx_ppa_exploration.md` §2)。

**含意**: 精度を犠牲にPPAを改善する方向の量子化ポリシー探索は、そもそもSDKが露出する範囲では選択肢がない(下位ビット幅が使えないため)。PPA改善の主戦場は量子化ではなく、SKU選定(`num_aces`)とモデル構造そのものになる。

### 3-6. 電力モデルはLeakage/ClockTree/PCIe/D2D未算入で相対比較専用

アナログ側`power_estimator.py`の`total_power`には Functional Unit と Interconnect のみが算入され、Leakage/Clock Tree/PCIe/Die-to-Dieはいずれもコメントアウトされ未算入(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §4.11)。デジタル側`Power@30fps`も、実際に出るfps(`eff. fps`)でのエネルギーを30fpsへ単純に線形外挿した値であり、「30fpsで動作した場合の電力の物理モデル」ではない(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §5.2)。デジタル側も同様にダイナミック電力のみで、leakage/clock tree/PCIe/D2Dは非含有([05_all_digital_ppa.md](05_all_digital_ppa.md) §5.3)。

**含意**: 本探索群で得られる数値はすべて**同一条件下での相対比較(A/B比較)専用**であり、絶対値のサインオフには使えない。異なるfps基準(BEVFormer m2048の23fpsクランプ vs m2072の30fps)や異なるSDKバージョン間(YOLOPX実測0.743W vs レポート値0.679W、+9.4%)の値を混在させて比較してはならない。

### 3-7. デジタル側の面積はSDKから推定できない

`vnnmap`バイナリの文字列にarea/mm²/die相当のシンボルは存在しない(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §6)。アナログ側は物理傾き5.278 mm²/ACEで算出可能だが、デジタル側はv-MPコア1個あたりの面積・SRAMマクロ面積/MBのいずれもSDK外の情報が必要。

**含意**: ハイブリッド構成(アナログ+デジタル)とオール・デジタル構成をarea軸で公平に比較することは、現状のSDKだけでは不可能。全デジタル案の実行可能性を評価するには、latency/powerだけでなくarea見積り(SDK外のデータシート等)を別途取得する必要がある(既にlatency/powerだけでも全デジタルはハイブリッドに対し1.9倍/5.7倍劣位であることが判明しているため、area面での劣位が確定すれば全デジタル案は完全に棄却できる)。

### 3-8. コンパイラ/ソルバーの非決定性・既知バグ

- BEVFormer m2024: CP-SAT `status: UNKNOWN`(122分後にタイムアウト。INFEASIBLE証明ではない。出典: `HOWTO_ppa_exploration_tools.md` §1)。
- YOLOPX m2024: 重み容量108.6〜121%超過による**真のINFEASIBLE**(出典: `PLAN_yolopx_ppa_exploration.md` §3.3)。
- YOLOPX m2072: 1回目コンパイルが`malloc(): invalid size (unsorted)`のヒープ破壊で異常終了、同一設定の再試行で成功(出典: `PLAN_yolopx_ppa_exploration.md` §3.6)。
- マルチスレッドソルバーの非決定性により、同一設定でもコンパイル結果が数%変動する(出典: `PLAN_yolopx_ppa_exploration.md` §6-2)。

**含意**: SKU探索の1点あたりコンパイル約90〜120分・PPA推定約1〜4時間という長いイテレーション時間の中に、タイムアウト・INFEASIBLE・一過性バグが混在する。探索の実務コストとして、単純な「コンパイルが通るか」の判定にすら複数回の再試行や重み容量の事前計算(§4参照)が必要になる。

### 3-9. A_max(面積上限)が事業側で未確定

BEVFormerの探索は「A_maxが380mm²以上でなければ、現行SDK・現行モデルでは可行SKUが存在しない」という結論に達しているが、A_max自体の値はビジネス側(コスト・パッケージング制約)から未確定(出典: `PLAN_bevformer_ppa_exploration.md` §0, §6, §7)。

**含意**: 「72 ACEが必須」という技術的結論の実効性は、A_maxが確定するまで保留状態にある。A_maxが380mm²未満であれば、モデル側の軽量化(アナログ側処理時間を約24%短縮して48 ACEを救済する)かA_max自体の見直しのいずれかが必須になる(`PLAN_bevformer_ppa_exploration.md` §4結論)。この軽量化の具体的な着手方法は§4で述べる。

---

## 4. 律速要因別に見たモデル構造の指針

質問「レイテンシ改善にはSRAMアクセスを減らすことが必要だと思うが、適したモデル構造は何か」への回答。

律速項がSRAM-bound(BEVFormer)かACE-bound(YOLOPX)かでモデルが2分される以上(§3-1)、「SRAMアクセスを減らすモデル構造は何か」だけでは片方のケースにしか答えられない。**ACE-boundなモデルには、対になる問い「ACE演算数(クロスバー利用率)を減らすモデル構造は何か」が別途必要**になる。以下、まず両ケースに共通する前提(4.1)、次にSRAM-bound向け(4.2)、ACE-bound向け(4.3)の指針を分けて述べる。

### 4.1 まず律速判定を行う(構造変更より先にすべきこと)

どちらの指針が有効かは、そのモデルの律速項がSRAMかACEかで決まる(§3-1)。SRAM-boundなモデルに対してACE演算削減を目的に構造を変えてもSRAM項が頭打ちのままで改善せず、逆にACE-boundなモデルにSRAM削減を適用してもACEクリティカルパスで頭打ちのままで改善しない。既存の`_ppa_*.tar.gz`から`tools/perf_breakdown/perf_breakdown.sh`で約9秒で判定できる(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.9(4))ため、モデル構造の変更に着手する前に必ず行うべき最初のステップである。

### 4.2 SRAM-boundなモデル向け——SRAMアクセスを減らす構造

SRAM負荷の主因は「並列化のための重み複製」である。§3-2で示した通り、SRAMトラフィックの増加はACE数を増やした際の重み複製に強く結びついている(YOLOPX 48→72で重み+18.0%、SRAM読+5.3%/書+7.3%)。したがって「SRAM削減に適したモデル構造」は、言い換えれば**「同じACE数でも重み複製が少なくて済む構造、またはSRAM往復あたりの計算密度が高い構造」**である。

具体的な指針(既存実測からの含意):

- **重み再利用度の高い畳み込みはSRAM往復あたりの計算密度が高く、アナログMMAに適する。** BEVFormerのResNetバックボーンはMAC比率98.3%を占めながらアナログ側で69.4%という比較的高いACE利用率を達成している(出典: [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)、`PLAN_bevformer_ppa_exploration.md` §2)。1回の重みロードで多数の出力位置を生成する構造(通常の畳み込み)はSRAM再アクセスの相対コストが低い。

- **Depthwise Convは強制的にデジタル(SALU)に落ちる。** `MarkDepthwiseConvsAsDigital`が`group == out_channels`かつ`in_channels/group == 1`の条件を満たすConvに`__digital_onchip`属性を付与し、アナログMMAではなくSALU(デジタル)で処理される(出典: [06_to_structural.md](06_to_structural.md) §8.1)。省パラメータ設計としてよく使われるDepthwise Convは、この意味でオンチップのアナログ処理密度を下げる方向に働く。on-chipでの計算密度を優先するなら、depthwise比率を絞るか、デジタル側で処理される前提でレイテンシ予算を確保する必要がある。

- **多視点・多カメラ入力は特徴マップサイズに比例してSRAM負荷を増やす。** BEVFormer(6カメラ、SRAM-bound、26.95ms)とYOLOPX(単一カメラ、ACE-bound、6.83ms)の対比がこれを裏付ける(出典: [02_ppa_estimation.md](02_ppa_estimation.md) §3.9)。カメラ数・解像度・特徴マップ解像度の増加は、SRAM-bound化のリスクを直接高める。

- **Attention/Transformer部分はoff-chip一括処理になりやすい。** BEVFormerの`to_structural`実装はtransformer側サブグラフの全ノードを一括off-chip化する(`everything_off_chip`、出典: [06_to_structural.md](06_to_structural.md) §7.1)。コンパイラ側には`AttentionDetr`等、Attentionを`group=8`のグループ畳み込みへ変換する最適化パスが存在するが(出典: [01_compilation.md](01_compilation.md) §3.1.2(G))、これは計算のマッピング先(アナログMMAへ載せられる表現への変換)を変えるものであり、SRAM往復の絶対量そのものを減らす保証はない**[推測]**——この変換パスがSRAMトラフィックに与える効果は本調査群では実測していない。

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

- **タイル分割・パーティションの粒度がACE演算回数に影響する。** `auto_partition`(コンパイラ内部)はSRAMサイズ制約に応じて演算を複数パーティションに分割する(出典: [01_compilation.md](01_compilation.md) §3.3.1、[00_overview.md](00_overview.md) §3.5)。1層が過剰に分割されるほどACE演算(ドット積発行)の回数が増え、1回あたりの充填率が下がりうる**[推測]**。層の重み・活性化サイズをタイル容量(1タイル=4,980,736重み)に対して過度に大きくしない設計が、不要な分割を避ける方向に働く。

- **総MAC数の削減は充填率が保たれる場合にのみACEクリティカルパス短縮に直結する。** ACE-boundなモデルでMAC数を減らしても、1回あたりの充填率が同時に下がれば`timesteps`数(=ACE演算回数)は減らない。したがって層のチャネル数・カーネル形状を削減する際は、削減後もクロスバー充填率が維持されるか(充填率がすでに低い層を優先して削るか、単純に全層を均等に縮小するか)を区別する必要がある**[推測]**。本調査群ではMAC数削減が充填率に与える影響そのものは実測していない。

- **SRAM-boundなモデルへの適用時は優先度が下がる。** BEVFormerのように既にSRAM側が律速している場合、ACE充填率を上げてもレイテンシは改善しない(§3-1)。この指針はACE-boundなモデル、またはSKU変更(ACE増設)でSRAM-boundからACE-boundに転換する場合に限って有効になる。

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

2. **動的シーケンス長・自己回帰デコードとの整合。** `vnnmap`はdynamic batch非対応で、batch=-1を1に固定する(出典: [05_all_digital_ppa.md](05_all_digital_ppa.md) §8)。pythia(GPT-NeoX系、SDK内で唯一のLLM相当モデル)の`to_structural`実装は、`use_kv_cache=True`の場合に`fix_sequence_length`/`simplify_inputs`をスキップする分岐を既に持っており(出典: [06_to_structural.md](06_to_structural.md) §7.3)、SDK側にKVキャッシュ運用を想定した仕組みは存在する。しかし自己回帰デコード時にシーケンス長が変化する場合の静的shape要求との整合性、およびそのPPAへの影響は本調査群では未検証。

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

## 6. まとめ表

| 課題(§3) | モデル構造への含意(§4, §5) | 対応する既存レバー(§2) |
|---|---|---|
| 律速項がACE-bound/SRAM-boundで分岐(3-1) | 構造変更の前に律速判定を行う(4.1) | `perf_breakdown.sh`による分解(コンパイル不要) |
| 重み複製がarea↔SRAM↔powerを結合(3-2) | 重み再利用度の高い演算を優先(4.2) | `num_aces` |
| デジタル側は固定レイテンシフロア(3-3) | Transformer比重の増加に伴い絶対値も増加と推測 | (SDK外。モデル設計側の判断) |
| デジタル側はメモリ帯域ボトルネック(3-4) | MAC数でなくメモリトラフィックを削減対象にする(5.1) | OCRAM0/OCRAM1、`nMPs`(逆効果に注意) |
| 量子化ビット幅は8/16固定(3-5) | 量子化以外の軸(SKU・構造)で改善する | `tensor_n_bits` |
| 電力モデルは相対比較専用(3-6) | 絶対値のサインオフには使わない | — |
| デジタル側面積が推定不可(3-7) | 全デジタル案の面積劣位は別途確認が必要 | — |
| コンパイラ非決定性・既知バグ(3-8) | 重み容量の事前計算でコンパイル試行を減らす(4.2) | — |
| A_max未確定(3-9) | モデル軽量化(約24%短縮目標)かA_max見直しの二択 | — |
| depthwise Convは強制デジタル化 | on-chip密度を優先する場合は比率を絞る(4.2) | `MarkDepthwiseConvsAsDigital` |
| 多カメラ・多視点入力はSRAM負荷増(4.2) | 特徴マップ解像度・カメラ数に注意 | — |
| Attentionはoff-chip一括処理になりやすい(4.2) | grouped Conv変換の効果は未実測 | `AttentionDetr`等のRewriteRule |
| ACE-boundなモデルはクロスバー充填率が低い(4.3) | 層形状・タイル分割で充填率を上げる余地(BEVFormer 32.6% vs YOLOPX 49.6%) | `auto_partition`、タイル容量4,980,736重み/タイル |
| LLM/VLAはKVキャッシュ・動的長で新課題(5.2) | メモリトラフィック主導の設計指針を継承 | `use_kv_cache`分岐、`vidRope`/`vidMultiQueryExpand` |

---

## 7. 参照

- [00_overview.md](00_overview.md) — SDK全体構成、レベルA/Bのアナログ/デジタル分割
- [01_compilation.md](01_compilation.md) — コンパイルフロー、RewriteRule分類、量子化ポリシー
- [02_ppa_estimation.md](02_ppa_estimation.md) — PPA推定式・SRAM/ACE分解手法・電力未算入項目
- [03_accuracy_simulation.md](03_accuracy_simulation.md) — 精度シミュレーションとBCM忠実度モデル
- [05_all_digital_ppa.md](05_all_digital_ppa.md) — 全デジタル実行の実測、`nMPs`/OCRAMスイープ
- [06_to_structural.md](06_to_structural.md) — on/off-chipマーキング機構、depthwise conv扱い
- [PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) — BEVFormer-Tiny SKU探索(72 ACEが唯一の可行点)
- [PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md) — YOLOPX SKU探索(48 ACEが最適点)
- [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) — `mythic-compiler`/`mythic-ppa-estimators`の使い方、既知バグ
- [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) — アナログ/デジタル演算振り分けの実測(MAC比率 vs 処理時間比率)
