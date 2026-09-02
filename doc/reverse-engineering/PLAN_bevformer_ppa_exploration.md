# 計画: BEVFormer-Tiny PPA(面積-電力トレードオフ探索)パラメータ探索

状態: **探索完了(SDK v26.05.2)。Phase 0-5 全完了。結論は「72 ACE のみ可行(33ms 制約)」。** m2048=43.47ms>33ms(不可行)/ m2072=31.58ms(可行、マージン4.3%)、24/32 はコンパイル不可。本ドキュメントは設計書であり、コード実装は含まない。

### 進捗サマリ

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | 資料精読(PPA Estimator Datasheet v0.4 / Compiler Opt Report v1.2)→方法論の疑問を事前解決 | ✅ 完了 |
| 1 | SDK 取得・展開・docker load・コンテナ起動・ツール動作確認 | ✅ 完了 |
| 2 | m2072サニティチェック(コンパイル→ppa推定、ドキュメント値31.58ms/4.5W/ACE数72と照合) | ✅ 完了。全実測値がドキュメント値と一致(下記)。環境正当性を確定 |
| 3 | m2048検証(6カメラconfig、Combined latency直読で33ms判定) | ✅ 完了。43.47ms>33ms=不可行を確定 |
| 4 | m2024/m2032検証(コンパイル可否) | ✅ 完了。両者とも利用不可(24=CP-SAT失敗/32=定義なし)。可行は48/72の2点 |
| 5 | トレードオフ表作成(物理面積5.278mm²/ACE) | ✅ 完了。§4.1 Stage C表 |

**方法論(重要):**
- **6カメラはネイティブ**。BEVFormer compiler config の `--input-dims` は `6 928 1600 3`(コンテナ内で実物確認)。`mythic-ppa-estimators` の `Combined Analog + Digital NPU Latency` は 6カメラの最終 latency を直読する。`--power-inference-rate 30` で1回だけ実行する(手動での ×6 換算・2回実行は不要——やると6倍の二重計上になる)。
- **ACE数は正表示**(structured output で `Number of ACEs: 72`)。
- 環境: compiler docker tag `v26.05.2`、venv `mythic-model-zoo/.venv/`(uv管理)、コンテナ `mythic_ppa_explore_2605_2b`、ワークスペース `/tmp/ppa_workspace_2605_2/`。詳細は [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md)。
- Compiler Optimization Report v1.2 の m2072 ドキュメント値(照合基準): Combined **31.58ms**(analog 26.95[6カメラ込]+digital 4.63)、Total power **4.505W@30fps**、ACE util 69.43%、ACE数72。

**Phase 2 m2072サニティチェック実測:** ドキュメント値と完全一致し環境の正当性を確定。

| 項目 | 実測 | ドキュメント値 |
|---|---|---|
| Combined Analog+Digital NPU Latency | **31.58 ms** | 31.58 ms |
| Analog NPU Processing Time | 26.95 ms(6カメラ込) | 26.95 ms |
| Digital Estimated Frame Processing | 4.63 ms | 4.63 ms |
| Combined Frame Rate | 31.67 fps | 31.67 fps |
| Total Combined Power @30fps | **4.505 W**(analog 3.287 / digital 1.218) | 4.505 W |
| ACE Utilization | 69.43% | 69.43% |
| **Number of ACEs** | **72(正表示)** | 72 |

- Critical Path ACE Latency 23.37ms、Min理論(72 ACE均等並列)18.71ms、Total ACE MACs 955,023,360,000、Digital MAC util 9.69%。
- analog 26.95ms に6カメラ分が既に含まれ、Combined 31.58ms が6カメラの最終 latency。**72 ACE は33ms制約を満たす(マージン1.42ms=4.3%)**。
- **die area 出力なし**: v26.05.2 の ppa-estimator は die area 行を出力しない。代わりに `Number of ACEs: 72` が正表示。**面積は物理傾き 5.278mm²/ACE で算出する**(§4.1)。

**Phase 3 m2048検証実測:**

| 項目 | m2048実測 | m2072参照 |
|---|---|---|
| **Combined Analog+Digital NPU Latency** | **43.47 ms** | 31.58 ms |
| Analog NPU Processing Time | 38.85 ms | 26.95 ms |
| Digital Estimated Frame Processing | 4.63 ms | 4.63 ms |
| Combined Frame Rate | 23.00 fps | 31.67 fps |
| Total Combined Power(fps基準が異なるため直接比較不可、§下記注記) | 3.428 W @23fps(analog 2.494 / digital 0.934) | 4.505 W @30fps(analog 3.287 / digital 1.218) |
| ACE Utilization | 72.26% | 69.43% |
| **Number of ACEs** | **48(正表示)** | 72 |
| Critical Path ACE Latency | 35.22 ms | 23.37 ms |
| Total ACE MACs(モデル同一) | 955,023,360,000 | 955,023,360,000 |

- **【判定】m2048 = 43.47ms > 33ms → 制約NG(不可行)。** L0 IR: Tiles 13, ACE Weights 31,253,248, ACE calc 8,421,600, Buffers 89.77%。コンパイルは~94分で完走(exit 0)。
- **power推定でfpsクランプが自動作動**: `--power-inference-rate 30` 指定に対しログに `Power inference rate of '30' exceeds estimated. Using estimated rate of '23' instead.` → m2048 の power **3.428 W は 23fps 時**の値(analog 2.494 / digital 0.934)。72(31.67fps)の30fps指定4.505Wとはfps基準が異なり**そのまま並べて比較できない**(§4.1 表の注記)。新フラグ `--allow-fps-over-max` で30fps強制推定も可能。

**Phase 4 の確定事項:**
- **m2032 は利用不可。** コンパイルすると `target.cpp:38 FATL| ABORT: Unable to build AMP architecture "m2032" without compatibility enabled!`。`--boreas-a-compatible` を付けてもエラーが `with compatibility enabled!` に変わるだけで中断。決定的証拠として `compilerd-bin:v26.05.2` の `/mythic/dnn_compiler` バイナリの strings 調査で**定義済みAMPアーキは `m2024`/`m2048`/`m2072` の3つのみ**(無効値 `INVALID_XYZ` と全く同じエラー文言=m2032のターゲット定義が存在しない)。
- **m2024 は利用不可。** CP-SAT段まで到達したが、122分(7319秒)後に `status: UNKNOWN` → `schedule.cpp:915 FATL| CHECK FAILED: success L0 optimization failed for Crate "main_graph_main_graph"!`(exit 1)。ACE Weights は 31.25M だが、24 ACE=6タイルへの充填は実用時間内に CP-SAT が解けない。
- **【Phase 4 結論】コンパイル可能な num_aces は 48/72 の2点のみ**。24=CP-SAT失敗、32=ターゲット定義なし。トレードオフ探索は 48 vs 72 の比較となる。

---

## 0. 目的と制約

**前提(重要)**: 実チップは未確定であり、本探索は**運用時のチューニングではなく SKU 選定(どの ACE 数でテープアウトするかの事前検討)に近い**。したがって `num_aces` は「既に製造済みのシリコンのうち何タイル使うか」という運用選択ではなく、**そのままダイ面積・製造コストを決める設計変数**として扱う。

**精度制約・レイテンシ制約は、既存の retrained モデル(`num_aces=72`, m2072構成)で実測済みであり、満たされている**(§2参照、出典: `doc/reports/Model Summary Report.pdf` v3.0, `doc/reports/Compiler Optimization Report - BEVFormer-Tiny.pdf`)。したがって本探索の目的関数は精度・レイテンシの制約下での探索ではなく、**この既知の可行点(72 ACEs, 380mm², 4.505W)を起点に、より小さい`num_aces`でも制約を満たせるか**という一点に絞られる。

**area と power の間には本質的なトレードオフがある**: `num_aces`を減らせばareaは下がるが、同じ演算量をより少ないタイルで処理するためlatencyが増える方向に働き、latency制約(33ms)を満たすために並列化・クロック等で補うとpowerが増える方向に働く場合がある。逆に`num_aces`を増やせばarea/コストは上がるが、余裕を持ってlatency制約を満たせ、無理な高稼働率を避けられるためpowerが下がる場合がある。**area は「小さいほど望ましい」という真の目的変数であり、power も同様に「小さいほど望ましい」目的変数**——両者は単純な優先順位(area優先→powerでタイブレーク)には落とし込めず、**2目的のトレードオフ(Paretoフロンティア)として同時に探索する**必要がある。

`A_max`(コスト・パッケージング等の事業制約からの上限)は、この2目的探索の**候補空間を絞るカットオフ**として使う(area > A_maxの候補はそもそも検討対象外にする)。既知の可行点である`num_aces=72`(380mm²)自体がA_maxを超えている可能性もあり、A_maxの値次第では「現状の実測済み構成そのものが候補として使えない」という結論もありうる。

これを踏まえると、問題は以下のように再定式化される:

> **area上限(A_max)以内で、レイテンシ制約(33ms)を満たす`num_aces`候補を特定し、その中でarea-power の Pareto最適点(どちらかを改善するには他方を犠牲にせざるを得ない点の集合)を求め、その中からユーザーが選択する。**

もし `area ≤ A_max` の範囲内でどう組んでも(既知の可行点である72 ACEsを除いて)レイテンシ制約を満たせない場合、それは「マッピングの工夫では解決できない、area上限そのものの見直しが必要」という重大な結論になる。したがって本探索の最初のマイルストーンは「Pareto探索」の前に「**A_max以内での可行性確認**」である。

**問題設定**:

| 項目 | 内容 |
|---|---|
| 制約1(area) | `num_aces` に対応する die area ≤ A_max(具体値は未確定 — §6に記載、候補空間のカットオフとして使用) |
| 制約2(精度) | 満たされている(既存retrainedモデルで実測済み、§2参照)。`num_aces`は精度に影響しないため(§3)、本探索のスコープでは再検証不要 |
| 制約3(レイテンシ) | 1フレームあたり ≤ 33ms。`num_aces=72`では31.58msで満たすが、マージンは1.42ms(4.3%)のみ(§2) |
| 目的関数 | area・power の2目的同時最小化(Pareto最適点の集合を求める。一方を優先して他方をタイブレークにするのではない) |

これは「area上限(A_max)以内かつレイテンシ制約を満たす`num_aces`候補群についてarea-powerのPareto最適点を求める」という多目的最適化問題であり、全探索ではなく変数の依存関係を利用した段階的探索で解く。

**SKU選定という位置づけの実務上の意味**: 実チップ設計では `num_aces` は連続変数ではなく、量産化する品種数(コスト)を考えると数値候補は絞られる(例: 24/32/48/72など少数のSKU案)。したがって本探索の「`num_aces` を振る」作業は、無数の値を試す最適化ではなく、**A_max以内の少数の現実的なSKU候補群について、それぞれの実現可能なpower/latencyを計測し、area-powerのトレードオフ表を作る**作業として設計する。最終的にどの点を選ぶか(area優先かpower優先か)はステークホルダーの判断に委ねる。

---

## 1. 前提: SDKが実際に提供する機能

### 1.1 ユーザー側(Python)から触れるPPA関連パラメータ

| パラメータ | 場所 | 値域 | 効き先 |
|---|---|---|---|
| `num_aces` | `power_estimator.py:619`, `perf_analysis.py:92-97,113-117` | 4の倍数、既定24。BEVFormer-Tinyの実測構成は72 | die area(単調)、ACEクリティカルパス |
| `n_mps` | `vnnmap/compilation.py:55-57`, `run_vnnmap.py` | `{1,4,8}` のみ | タイル分割・並列度 |
| `OCRAM1` / `DDR` | `run_vnnmap.py:14-19`(`system.cfg`) | 既定2MB/512MB | DDR↔OCRAM昇格、latencyへの影響 |
| `frequency` / `DDRConfig` | 同上 | 既定625MHz/100 | latency全体。CLIフラグなし、`system_config`直書きが必要 |
| `QuantizationConfig.tensor_n_bits` | `vnnort/quantizer/quantization_config.py` | テンソル/ブロック単位のビット幅上書き。`vid_quantizer.py:176-182`により**8か16のみ**(6/4bitは`ValueError`で拒否される) | 精度⇄PPA。本探索では既定値(8bit)で精度制約を満たすため使用しない(§2) |

**露出されていないもの**: ADC/DAC bit深度(8bit固定, [pow:17-22])、sparsity、BitSpreadingMode(クローズドソースの`dnn_compiler`/`vnnmap`内部で決定)。

### 1.2 計測ツールと既知の限界

ユーザーが実際に叩くエントリーポイントは3つ(`mythic-compiler`/`mythic-ppa-estimators`/`convert_model.py steps=eval_trained`、`doc/user-guides/GEN2 User Guide.pdf`)。以下の`perf_analysis.py`/`power_estimator.py`は、公式CLI`mythic-ppa-estimators`の内部実装(この抽出済みリポジトリで直接読める部分)であり、ユーザーは個別に叩く必要はない。ツールの具体的な使い方は[HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md)を参照。

- `perf_analysis.py`: HDF5トレースからlatency/fps/ACE利用率を算出([doc 02_ppa_estimation.md](02_ppa_estimation.md)に式レベルで解析済み)。
- `power_estimator.py`: L0 protobufからpower/TOPS-Wを算出。
- Leakage/Clock Tree/PCIe/Die-to-Die電力は算出はされるが最終値`total_power`には未算入(`power_estimator.py` 該当コードはコメントアウト、doc 02 §4.11。GEN2 User Guideの実行ログにも同旨のNOTE有り)。したがって本探索で得られる数値は**絶対値のサインオフではなく、同一条件下での相対比較(A/B)専用**として扱う。
- `munc`/BCM/TorchNet: 精度(mAP/mIoU等)専用シミュレータ。PPAは出さない。量子化ノイズ・ADC特性を含む。実際のCLIエントリーポイントは`convert_model.py steps=eval_trained`。本探索では既存のretrainedモデルの精度が既に判明しているため使用しない(§2)。

**BEVFormerの6カメラ入力はネイティブ**: v26.05.2 の BEVFormer compiler config は `--input-dims 6 928 1600 3`(6カメラ)で、`mythic-ppa-estimators` の `Combined Analog + Digital NPU Latency` は6カメラの最終 latency をそのまま出力する。この値を直接 33ms 制約と比較してよい(手動での ×6 換算は不要——やると6倍の二重計上になる)。アナログ側処理時間(`Analog NPU Processing Time`)にも6カメラ分が含まれる。

**`num_aces`はPPA Estimator側でなくCompiler側で変える**: `mythic-ppa-estimators --help`にはpositional引数(artifact path)以外に`num_aces`をオーバーライドするフラグが無い(実機で確認済み)。実行ログの`Number of ACEs`はコンパイル済みartifactに埋め込まれた値を表示する。つまり、**本探索で`num_aces`を振るには、`mythic-compiler`のコンパイル条件(`--compiler-config`のYAML内`--amp-arch`)を変えて複数のartifactを作り分ける必要がある**。

### 1.3 コンパイラ内部の自動最適化ソルバー

外側(Python API)には複数パラメータを横断的に試して制約付き最適解を選ぶ機能は存在しない(`explore_model()`は単発の1設定推定関数、`run_vnn_flow`も同様)。

一方、**`dnn_compiler`バイナリ内部には配置・並列化を自動探索するパスが実在することがコード調査(strings解析)で確認済み**:

- `mythic/optimizer/high/passes/auto_partition.cpp` — アナログ/デジタル振り分け([01_compilation.md](01_compilation.md) §3.3.1)
- `mythic/optimizer/high/passes/joint_parallelize_and_partition.cpp` — 並列化とパーティションを同時に最適化するパス(同 §3.3.1)

これを制御するCLIフラグとして、社内資料(Mythic Compiler Guide Part I/II, 20251021版)に以下が記載されている。**これらのフラグ名・デフォルト値は、この抽出済みリポジトリにはHydra設定YAML(`cfg.COMPILER_OPTIONS`の実体)が含まれていないため、コードから直接検証できていない([未検証・社内資料由来])**:

| フラグ | 既定値 | 意味 |
|---|---|---|
| `--relative-objective-target` | 0.7 | ACE利用率の目標値(0.0-1.0)。1.0=理論最良値目標。高いほど探索が粘る=コンパイル時間増 |
| `--optimization-effort-limit` | 10.0 | 配置・割当探索の深さ上限。上げても必ず性能改善するとは限らない |
| `--relative-parallelization` | 1.0(自動決定値) | 並列化係数を自動値から相対的に増減 |

**この探索が目的関数にしているのは ACE利用率であり、area/powerを直接最小化するものではない**。理由: `num_aces`(=area)は外側からコンパイラに与える固定値で、ソルバーはこれを変更する権限を持たない。ソルバーができるのは「与えられた`num_aces`枚の中で、演算配置・並列化・バッファ割当を工夫して理論最小処理時間にどれだけ近づけるか」だけ。ACE利用率 = 理論最小処理時間 / 実測処理時間([02_ppa_estimation.md](02_ppa_estimation.md) §3.6)なので、**利用率を上げることは「同じarea(同じnum_aces)でlatencyを下げる」ことと等価**。したがって、この内部ソルバーは「ある`num_aces`が制約(latency≤33ms)を満たせるかどうか」を正しく判定するための前処理として使う——ソルバーを働かせずに測ると、実際には十分な`num_aces`でも「制約を満たせない」と誤判定するリスクがある。

**注意点(社内資料に明記)**: コンパイラ内部では高速化のため簡略化したACE利用率モデルを使うため、内部目標値と`perf_analysis.py`実測の利用率は一致しない場合がある。したがって**最終判断は必ず`perf_analysis.py`/`power_estimator.py`の実測値で行う**。また、マルチスレッドソルバーのため同一設定でも複数回コンパイルすると結果が微妙に変わることがある。

---

## 2. 精度・レイテンシの現状(既知事実)

`doc/reports/Model Summary Report.pdf`(v3.0)および`doc/reports/Compiler Optimization Report - BEVFormer-Tiny.pdf`(v1.2)により、BEVFormer-Tiny(6x1600x900, `num_aces=72`, m2072構成・380mm²ダイ、`high_optimization`コンパイラconfig)について以下が計測されている(**ドキュメント値**。Mythic 提供レポートの引用であり、本探索で再計測したものではない。データセットは nuScenes val 全体):

**精度**(FP32 → ANA8 Retrained Model, ドキュメント値):

| 指標 | FP32 | ANA8 Retrained | ANA8 Retrained 100PPM |
|---|---|---|---|
| mAP | 0.2293 | 0.2290 | 0.2270 |
| NDS | 0.2649 | 0.2552 | 0.2529 |

mAPはFP32とほぼ同一。NDSはretrainedで-3.7%、100PPMで-4.5%。

#### 2.1 精度の自前実測(SDK コンテナ、nuScenes v1.0-mini val)

[2026-07-30 実測] 上表のドキュメント値とは別に、`convert_model.py steps=eval_trained` / `steps=eval_fp32` を SDK コンテナ(`mythic_sdk_impl`, v26.05.2)で実行し、精度を自前計測した。**データセットは v1.0-mini val(2シーン・81フレーム)で、ドキュメント値の nuScenes val 全体とは規模が異なる。絶対値をドキュメント値と直接比較してはならない**(mini は小標本のため値が大きく異なる)。目的は「ANA8 Retrained(Mythic モデル)の精度シミュレーションが確率的に変動するか」の確認である。

使用モデル: `models/training/bevformer/bevformer-tiny-{fp32,-1600x900-trained}.onnx`。`val_fraction=1.0`(全81フレーム固定)、seed 非固定。

| モデル | 評価経路 | mAP | NDS | 実行回数 |
|---|---|---|---|---|
| FP32 | `eval_fp32` → `eval_onnx_model`(onnxruntime, CPU, 決定論的) | 0.1998 | 0.2029 | 1(単発で確定) |
| ANA8 Retrained | `eval_trained` → `eval_mythic_model`(TorchNet, GPU, ノイズ注入) | 平均 **0.2162** | 平均 **0.2149** | 5 |

ANA8 Retrained の 5 回実測(同一モデル・同一データ・同一設定):

| run | mAP | NDS |
|---|---|---|
| 1 | 0.21451 | 0.21434 |
| 2 | 0.21784 | 0.21745 |
| 3 | 0.21596 | 0.21574 |
| 4 | 0.21257 | 0.21007 |
| 5 | 0.22017 | 0.21713 |
| **統計** | mean 0.21621 / std 0.00263 / range **0.76pt** | mean 0.21495 / std 0.00268 / range **0.74pt** |

**確認できた事実**:

- **`eval_trained`(Mythic モデルの精度シミュレーション)は run ごとに結果が変動する**。入力(モデル・データ・`val_fraction`)を完全に固定しても、mAP/NDS が約 0.7pt の幅で揺れた。原因は Mythic モデルの `analog_model` が推論のたびにノイズを再サンプルする(seed 非固定・`torch.randn`)ためで、eval モードでもノイズは注入され続ける(`03_accuracy_simulation.md` §6、および §4.2 参照)。
- **FP32(`eval_fp32`)は決定論的**で、単発で値が確定する(onnxruntime CPU 経路、ノイズなし)。
- 変動幅は絶対値では小さい(std ≈ 0.0026、相対で約1.2%)。81フレーム全体の集計で画像単位のノイズが平均化されるため。それでもゼロではなく、**単発 `eval_trained` は点推定にすぎない**。製品保証精度を出すにはモンテカルロ(`mc_eval_trained`)+ 下側トレランス限界での統計処理が要る(`03_accuracy_simulation.md` §7)。
- 注意: mini val 上では ANA8 実測 > FP32 実測 となっており、ドキュメント値(FP32 ≳ ANA8)と傾向が逆転している。これは (a) mini val の小標本性、(b) FP32 と Mythic で evaluator 経路が異なること(`eval_onnx_model` vs `eval_mythic_model`)による。**mini val の絶対値・大小関係は本探索の判断材料にはせず、あくまで「変動の有無」の確認に用いる**。

本探索(`num_aces` を振る area-power トレードオフ)への含意は下記の通り変わらない: `num_aces` はハードウェアマッピングの設定で精度シミュレーション結果に影響しないため、この精度変動は探索の各点で再計測する必要がない(§3、下記の箇条書きも参照)。

**レイテンシ・電力・面積**(6カメラ、Phase 2 で実測再現済み):

| 項目 | 値 |
|---|---|
| Mythic NPU構成 | m2072(72 ACEs, 18 ACEタイル, 4MB/タイル, **380 mm²**ダイ面積), 288 Digital NPUコア |
| Latency(6カメラ、high optimization) | **31.58 ms**(33ms制約に対しマージン1.42ms=4.3%) |
| Frames per second | 31.67 fps |
| Power(target 30fps) | **4.505 W**(analog 3.287 / digital 1.218) |
| ACE利用率 | 69.43% |
| Weight utilization | 39.88%(35.8M / 89.65M) |
| SRAM utilization | 8.21% |

**この事実が本探索に与える意味**:

- 精度制約(FP32同等)は既定の量子化ポリシー(活性化8bit/重み8bit per-channel/bias等16bit)で既に満たされている。`tensor_n_bits`は8/16のみしか選べない仕組み上、これより低ビット化してコストを下げる余地はそもそもない(§1.1)。したがって**量子化ポリシーを振って精度を確認する作業は不要**であり、本探索は量子化ポリシーを固定したまま`num_aces`のみを振る。
- レイテンシ制約は`num_aces=72`という、検討対象の中でも最大級のSKUでかろうじて(4.3%マージン)満たされている。`num_aces`を減らすとlatencyが増える方向に働くため(§0)、**より小さい`num_aces`候補が33ms制約を満たせない可能性が高い**。72より小さいSKU候補それぞれについて制約を満たすかどうかを確認することが、本探索の中心課題になる。
- `num_aces`はハードウェアマッピング(area)の設定であり、精度シミュレーション結果に影響しない(§3)。したがって`num_aces`を振る探索では、精度を再計測する必要はなく、既存のretrained ONNXアーティファクト(`models/training/bevformer/bevformer-tiny-1600x900-trained.onnx`)を異なる`--compiler-config`でコンパイルし直すだけでよい——nuScenesデータセットへのアクセスや精度シミュレーションの再実行は不要。

---

## 3. 探索変数の分解(なぜ全探索でなく段階的探索が可能か)

- **areaは`num_aces`のほぼ単調関数**(`ACE_TILE_AREA_PS_MM2 * num_aces/4`, [02_ppa_estimation.md](02_ppa_estimation.md) §5)。
- **latencyは`num_aces`/`n_mps`/`frequency`/`OCRAM`(スピル有無)/コンパイラソルバー設定に依存**。ビット幅(量子化ポリシー)は本探索では固定するため変数としては扱わない(§2)。
- **powerは上記全部に依存**(ACE動作電流は稼働率次第、SRAMアクセス電力はメモリ構成次第)。
- **精度は`num_aces`/`n_mps`/`OCRAM`/`frequency`のいずれにも影響されない**(コンパイル後の決定論的ハードウェアマッピングのため)。これにより、量子化ポリシー(固定)と`num_aces`/`n_mps`(探索対象)を独立に扱える。

BEVFormer-Tinyは[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)で確認済みの通り、アナログ側(ResNetバックボーン+FPN)がMAC数の約98.3%を占め、処理時間でも85.5%を占める。デジタル側(Transformerエンコーダ/デコーダ+検出ヘッド)はMAC数1.7%だが処理時間14.5%(MatMul/LayerNorm/Softmax/GridSampleのシーケンシャル処理のため)。したがって`num_aces`(アナログコア数)は処理時間の大部分を占めるアナログ側に直接効き、latency制約への影響が大きい。デジタル側のlatencyは主に`frequency`やコンパイラの配置最適化で効くため、`num_aces`探索とは直交する。

---

## 4. 探索アルゴリズムと結果

### Stage B: A_max以内の`num_aces`候補それぞれについて、latency制約を満たせる構成を特定

**`num_aces`の候補は連続値ではなく、SDK同梱の`funcsim`が対応する4段階(24/32/48/72)のみ**(`--amp-arch m2024`/`m2032`/`m2048`/`m2072`)。ただし**実機で試したところ、このうちコンパイルが実際に通るのは48と72の2点のみ**であることが判明した([HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §1):

| `num_aces` | コンパイル結果 |
|---|---|
| 24 (`m2024`) | **失敗**。CP-SAT段まで到達するが122分後に`status: UNKNOWN`→`L0 optimization failed`(exit 1)。ACE Weights 31.25M。24 ACE=6タイルへの充填が実用時間内に解けない(高充填領域でCP-SATがタイムアウト、INFEASIBLE証明ではない)。SKU候補としては利用不可 |
| 32 (`m2032`) | **利用不可**。`dnn_compiler`バイナリに`m2032`のターゲット定義自体が存在しない(strings/逆アセンブル調査で確認、定義済みは m2024/m2048/m2072 のみ)。抜け道候補の`--boreas-a-compatible`(compatibility mode トグル)を実機で試したが、compatibilityモードにしても中断する(エラーが`without`→`with`に変わるだけ)。[HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §1参照 |
| 48 (`m2048`) | **成功**(~94分、exit 0)。**PPA推定完了: Combined latency 43.47ms → 33ms制約に対し10.47ms超過でNG**(下記) |
| 72 (`m2072`) | **成功**(既知の可行点、§2参照)。Combined latency 31.58ms(33ms制約OK、マージン4.3%) |

**したがって本探索でコンパイル可能な`num_aces`候補は48と72の2点のみに確定**——24はCP-SATが実用時間内に解けず、32はターゲット定義がない。24について仮に無理やり通しても、約84%の高充填で並列化の余地がほぼ無く、並列化を削ればlatencyが悪化して33ms制約(72 ACEですらマージン4.3%)を満たす見込みがほぼゼロなので、いずれにせよSKU候補にならない。

#### 48 ACEs PPA推定結果(確定)

| 項目 | 48 ACEs (m2048) | 72 ACEs (m2072) |
|---|---|---|
| Analog NPU Processing Time(6カメラ込) | 38.85 ms | 26.95 ms |
| Digital Estimated Frame Processing | 4.63 ms | 4.63 ms |
| **Combined Latency(6カメラ直読)** | **43.47 ms** | 31.58 ms |
| 33ms制約判定 | **NG(10.47ms超過)** | OK(マージン1.42ms) |
| ACE Utilization | 72.26% | 69.43% |
| Number of ACEs(正表示) | 48 | 72 |

**判定: 48 ACEsは33ms制約を満たさない(43.47ms)。72 ACEsが唯一の可行SKU。** アナログ側処理時間が72→48で26.95→38.85 msに増加(コア数減少で並列度が下がるため。§0の予測通り)。

補足(実測で確認した推定ツールの挙動、結論に影響しない):
- **fpsクランプ**: `--power-inference-rate 30`指定に対し、m2048は23.00fps(=1/43.47ms)しか出せないため`Power inference rate of '30' exceeds estimated. Using estimated rate of '23' instead.`で23fpsにクランプされる。クランプ後のpower(3.428W)は**23fps基準**であり、72の30fps基準4.505Wとfps基準が異なるため直接比較できない(§4.1 表の注記)。ただしlatency判定はpower推定と独立に算出されるため、この判定(43.47ms>33ms)には影響しない。`--allow-fps-over-max`で30fps強制推定も可能。
- **Digital Estimated Frame Processing一致(4.63ms)**: 48/72でデジタル側の構成(288コア)・処理が同一のため、デジタル処理時間が完全に一致する。`num_aces`がデジタル側に影響しないという§3の前提の実測裏付け。

**この段階の出力は「area(=num_aces)を横軸に取ったときの、各areaで実現可能な最小latency/最小powerの表」である。** `num_aces=72`未満の全SKU候補が33ms制約を満たせない/コンパイル自体が通らないため、それは「マッピングの工夫では解決できない、area上限(A_max)そのものの見直しが必要」という重大な結論になる(§0参照)——`num_aces=72`(380mm²)が採用可能なA_max内かどうかが唯一の判断材料になる。24/32が脱落した今、**実質的な最終判断は「48で足りるか、72が必要か」の2択**に単純化されており、48が不可行のため**72が必須**と確定した。

### 4.1 die area(面積)の扱い

**面積は本探索の中核変数(§0)。v26.05.2 の `mythic-ppa-estimators` は die area 値を出力しない**(旧版が出していた `Estimated Die Area to Achieve Estimated Processing Time` 行は撤去された)。代わりに `Number of ACEs` が正しく表示される。

したがって**SKUの面積比較には、データシート由来の物理傾き 5.278 mm²/ACE(=380/72)を使う**:

| `num_aces` | 物理ダイ面積 |
|---|---|
| 24 (m2024) | 158 mm²(=5.278×24) |
| 48 (m2048) | 253 mm²(=5.278×48) |
| 72 (m2072) | 380 mm²(=5.278×72、既知・確定値) |

これにより §3 の「areaは`num_aces`のほぼ単調増加」が物理的に正しく成立する(24<48<72 で 158<253<380)。詳細な背景は [[ppa-die-area-not-physical]] を参照。

### Stage C: area-powerのトレードオフ表

Stage Bで得た「各`num_aces`候補(area) × その最小latency/power」の対を area昇順で並べる。面積列は §4.1 の物理傾き 5.278 mm²/ACE で算出。全て6カメラネイティブの直読値。

**完成したトレードオフ表(確定版):**

| `num_aces` | 物理ダイ面積(§4.1) | Combined Latency(6カメラ直読) | 33ms制約 | ACE Util | Total Power | 採否 |
|---|---|---|---|---|---|---|
| 24 (m2024) | 158 mm² | — | — | — | — | **コンパイル不可**(§4、CP-SAT `status:UNKNOWN`、122分で失敗) |
| 32 (m2032) | — | — | — | — | — | **利用不可**(ターゲット定義なし、バイナリに m2024/m2048/m2072 のみ、§4) |
| 48 (m2048) | **253 mm²** | **43.47 ms** | **NG(10.47ms超過、31%オーバー)** | 72.26% | 3.428 W @**23fps**※ | **不可行** |
| 72 (m2072) | **380 mm²** | **31.58 ms** | **OK(マージン1.42ms=4.3%)** | 69.43% | 4.505 W @30fps | **唯一の可行SKU** |

※ m2048は23fpsしか出せず、`--power-inference-rate 30`指定に対しツールが23fpsへ自動クランプ。したがって3.428Wは**23fps基準**で、72の4.505W(30fps基準)とfps基準が異なる。48が不可行のためpower直接比較は行わない(可行でない以上、power比較は無意味)。

**結論(area軸): 33ms制約を満たすのは72 ACEs(物理380mm²)のみ。48 ACEs(253mm²)以下は全滅。** これは「マッピングの工夫では埋められない、area上限A_maxそのものの問題」(§0)であり、**A_maxが380mm²以上でなければ、現行SDK・現行モデルでは可行なSKUが存在しない**という重大な結論。面積を削る方向の余地はゼロで、A_maxの見直しか、モデル側の軽量化(アナログ側処理時間の短縮=48の43.47msを33ms以下に約24%短縮)が次の検討事項になる。

### 探索コスト(実績)

`num_aces`候補は48と72の2点(24/32は利用不可、上記参照)。両者ともPPA推定まで完了済み(72=31.58ms、48=43.47ms)。Stage B/Cは完了。`n_mps`のYAMLキーが未確認のため(§6参照)、`n_mps`側の追加探索は実施していない——ただし面積の主変数`num_aces`で既に「72のみ可行」が確定しており、`n_mps`探索は48以下を救済できる見込みが薄い(48は33ms制約に対し10.47ms=31%超過しており、`n_mps`調整程度では埋まらない差)。

**実測所要時間(参考):**
- コンパイル: m2072 ~108分、m2048 ~94分、m2024 ~122分(失敗)、m2032 即失敗(定義なし)。
- ppa-estimator(funcsim, CPU専用シングルコア): m2072/m2048 とも ~4時間(6カメラ)。**GPU不可**(funcsimはCPU逐次イベントドリブン)。
- 総ウォールクロック短縮策として、期限なしコンテナ(`sleep infinity`)を別立てし、複数SKUのコンパイル/推定を並列実行した(MYTHIC_ROOTはコンパイルごとにユニークなサブディレクトリを作るため並列衝突なし)。

---

## 5. 使用するコマンド/ツール(参照)

Stage B/Cで実際に使うのはユーザー向けCLIの2つのみ(内部Python APIを直接呼ぶ必要はない、[HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md)参照):

- コンパイル: `mythic-compiler --input-artifact <trained.tar.gz> --compiler-config <yaml> --output-artifact <out.tar.gz>`。`num_aces`はYAML内`COMPILER_OPTIONS`の`--amp-arch`で指定(§4参照)。実行には`MYTHIC_ROOT`・`MYTHIC_COMPILER_DOCKER_TAG`環境変数の設定が必須(HOWTO §0.2.1)。入力は既存の`bevformer-tiny-1600x900-trained.tar.gz`(再学習不要)。
- 性能・電力計測: `mythic-ppa-estimators --estimate-performance --estimate-power --power-inference-rate <fps> <compiled.tar.gz>`。6カメラネイティブのため`--power-inference-rate 30`で1回実行する(HOWTO §2)。

---

## 6. リスク・未検証事項

1. PPA推定値は相対比較専用。Leakage/Clock Tree/PCIe/D2D電力が未算入のため、絶対的な面積・電力のサインオフには使えない(§1.2)。
2. マルチスレッドソルバーの非決定性: 同一設定でもコンパイル結果がわずかに変わることがあるため、境界に近い`num_aces`候補は複数回コンパイルして再現性を確認するのが望ましい。
3. A_max(area上限)の具体的な数値が未確定。本計画は「A_maxが与えられれば可行性判定→power最小化を行う」という手順を定義したものであり、A_max自体の値はビジネス側(コスト・パッケージング制約)から別途確定させる必要がある。既知の可行点(72 ACEs, 380mm²)がA_max以内かどうかを確認する必要がある。
4. **`n_mps`に対応するYAMLキーがBEVFormer既定のCompiler config内に見当たらない**([HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §4)。判明していないため`n_mps`側の探索は未実施(`num_aces`のみの1次元探索で完了)。
5. **本物のnuScenesデータセットは`s3://s3-srdm/nuScenes/`に配置されているが、現状は850シーン中96シーン相当分のみ(`v1.0-trainval01_blobs.tgz`、10パート中の1パートのみ)。** さらにそのうち11シーンはパート内でも一部サンプルのLiDAR/カメラファイルが欠落しており、実際に完全な形で使えるのは**85シーン**(標準train/val splitでtrain 62 / val 23相当)。メタデータ(`v1.0-trainval_meta.tgz`)は850シーン全部を含むため、`nuscenes_converter.get_available_scenes`のファイル存在チェック(`mmcv.is_filepath`で文字列型チェックのみ、実ファイル存在は確認しない)を素通りしてしまう——annotation生成(`nuscenes_data_prep`)を実行する前に、`sample_data.json`の`filename`が実際に存在するかを事前チェックし、欠落のあるシーンをメタデータ側で除外してから流す必要がある(実際に85シーンでの生成・推論は動作確認済み)。データセットは今後拡充される予定(ユーザー確認済み)。**なお本計画のStage B/C自体はnuScenesデータに依存しないため(§2)、このデータ状況は直接の障害にはならない。**
6. **BEVFormerのBEV可視化(`bevformer_inference.py`のBEVパネル)には、検出ボックスの描画がマップ・自車マーカーに対して90°ズレる既知の表示バグがある**(`bevformer_lib/custom_utils/visualization.py`の`_layer_boxes`が`_layer_ego`/実際に使われる`_layer_map_raster`と異なる座標変換規則を使っている可能性)。本物のnuScenesデータ(scene-0003)で`--map-bev`付きの動画を生成した際に発見。**これは可視化(見た目)のみの問題で、PPA数値や精度指標には影響しない**——本計画(Stage B/C)の判断に影響しないため、対応は任意。
7. **複数`num_aces`候補の並列実行**: `mythic-compiler`/`mythic-ppa-estimators`はDocker-out-of-Docker(`docker.sock`共有)で子コンテナを起動するが、`MYTHIC_ROOT`がコンパイルごとにユニークなサブディレクトリを作るため、複数SKUの同時コンパイル/推定で衝突しないことを実機で確認済み(本探索でも並列実行して総時間を短縮した)。

---

## 7. 次のアクション

探索本体(Stage B/C)は完了。残る検討事項は以下:

- [ ] A_max(area上限)の具体値をビジネス側から確定し、可行点は72 ACE(380mm²)のみのため**A_maxが380mm²以上か**を確認(§6-3)。380mm²未満なら現行SDK・現行モデルでは可行SKUなし(§4結論)
- [ ] モデル側の軽量化(アナログ側処理時間の短縮=48の43.47msを33ms以下に約24%短縮)で48を救済できるかの検討(§4結論)
- [ ] `n_mps`に対応するYAMLキーの有無を確認(§6-4、未着手)。ただし48救済の見込みは薄い
- [ ] 同area内でのlatency-powerトレードオフ軸の検討: `dnn_compiler --help`に、`num_aces`を固定した上でチップ内の最適化目標を調整するフラグがある(`--target-frame-rate`(`-f`、既定300fps)、`--parallelism-multiplier`(既定1.0)、`--auto-parallelize`(既定0.0=オフ)、`--opt`(default/slow/fast))。PLAN §0 の area-power トレードオフとは異なる次元(同area内のlatency-power)。必要なら次の探索軸として検討する

### 実行環境の状態(セッション引き継ぎ用)

- **コンテナ**: `mythic_ppa_explore_2605_2b`(`sleep infinity`、期限なし)。稼働中。再作成手順は [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §0.2.1(`MYTHIC_ROOT=/tmp/ppa_mythic_root_2605_2`, `MYTHIC_COMPILER_DOCKER_TAG=v26.05.2`が必須)。**コンテナ内に直接作成したファイル(Compiler config YAML等)はコンテナ再作成で消える**——コンテナ内 `$MYTHIC_SDK_ROOT/mythic-model-zoo/configs/bevformer/compiler/` に m2072既定を各archへ複製した YAML を再作成する必要がある。
- **ホストの作業ディレクトリ**: `/tmp/ppa_workspace_2605_2/out/` に artifact を保存済み。
  - `bevformer_m2072_high_2605_2.tar.gz` — 72 ACEsのコンパイル済みartifact(PPA推定済み)
  - `bevformer_m2048_high_2605_2.tar.gz` — 48 ACEsのコンパイル済みartifact(PPA推定済み)
  - `bevformer_m2072_high_2605_2_ppa_*.tar.gz` / `bevformer_m2048_high_2605_2_ppa_*.tar.gz` — 各PPA推定結果(推定ログ付き)

---

## 参照

- [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) — 本計画で使うツール(`mythic-compiler`/`mythic-ppa-estimators`)の具体的な使い方
- `doc/user-guides/GEN2 User Guide.pdf` — 公式ユーザーガイド。3ツールの正式なエントリーポイント
- `doc/reports/Model Summary Report.pdf`(v3.0) — 全モデルの精度とPPAサマリの出典。BEVFormer-Tinyの精度・latency・power実測値(§2)の一次出典
- `doc/reports/Compiler Optimization Report - BEVFormer-Tiny.pdf`(v1.2) — `num_aces=72`構成のコンパイラ/PPA推定コマンドと詳細レポート(ACE/Weight/SRAM utilization)の出典
- [00_overview.md](00_overview.md) — SDK全体構成
- [01_compilation.md](01_compilation.md) — コンパイルフロー、`n_mps`/`OCRAM`/量子化ポリシーの詳細(§3.2, §3.3, §3.4)
- [02_ppa_estimation.md](02_ppa_estimation.md) — PPA推定式・定数・未算入項目の完全解析
- [03_accuracy_simulation.md](03_accuracy_simulation.md) — munc/BCM精度シミュレーションの仕組み
- [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) — BEVFormer-Tinyのanalog/digital演算振り分け実測値、推論パイプラインの実行手順
