# 計画: BEVFormer-Tiny PPA(面積-電力トレードオフ探索)パラメータ探索

状態: **計画のみ(未実装)**。本ドキュメントは実験を実行する前の設計書であり、コード実装は含まない。

## 0. 目的と制約

**前提(重要)**: 実チップは未確定であり、本探索は**運用時のチューニングではなく SKU 選定(どの ACE 数でテープアウトするかの事前検討)に近い**。したがって `num_aces` は「既に製造済みのシリコンのうち何タイル使うか」という運用選択ではなく、**そのままダイ面積・製造コストを決める設計変数**として扱う。

**area と power の間には本質的なトレードオフがある**: `num_aces`を減らせばareaは下がるが、同じ演算量をより少ないタイルで処理するためlatencyが増える方向に働き、latency制約(33ms)を満たすために並列化・クロック等で補うとpowerが増える方向に働く場合がある。逆に`num_aces`を増やせばarea/コストは上がるが、余裕を持ってlatency制約を満たせ、無理な高稼働率を避けられるためpowerが下がる場合がある。**area は「小さいほど望ましい」という真の目的変数であり、power も同様に「小さいほど望ましい」目的変数**——両者は単純な優先順位(area優先→powerでタイブレーク)には落とし込めず、**2目的のトレードオフ(Paretoフロンティア)として同時に探索する**必要がある。

`A_max`(コスト・パッケージング等の事業制約からの上限)は、この2目的探索の**候補空間を絞るカットオフ**として使う(area > A_maxの候補はそもそも検討対象外にする)。

これを踏まえると、問題は以下のように再定式化される:

> **精度制約・レイテンシ制約・area上限(A_max)制約を満たす設定の中から、area-power の Pareto最適点(どちらかを改善するには他方を犠牲にせざるを得ない点の集合)を求め、その中からユーザーが選択する。**

もし `area ≤ A_max` の範囲内でどう組んでも精度・レイテンシ制約を満たせない場合、それは「量子化やマッピングの工夫では解決できない、area上限そのものの見直しが必要」という重大な結論になる。したがって本探索の最初のマイルストーンは「Pareto探索」の前に「**A_max以内での可行性確認**」である。

**問題設定**:

| 項目 | 内容 |
|---|---|
| 制約1(area) | `num_aces` に対応する die area ≤ A_max(具体値は未確定 — §5に記載、候補空間のカットオフとして使用) |
| 制約2(精度) | FP32モデルからの精度劣化なし(mAP/NDS等の具体的しきい値ではなく「FP32同等」を基準とする) |
| 制約3(レイテンシ) | 1フレームあたり ≤ 33ms |
| 目的関数 | area・power の2目的同時最小化(Pareto最適点の集合を求める。一方を優先して他方をタイブレークにするのではない) |
| 量子化ポリシーの粒度 | ブロック単位の粗い候補セット(層単位の探索は今回スコープ外) |

これは「精度・レイテンシ制約下で、area上限(A_max)以内の候補群についてarea-powerのPareto最適点を求める」という多目的最適化問題であり、全探索ではなく変数の依存関係を利用した段階的探索で解く。

**SKU選定という位置づけの実務上の意味**: 実チップ設計では `num_aces` は連続変数ではなく、量産化する品種数(コスト)を考えると数値候補は絞られる(例: 16/24/32など少数のSKU案、いずれもA_max以内)。したがって Stage B の「`num_aces` を振る」探索は、無数の値を試す最適化ではなく、**A_max以内の少数の現実的なSKU候補群について、それぞれの実現可能なpower/latencyを計測し、area-powerのトレードオフ表を作る**作業として設計する。最終的にどの点を選ぶか(area優先かpower優先か)はステークホルダーの判断に委ねる。

---

## 1. 前提: SDKが実際に提供する機能(調査結果のまとめ)

### 1.1 ユーザー側(Python)から触れるPPA関連パラメータ

| パラメータ | 場所 | 値域 | 効き先 |
|---|---|---|---|
| `num_aces` | `power_estimator.py:619`, `perf_analysis.py:92-97,113-117` | 4の倍数、既定24 | die area(単調)、ACEクリティカルパス |
| `n_mps` | `vnnmap/compilation.py:55-57`, `run_vnnmap.py` | `{1,4,8}` のみ | タイル分割・並列度 |
| `OCRAM1` / `DDR` | `run_vnnmap.py:14-19`(`system.cfg`) | 既定2MB/512MB | DDR↔OCRAM昇格、latencyへの影響 |
| `frequency` / `DDRConfig` | 同上 | 既定625MHz/100 | latency全体。CLIフラグなし、`system_config`直書きが必要 |
| `QuantizationConfig.tensor_n_bits` | `vnnort/quantizer/quantization_config.py` | テンソル/ブロック単位のビット幅上書き | 精度⇄PPAの主軸 |

**露出されていないもの**: ADC/DAC bit深度(8bit固定, [pow:17-22])、sparsity、BitSpreadingMode(クローズドソースの`dnn_compiler`/`vnnmap`内部で決定)。

### 1.2 計測ツールと既知の限界

**[訂正・確定 — `doc/user-guides/GEN2 User Guide.pdf`で確認]** ユーザーが実際に叩くエントリーポイントは3つのみ(`mythic-compiler`/`mythic-ppa-estimators`/`convert_model.py steps=eval_trained`)。以下の`perf_analysis.py`/`power_estimator.py`は、公式CLI`mythic-ppa-estimators`の内部実装(この抽出済みリポジトリで直接読める部分)であり、ユーザーは個別に叩く必要はない。ツールの具体的な使い方は[HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md)を参照。

- `perf_analysis.py`: HDF5トレースからlatency/fps/ACE利用率/die areaを算出([doc 02_ppa_estimation.md](02_ppa_estimation.md)に式レベルで解析済み)。
- `power_estimator.py`: L0 protobufからpower/TOPS-Wを算出。
- **[確定] Leakage/Clock Tree/PCIe/Die-to-Die電力は算出はされるが最終値`total_power`には未算入**(`power_estimator.py` 該当コードはコメントアウト、doc 02 §4.11。**GEN2 User Guideの実行ログにも同旨のNOTE有り、公式ドキュメントでも確認**)。したがって本探索で得られる数値は**絶対値のサインオフではなく、同一条件下での相対比較(A/B)専用**として扱う。
- `munc`/BCM/TorchNet: 精度(mAP/mIoU等)専用シミュレータ。PPAは出さない。量子化ノイズ・ADC特性を含む。実際のCLIエントリーポイントは`convert_model.py steps=eval_trained`。

**[新規確認・重要] BEVFormerの6カメラ換算**: GEN2 User Guide §2(Changes in SDK 26.05)に明記——「Mythicの機能シミュレータの制約により、性能推定は単一カメラ入力でのみ実行可能。最終latencyは単一カメラのlatencyを6倍(6カメラ分)して推定する必要がある」。したがって`mythic-ppa-estimators`が出す`Combined Analog + Digital NPU Latency`の値をそのまま33ms制約と比較してはならず、**アナログ側の値を6倍してからデジタル側と合算**する必要がある(アナログ側=1カメラ換算・デジタル側=6カメラ換算という非対称性は[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)のMAC比率分析でも既に確認済みの事実と整合)。Stage Bの可行性判定はこの換算を必ず適用すること。

**[新規確認・重要] `num_aces`はPPA Estimator側でなくCompiler側で変える**: `mythic-ppa-estimators --help`の出力にはpositional引数(artifact path)以外に`num_aces`をオーバーライドするフラグが見当たらない([未検証]、実機で全文確認が必要)。実行ログの`Number of ACEs: 24`はコンパイル済みartifactに埋め込まれた値をそのまま表示していると見られる。つまり、**Stage Bで`num_aces`を振るには、`mythic-compiler`のコンパイル条件(`--compiler-config`のYAML)を変えて複数のartifactを作り分ける必要がある**——前提としていた「PPA estimator実行時に`--num-aces`を指定して再計算する」という想定は誤りだった可能性が高い。これはStage Bのコスト見積り(コンパイルの再実行回数)に直接影響する。

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

## 2. 探索変数の分解(なぜ全探索でなく段階的探索が可能か)

前述の変数群には都合の良い分離構造がある。

- **精度に影響するのは実質`tensor_n_bits`(量子化)のみ**。`num_aces`/`n_mps`/`OCRAM`/`frequency`はハードウェアマッピングの構成なので、`munc`の精度シミュレーション結果には影響しない(コンパイル後の決定論的写像)。
- **areaは`num_aces`のほぼ単調関数**(`ACE_TILE_AREA_PS_MM2 * num_aces/4`, [02_ppa_estimation.md](02_ppa_estimation.md) §5)。ビット幅を変えてもarea推定はほぼ変わらない。
- **latencyは`num_aces`/`n_mps`/`frequency`/`OCRAM`(スピル有無)/ビット幅(ADCサイクル数)/コンパイラソルバー設定の全部に依存**。
- **powerは上記全部に依存**(ACE動作電流はビット幅・稼働率次第、SRAMアクセス電力はメモリ構成次第)。

この分離により、「量子化ポリシー(accuracy軸)」→「num_aces(area軸、A_max以内のSKU候補)」→「power比較(トレードオフ表作成)」という3段階に問題を分解できる。**Stage B は「A_max以内で精度・レイテンシ制約を満たせるSKU候補は何か」の可行性判定**であり、Stage C はその可行なSKU候補それぞれのpowerを並べて**area-powerのトレードオフ表(Pareto点の集合)**を作る。「どちらを優先するか」の決定はステークホルダー判断に委ねる——これはStage Cの中で自動的に一意の答えを出す最適化ではない。

### 2.1 BEVFormer-Tinyのモデル構造を踏まえた量子化ポリシーの重点

[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) §「補足: analog/digital演算振り分け」で確認済みの通り、BEVFormer-Tinyは:

- **アナログ側(ResNetバックボーン + FPN)がMAC数の約98.3%を占める**が、処理時間では27.30ms(全体の85.5%)
- **デジタル側(Transformerエンコーダ/デコーダ + 検出ヘッド)はMAC数1.7%だが、処理時間4.63ms(14.5%)** — MatMul/LayerNorm/Softmax/GridSampleなどシーケンシャル処理が多いため

したがって、**量子化ポリシーの候補はアナログ側(ResNet+FPN)のビット幅を主軸に設計する**。デジタル側はACE(アナログコア)に載らないため、`num_aces`/area探索とは直交する(デジタル側のlatencyは主に`frequency`やコンパイラの配置最適化で効く)。

---

## 3. 探索アルゴリズム(3段階)

### Stage 0: ベースライン計測

FP32モデル(`bevformer-tiny-fp32-1600x900.onnx`)の精度を基準値として記録する。既定の`tensor_n_bits`(8bit活性化/8bit重み per-channel/16bit bias、[01_compilation.md](01_compilation.md) §3.2.2)でコンパイルした`bevformer-tiny-1600x900-trained.onnx`のPPAも記録し、「現状の設定がどの程度制約に対して余裕があるか」の起点とする。

### Stage A: 量子化ポリシー候補を精度制約でフィルタ

ブロック単位の粗い候補セット(数〜十数パターン)を用意し、`munc`/BCM/TorchNet精度シミュレータ(既存の推論パイプライン、[FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md)で動作確認済みの経路)に通す。「FP32からの精度劣化なし」を満たさないポリシーを落とす。

**[確定 — 計画修正]** `QuantizationConfig.tensor_n_bits`(`vnnort/quantizer/quantization_config.py`)は `list[tuple[str, int]]` で、各要素は「既存テンソル名」と「ビット数」の組。`vid_quantizer.py:176-182` が明示的に **`n_bits`は8か16以外を許可しない**(`ValueError: "Currently only 8 and 16 bits are supported."`)ことを確認済み。したがって当初想定していた6bit/4bitへの低減は**現行SDKでは不可能**——量子化ポリシーの軸は「どのテンソルを既定8bitから16bitに上げるか(=精度を上げる代わりにコストが増す方向)」のみであり、8bit未満への削減による省コスト化はできない。

候補ポリシー案(粗粒度、ブロック単位。8/16bitの二値のみ):
1. 既定(全テンソル既定値: 活性化8bit/重み8bit per-channel/bias・LayerNorm scale以外16bit) — ベースライン
2. ResNet浅層(low-level特徴、量子化誤差の影響が出やすい層)の活性化を16bitに上げ、精度余裕を確認
3. FPN(img_neck)出力を16bitに上げ、Transformer側への特徴伝播の精度を確認

※ この軸では「精度を落として省コスト化する」方向の探索ができないため、Stage Aの主目的は「量子化ポリシーの選別」ではなく「既定設定がFP32同等精度を満たすか、満たさない場合どのテンソルを16bitに上げれば満たすか」の確認になる。省コスト化(area/power低減)の主軸は、この後のStage Bで振る`num_aces`/`n_mps`である。

### Stage B: A_max以内の`num_aces`候補それぞれについて、latency制約を満たせる構成を特定

`num_aces`の候補は、A_maxに対応する上限値以下の少数のSKU案(例: 16/24/32など)に絞る(§0のSKU選定の実務上の意味を参照。無数の値を試す最適化ではない)。各量子化ポリシー × 各`num_aces`候補について:

1. コンパイラの内部ソルバーに「ACE利用率を最大化」させる設定(`--relative-objective-target`を高め、必要なら`--optimization-effort-limit`も増加 — §1.3の注意点の通り未検証フラグのため、実装時に実際にコンパイルへ渡して受理されるか確認する)でコンパイルする。
2. `perf_analysis.py`実測のlatencyが33ms以下かを判定する。事前スクリーニングには`explore_model()`(単発・軽量)を使い、有望な範囲に絞ってから本コンパイル+`perf_analysis.py`計測を行う2段階でコスト削減する。
3. `n_mps ∈ {1,4,8}`も同様に振り、`num_aces`との組み合わせでlatency制約を満たす構成を特定する(小さい格子なので全探索可能)。同じ`num_aces`でも複数の`n_mps`が制約を満たす場合、その中でpowerが最小になる`n_mps`を選ぶ(=各`num_aces`ごとに1つの代表構成に絞る)。

**この段階の出力は「area(=num_aces)を横軸に取ったときの、各areaで実現可能な最小latency/最小powerの表」である。** 全SKU候補が33ms制約を満たせない場合、量子化やマッピングの工夫では解決できない可能性が高く、A_maxの見直しをステークホルダーに提起する必要がある(§0参照)。

### Stage C: area-powerのトレードオフ表(Pareto点)を作成

Stage Bで得た「各`num_aces`候補(area) × その最小power」の対を、量子化ポリシーごとに並べ、**area昇順でpowerも確認し、Pareto最適点(あるnum_aces候補より小さいareaかつ同等以下のpowerを実現する候補が他にない点)を抽出**する。同じ量子化ポリシー内でも、量子化ポリシーをまたいでも(例: 軽い量子化+小さいnum_acesの組が、重い量子化+大きいnum_acesの組よりarea/power両方で優れる場合がある)Pareto比較を行う。

出力は単一の「最終候補」ではなく、**area-powerのトレードオフ表**(例: 「16 ACEならXW/33ms、24 ACEならYW/28ms、32 ACEならZW/20ms」)。どの点を選ぶかはコスト(area)と電力予算のどちらを重視するかというビジネス判断であり、本探索のスコープは判断材料を揃えることまでとする。

### 探索コスト

全探索(量子化ポリシー数 × num_aces候補数 × n_mps候補数)ではなく、量子化ポリシー数 × A_max以内のnum_aces候補数(少数のSKU案に限定) × 3(n_mps) 程度のコンパイル/計測回数で済む。

---

## 4. 使用するコマンド/ツール(参照)

- 精度シミュレーション: `bevformer_inference.py torchnet ...`([FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) 実行例を参照)
- コンパイル: `mythic.model_deployment.rmcr.compile.compile_artifact` → `dnn_fw_compile`→`dnn_compile`(`_extracted_sdk/rmcr/compiler.py:118`)。`cfg.COMPILER_OPTIONS`に`--relative-objective-target`等を追加できるか実装時に検証。
- 軽量事前スクリーニング: `vnnmap.exploration.explore_model()`(`n_mps`/`ocram1_bytes`を指定して単発推定)
- 性能計測: `perf_analysis.py --hdf5-path perf_trace_dump.h5`
- 電力計測: `power_estimator.py`(L0 protobuf入力)

---

## 5. リスク・未検証事項

1. **`--relative-objective-target`等のコンパイラフラグの実在・正確な挙動は社内資料(Mythic Compiler Guide)由来の未検証情報**。このリポジトリの抽出済みコードにはHydra YAML実体が含まれておらず、直接確認できていない。実装開始時に、実際のコンパイルコマンドにフラグを渡してエラーにならないか、`--help`相当の出力があるか確認する必要がある。
2. **PPA推定値は相対比較専用**。Leakage/Clock Tree/PCIe/D2D電力が未算入のため、絶対的な面積・電力のサインオフには使えない(§1.2)。
3. **コンパイラ内部のACE利用率モデルと`perf_analysis.py`実測値は一致しない場合がある**(社内資料の注記)。Stage Bの最終判定は必ず実測値で行う。
4. **マルチスレッドソルバーの非決定性**: 同一設定でもコンパイル結果がわずかに変わることがあるため、境界に近い`num_aces`候補は複数回コンパイルして再現性を確認するのが望ましい。
5. **[解消・確定]** 量子化ビット幅は`vid_quantizer.py:176-182`のチェックにより**8か16のみ**(6/4bitは`ValueError`で拒否される)。§3.1のStage A候補を8/16のみに修正済み。
6. **A_max(area上限)の具体的な数値が未確定**。本計画は「A_maxが与えられれば可行性判定→power最小化を行う」という手順を定義したものであり、A_max自体の値はビジネス側(コスト・パッケージング制約)から別途確定させる必要がある。Stage B着手前に確定していることが前提。
7. **[新規]** `mythic-compiler`の`--compiler-config`YAML内で、量子化ポリシー(Stage A)・`num_aces`/並列化(Stage B)に対応する実際のキー名が未確認(§1.2)。実機でYAMLファイルの内容を確認するのが実装着手前の最優先タスク。
8. **[新規]** BEVFormerの性能推定は単一カメラ入力でのみ実行可能で、6倍換算が必要(§1.2)。この換算を誤ると33ms制約の判定を誤る。
9. **[新規]** `num_aces`はPPA Estimator実行時ではなく、Compiler(`mythic-compiler`)のコンパイル条件側で変える必要がある可能性が高い(§1.2)。これが確定すると、Stage Bの探索コスト(§3「探索コスト」の見積り)は「`num_aces`候補ごとに`mythic-compiler`を再実行する」前提に修正が必要。
10. **[前提・更新]** 本探索は**本物のnuScenesデータセット・CAN bus拡張データ・annotation**(いずれもユーザー側で用意)を使う想定に確定。CARLA代替パスは使わない。GEN2 User Guideには「nuScenesはBEVFormer training guideの手順でannotationを生成する必要があるが、**Collaboration Chamber環境では**ディレクトリ権限の問題でMythicがannotation生成できていない」という既知のブロッカーが明記されている(§8.3.5)が、**これはCollaboration Chamber固有の制約であり、本探索(下記11参照、AWS環境でdocker直接操作)では直接は関係しない**。実装着手時に、annotation生成(`nuscenes_infos_temporal_{train,val}.pkl`)がこのAWS環境上で問題なく完了することを確認するのが最初の検証項目。
11. **[新規]** 本探索はCadence Collaboration Chamber(SLURM経由のノード確保)を使わず、このリポジトリのあるAWS環境上でdocker直接操作により実施する。GEN2 User Guideの手順(§4-8)はCollaboration Chamber前提のため、`DATASET_DIR`/`TRAINING_MODELS_HOST_DIR`等のパスをこの環境向けに読み替える必要がある(具体的な読み替えは[HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) §0参照)。本物のnuScenesデータセット・CAN bus拡張データ・annotationはユーザー側で用意し、このAWS環境からアクセスできる場所に配置する。

---

## 6. 次のアクション(実装時のチェックリスト)

- [ ] A_max(area上限)の具体値をビジネス側から確定(§5-6)——本計画の前提条件
- [ ] このAWS環境上でSDKコンテナを起動し(`DATASET_DIR`/`TRAINING_MODELS_HOST_DIR`を本環境向けに上書き)、本物のnuScenes+CAN bus拡張データでannotation生成が問題なく通ることを確認(§5-10/§5-11)——実装着手前の前提確認
- [ ] `mythic-compiler`の`--compiler-config`YAMLを実機で開き、量子化ポリシー・`num_aces`に対応するキー名を確認(§5-7)——実装着手前の最優先タスク
- [ ] `mythic-ppa-estimators --help`全文を確認し、`num_aces`オーバーライドフラグが本当に無いか検証(§5-9)
- [ ] Stage 0: FP32ベースラインの精度・PPA計測を実行し基準値を記録(BEVFormerは6カメラ換算を適用、§5-8)
- [ ] `cfg.COMPILER_OPTIONS`に`--relative-objective-target`等を追加して実際にコンパイルが通るか検証(§5-1)
- [ ] `tensor_n_bits`のスキーマ制約を確認し、Stage Aの量子化ポリシー候補を確定(§5-5)
- [ ] Stage A: 候補ポリシーを`convert_model.py steps=eval_trained`(本物のnuScenesデータセット経由)で精度評価、FP32同等を満たすものを選別
- [ ] Stage B: A_max以内のSKU候補(`num_aces`/`n_mps`の組)ごとに、`mythic-compiler`で個別にコンパイルし、6カメラ換算後のlatencyが33ms制約下での最小power構成を特定。全候補が制約を満たせないならA_max見直しを提起
- [ ] Stage C: area-powerのトレードオフ表(Pareto点)を作成し、ステークホルダーへの判断材料として提示
- [ ] 境界付近の`num_aces`候補は複数回コンパイルして再現性を確認(§5-4)

---

## 参照

- [HOWTO_ppa_exploration_tools.md](HOWTO_ppa_exploration_tools.md) — 本計画で使う3ツール(`mythic-compiler`/`mythic-ppa-estimators`/`convert_model.py steps=eval_trained`)の具体的な使い方
- `doc/user-guides/GEN2 User Guide.pdf` — SDK v26.05.0公式ユーザーガイド。3ツールの正式なエントリーポイントとBEVFormerの6カメラ換算の記載元
- [00_overview.md](00_overview.md) — SDK全体構成
- [01_compilation.md](01_compilation.md) — コンパイルフロー、`n_mps`/`OCRAM`/量子化ポリシーの詳細(§3.2, §3.3, §3.4)
- [02_ppa_estimation.md](02_ppa_estimation.md) — PPA推定式・定数・未算入項目の完全解析
- [03_accuracy_simulation.md](03_accuracy_simulation.md) — munc/BCM精度シミュレーションの仕組み
- [FUTURE_bevformer_inference_run.md](FUTURE_bevformer_inference_run.md) — BEVFormer-Tinyのanalog/digital演算振り分け実測値、推論パイプラインの実行手順
