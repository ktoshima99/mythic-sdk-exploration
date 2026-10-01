# 08. BEVFormer 量子化・再学習の GPU メモリ要因解析

対象: BEVFormer-tiny 1600x900（SDK `v26.05.2`）の `to_structural → to_training → train → to_acm → create_artifact` パイプライン。「なぜ再学習に約45〜50GBのGPUメモリが必要なのか」を、コード解析と実測の両方から説明する。実測はホスト NVIDIA L40S（46GB）、コンテナ `mythic_bevformer_train`（イメージ `mythic-sdk-ubuntu-24.04:m2000-v26.05.2`）、nuScenes v1.0-mini（2 scenes 81 frames）で行った。

主張はファイルパス:行番号またはログ実測値を根拠に引用する。確定できない箇所は **[推測]** と明記する。

---

## 1. 結論の要約

- GPUメモリを食っているのは**重みではなく、backward 用に保持される中間活性**である。BEVFormer-tiny の重みは25M params（FP32で約0.1GB）しかない。
- 支配要因は2つの掛け算: **(a) 6カメラ×1600x900の高解像度入力**と、**(b) Mythicのアナログ量子化シミュレーション層（Denali separable model）が素のConv+ReLUの約4.5倍の活性を保持する**こと。両方が重なるbackbone（ResNet-50、`MythicConv2d`×74層）が主犯で、素のFP32学習なら約9.6GBで済む同じ構成が、Mythic化すると数十GBに膨らむ。
- SDKはこれを**activation checkpointing**で緩和する設計になっている。既定の45GBプリセットは48GB級GPU（L40S/A6000）に収まるよう調整された値であり、全層チェックポイントすれば約18GBまで落とせるが速度が落ちる。

---

## 2. パイプラインと量子化の所在

BEVFormerに `m2000.yaml` は存在しない（`m2000.yaml` を持つのは huggingface_classifiers / yolov8 / zero_dce / dummy のみ）。`configs/bevformer/bevformer_tiny.yaml` に `step_order` の指定が無いため、generic の既定順序（`configs/common/base_config_generic.yaml:37-50`）がそのまま使われる。README（`mythic/model_zoo/bevformer/README.md:182-322`）が案内する実際の手順は次の7段:

```
to_structural → to_training → train → post_retraining_simplification → eval_trained → to_acm → create_artifact
```

量子化は1ステップではなく3段に分かれる。

| ステップ | 役割 | GPU使用 | forward回数 | backward | メモリ支配要因 | 実測メモリ・時間 |
|---|---|---|---|---|---|---|
| **to_structural** | グラフ再構成のみ（量子化なし） | 不要 | 無し | 無し | ONNXロードのみ（約137MB） | 約1分（[03_accuracy_simulation.md](03_accuracy_simulation.md) §4.7.4） |
| **to_training** | 統計収集でレンジ・スケール(pFSR/iFSR/DSF)を確定し、ONNX上でMythic opに書き換え（(a)グラフ再構成+(b)統計確定+(c)fake-quant opへの変換、いずれも静的） | 統計収集TorchNetが既定でGPU | 有り、**STATS_REQUIREDなopごとに繰り返し**（本実測では1回の`to_training`実行中に5回、§4.2） | **無し**（`torch.no_grad()` 内） | **全エッジの中間活性がobserverにより保持される。解放されず回を跨いで積み上がる**（§4） | 本実測で45GiB付近でOOM（44.39GiB GPU）。既存ログでは43.90GiBでOOM（45GiB空きGPUでも）。`device_name=cpu`で回避、約35〜42分（03 §4.7.4、§4.6） |
| **train** | QAT本体。forwardのたびにfake-quant+ノイズを適用、STEでbackward | GPU（DDP対応） | 学習データ全体×24epoch | **有り** | **Mythic層の保持活性**（§3）。checkpointで調整可能 | 本ドキュメント §5 で実測 |
| **to_acm** | 重みを1/256格子に実丸め、BCMに変換。統計収集あり | GPU | 有り（`stat_n_samples_default: 200`） | 無し | 統計収集TorchNet（to_trainingと同じ仕組みでOOMしうる） | 本ドキュメント §5 で実測（約24GB、サンプル数10で約3分37秒） |
| **create_artifact** | 忠実度を`munc_digital`に固定し分割・packaging | 基本不要 | 原則無し | 無し | 小 | 短時間 |

- `train` の出力先は `trained_model_pre_cleanup`（`bevformer_tiny.yaml:137`）。`trained_model` を作る `post_retraining_simplification` は Dropout除去のみで `step_order` には入っていない。
- 量子化のうち「fake-quantの実行」自体はto_training後半の統計収集パス・train・eval_trainedいずれのforwardにも入っている（op自体はto_trainingでMythic opに書き換わっているため）。重みを実際の整数格子に丸めるのはto_acmのみ。

---

## 3. なぜ train が重いのか — Mythic層の保持活性

### 3.1 入力の形

- batch sizeはコード上1に固定（`GeneralizeBatchSize`が無効、`bevformer_tiny.yaml:to_training.conversion_parameters.ops.GeneralizeBatchSize.enabled: false`）。ただし1サンプル=6カメラなので、backboneの実効batchは6。
- 解像度は1600x900（pad後 928x1600）、ダウンスケールなし。800x450設定（デフォルト）に対して画素数は約4倍。

### 3.2 Mythic層が素のConv+ReLUより重い理由

`train` ステップは Denali separable model（`hydra_configs/training_model/denali.yaml` → `denali_training_model.yaml` で全非理想性を有効化）を使う。`MythicConv2d`（`BaseAnalogModel.forward`, `_ace_model.py`）の1層は、素のConv+ReLUに対して概ね以下を追加で保持する:

1. 符号付き入力の正負分離（`cat([clamp(X,0), clamp(-X,0)], dim=1)`）— signed-input層（stem convなど）でチャネル2倍。
2. 入力ノイズ適用後の多項式近似2項分、**畳み込みを2回実行**（各々が自分の入力をautograd用に保持）。
3. ADC（SAR近似）のoffset/noise/floor/clip。
4. DSF（trainableな場合あり）によるスケール+活性化clamp。

実測ベースライン比較（§5のF条件・A条件）で、Mythic化による保持活性の倍率は**約4.5倍**（9.6GB→43.5GB、ただし両者ともbackbone以外に同じFPN/transformer/headを含むため、backbone単体の倍率はこれより大きい）。

### 3.3 Activation checkpointing — 既定の45GBはL40S/A6000(48GB)向けの意図的な調整値

`TorchNet.forward` は指定パターンに一致する層を `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` でラップする（`_torchnet.py:461-462`）。BEVFormer 1600x900 の既定設定（コンテナ内 `configs/bevformer/model_setup/tiny_1600x900.yaml`）:

```yaml
checkpointed_torchnet:
  activation_ckpt_config:
    enable: true
    # ~45GB VRAM
    pattern: 'MythicConv2d::(n_ResNet_conv1_Conv|.*layer1_|.*layer2_[01]_)'
    # ~38GB VRAM
    # pattern: 'MythicConv2d::(n_ResNet_conv1_Conv|.*layer1_|.*layer2_)'
    # ~30GB VRAM
    # pattern: 'MythicConv2d::(.*layer1_|.*layer2_|.*layer3_)'
```

同じ3値はベンダー資料「BEVFormer Retraining Guide」(Rev 1.2, July 2026) §1.17.1にも「48GB GPU向け」の既定として明記されており、コードとドキュメントが一致している。SDK全体の既定は `enable: false`（`hydra_configs/torchnet/default.yaml:31-33`）で、BEVFormer 1600x900 だけが明示的にオンにしている。800x450設定は上書きしておらず、既定で checkpoint 無効。

### 3.4 AMP非対応・DDP

- fp16で呼んでも効果が無い。`DenaliSeparableModel.forward` が入力・重み・biasを明示的に`float32`へキャストしてから計算し、結果を元のdtypeに戻す（`_denali_ace_separable_model.py:440-447`、コメント:「分離モデルの精度問題を避けるため計算は必ずfp32で行う」）。したがってAMP等でfp16を使っても、Mythic層内部の保持活性はfp32のまま変わらない。
- DDPは `torchrun --nproc-per-node=N` に対応するが（`bevformer_train_mythic.py:486-497`）、各GPUが依然1サンプル(6カメラ)を処理するため**GPUあたりのメモリは減らない**。スループットのみ上がる。
- optimizer状態（AdamW、25M params分のm/v）は約0.4GBで無視できる規模。

---

## 4. to_training が別の理由でOOMする機構（詳細）

`train` とは独立に、`to_training`（統計収集）はGPU上で別のメモリ問題を起こす。本節は「量子化スケール決定で何を計算し、何を保持しているのか」「最大どれだけGPUメモリが必要か」「CPUで実行して問題ないか」を、コード解析と実測（本ドキュメントのための追加実測、GPU上で`to_training`を再現してOOMを直接観測）の両方から説明する。

### 4.1 量子化スケール決定は2段階に分かれる — 「データ収集」と「スケール計算」

`to_training` の量子化スケール決定（pFSR/iFSR/DSF の確定）は、実装上は次の2段に分かれており、**重いのは前者、軽いのは後者**である。

1. **統計収集**（`StatsCollector.collect()`, `_stats_collector.py:172-189`）: モデルの**全エッジ**（`model.get_all_edges()` — 畳み込みや重みのエッジだけでなく、グラフ中のあらゆる中間テンソルを指す。`build_edge_observers(..., include_outputs=False)` が全エッジに対して無条件にobserverを割り当てる、`_stats_collector.py:79-104`）に対し、2種類のobserverを取り付けて`stat_n_samples_default`件のバッチを**実際に推論**させる（`collect_edge_data`, `torch.no_grad()`内）:
   - **`HistogramObserver`**（`torch.ao.quantization.observer`標準クラス）: 固定ビン数のヒストグラムとmin/max/running統計を更新するだけ。**1エッジあたり数KB程度**で、バッチ数が増えても大きくならない。
   - **`EdgeMetadataObserver`**（`_observers.py:16-34`）: shape/dtypeに加え、**「このエッジは全サンプルで値が変わらない定数か」を判定する**ために、**最初に流れてきたバッチの値をそのまま`.detach().clone()`して保持し続ける**（`_observers.py:32`）。2バッチ目以降は`torch.equal`で比較するだけだが、1バッチ目のクローンは`collect()`が終わるまで（実装上は`edge_observers`辞書が破棄されるまで）解放されない。
   
   この「全エッジ×1バッチ分のクローンを同時に保持する」コストが、[03_accuracy_simulation.md](03_accuracy_simulation.md) §4.7.4 が報告した**7.528 G要素 = FP32で28.04GiB**（最大エッジ`[6,256,232,400]`=543.8MiB）に相当する。

2. **スケール計算**（`ScaleAllNodes`等, `munc/ops/scale_all_nodes.py`）: 上記の統計収集が返す`stats`辞書——エッジごとの`clip_min`/`clip_max`/`mean`/`std`/`shape`という**小さなスカラー値の集合**（`metadata_post_processor`/`make_histogram_post_processor`, `_stats_collector.py:277-387`）——だけを使い、グラフを辿ってスケールを伝播・分解する（`_scale_node`等, numpy/Python、GPU不要）。CSF（Composite Scale Factor）を`BreakFSRIntoPFSRAndIFSR`がpFSR（アナログ側）・iFSR（デジタル側）・DSFに分解するのもこの軽い段階に属する（[conversion_steps/to_training.md](conversion_steps/to_training.md) §7）。

   **つまり「スケールを計算する演算」自体はGPUメモリをほとんど使わない。GPUメモリを食うのは、その計算に使う統計を得るために全エッジの生テンソルを一時的に(1バッチ分)保持しながら推論を回す工程である。**

### 4.2 なぜ1回では済まないのか — STATS_REQUIREDなopごとに統計収集が繰り返される

`BaseOp.collect_stats_if_needed`（`_base_op.py:157-174`）は、各opの`requires_stat_collection`が`STATS_REQUIRED`なら**既存の統計があっても必ず`self.stats.collect()`を呼び直す**（`STATS_EXISTING`なら既存統計を再利用）。`ScaleAllNodes`は`requires_stat_collection: STATS_REQUIRED`（`scale_all_nodes.py`の`_get_info`）であり、他にも複数のopが同条件を持つ。

本ドキュメントのための追加実測（GPU、structural ONNXから`to_training`を実行、`stat_n_samples_default=20`）のログでは、**`Collecting required stats...`が合計5回出力**された。直前のopのログ行で位置を示すと:

| # | 直前のopログ | 役割 |
|---|---|---|
| 1 | `Removing shape inference nodes...` | グラフクリーンアップ直後の最初の統計収集 |
| 2 | `Marking relevant nodes as signed...`（1回目） | BatchNorm畳み込み・off-chip判定後 |
| 3 | `Marking relevant nodes as signed...`（2回目） | 入力shift/scale注入後 |
| 4 | `Balancing Concat inputs...` | Mul/MatMul/Softmax/Add/Concatへのスケール注入ノード挿入後 |
| 5 | `Scaling all nodes...`（`ScaleAllNodes`本体） | 最終スケール確定 |

ログ上の「直前の行」とそれを引き起こしたop定義の対応は、`to_training.md` §10.1 で既に指摘されている通り完全には一致しない場合がある[推測]が、**「1回のto_training実行中に統計収集(=全エッジクローン)が複数回走る」こと自体はログから直接確認できる事実**である。各回とも`stat_n_samples_default`件のバッチを新しい`TorchNet`で最初からやり直す（`StatsCollector.collect()`が毎回`self._make_torchnet()`で新規インスタンスを作る, `_stats_collector.py:163-166,175`）。

### 4.3 実測: GPUメモリは5回の統計収集を経て階段状に積み上がり、最後の回でOOMする

上記と同じ実測で、`nvidia-smi`のメモリ使用量を1秒間隔で記録し、ログの`Collecting required stats...`の出現位置と対応させた結果:

| 区間 | GPUメモリ（nvidia-smi, 概算） | 備考 |
|---|---|---|
| 統計収集#1の直前 | 3 MiB | モデルロード・グラフ編集はほぼメモリを使わない |
| 統計収集#1 完了時 | **31,985 MiB**（≈31.2GiB） | 03§4.7.4の28.04GiBに近い。ここで全エッジのクローンが一気に積まれる |
| 統計収集#2〜#3 完了時 | 31,985〜31,991 MiB | ほぼ変化なし（この間のグラフ編集で増える有効エッジ数は小さい） |
| 統計収集#4 完了時 | **41,139 MiB**（≈40.2GiB） | 直前にMul/MatMul/Softmax/Add/Concatへのスケール注入ノードが多数挿入され、インストゥルメント対象のエッジ数自体が増えたため |
| 統計収集#5（ScaleAllNodes）進行中 | 45,389 MiB まで上昇して**OOM** | L40S 44.39GiB の天井に到達 |

OOM発生箇所は本実測でも`_observers.py:32`の`self._constant_value = value.detach().clone()`そのもの（エラーメッセージ: `"this process has 44.32 GiB memory in use"`、GPU容量44.39GiB）。既存ログ（[03_accuracy_simulation.md](03_accuracy_simulation.md) §4.7.4）が報告する`Dropout.py:61`/`_observers.py:32`/`Div.py:23`という複数の発生箇所のばらつきも、「複数回の統計収集のどの回で、かつその回のどのエッジ処理中に天井を超えるか」が実行ごとに(スケジューリングやアロケータの断片化状況により)変わることの反映と考えられる[推測]。

**重要な点**: 一度積まれたメモリは**次の統計収集が始まる前に明示的には解放されない**。`munc`パッケージ内を`grep`しても`torch.cuda.empty_cache()`の呼び出しは1件も無い。各`collect()`呼び出しのローカル変数（`torch_model`・`edge_observers`）はPython的にはスコープを抜ければ参照が切れるはずだが、PyTorchのキャッシングアロケータが確保済み領域を保持し続けるため、`nvidia-smi`上の使用量は前の回より下がることなく、**グラフが成長する(エッジが増える)たびに新たな積み増しが乗る**階段状の挙動になる。これが、既存ドキュメントが報告していた「43.90GiB」「35.05GiB」といった複数の異なるOOM時使用量（§4.7.4参照）が実行ごとに違う理由でもある——どの回でOOMするかで、積算される段数が変わる。

### 4.4 サンプル数を減らしても解決しない理由

`EdgeMetadataObserver`のクローンは**各collect()呼び出しの最初の1バッチ目**でのみ発生し、2バッチ目以降は比較のみ（§4.1）。したがって`stat_n_samples_default`を200→20に減らしても、各回の「1バッチ目クローン」のコストはそのまま残る。サンプル数が効くのは**処理時間**（バッチ数分の推論を回す時間）だけであり、**ピークメモリには効かない**。これが[03_accuracy_simulation.md](03_accuracy_simulation.md) §4.7.4が実測で確認した「200→20に減らしてもOOM」という結果の正確な理由である。

### 4.5 最大どれくらいのGPUメモリが必要か

本実測ではL40S(44.39GiB)が完全に空の状態でも、5回目の統計収集(ScaleAllNodes)の途中で45.4GiB付近に達してOOMした。**この回は途中で死んでいるため、仮に十分なメモリがあった場合の最終的なピークは本実測だけでは確定できない**。既存ログ（03§4.7.4）でも45GiB空きのGPUでOOMしていることから、**BEVFormer-tiny 1600x900の`to_training`をGPU上で完走させるには、48GB級GPUでも不足する可能性が高く、必要量は50GB以上になる可能性がある**[推測、本リポジトリではより大きいGPUでの検証はできていない]。少なくとも「`train`の既定ckpt設定(43.5GiB, §3-5参照)と同程度か、それ以上」の予算が必要になる。

### 4.6 CPUで実行して問題ないか

**精度への影響は無い。** `device_name`は`Session.__init__`内で`StatsCollector`のコンストラクタにのみ渡され（`_session.py:90,99`）、`StatsCollector._make_torchnet()`がこの値をそのまま`TorchNet(..., device_name=...)`に渡す（`_stats_collector.py:163-164`）。この経路は**統計収集専用のTorchNetインスタンスにしか影響せず**、`train`や`eval_trained`が使う別のTorchNetインスタンスには伝播しない。ヒストグラム（min/max/running累積）とメタデータ（shape/dtype/定数判定のための等値比較）はいずれも**デバイスに依存しない決定論的な演算**であり、CPU/GPUどちらで計算しても得られる`clip_min`/`clip_max`/`mean`/`std`の値は（浮動小数点演算順序の違いによる末尾ビットの差を除き）実質的に同一になる。したがって`++to_training.device_name=cpu`は**正確さを犠牲にせず、速度だけを犠牲にする**回避策である。

CPUでも「全エッジ1バッチ分のクローンを5回積み上げる」という同じ構造のメモリ消費は起きるが、ホストRAM(本実測環境は248GB、使用中はわずか約14GB)はGPUの44GBクラスのVRAMより桁違いに大きいため、同じ消費量が**単に余裕を持って収まる**。CPUが「メモリ効率がよい」わけではなく、「使える容量が大きい」だけである点に注意。実測の所要時間は約35〜42分（mini, 20サンプル×複数回の統計収集を含む, 03§4.7.4）。

---

## 5. 実測結果（本ドキュメントのための追加実測）

コンテナ `mythic_bevformer_train` で `steps=train`（mini、`train.workers_per_gpu=0`）を条件別に数バッチ実行し、ログの `Mem:`（`torch.cuda.max_memory_allocated()`、`bevformer_train_mythic.py:348`、`reset_peak_memory_stats`は呼ばれないのでプロセス開始からのピーク）と並行して`nvidia-smi`のメモリ使用量（reserved、ドライバ込み）を記録した。

| 条件 | 設定 | `Mem:`(allocated) | nvidia-smi(reserved) | IterTime |
|---|---|---|---|---|
| **A. 既定（45GBプリセット）** | ckpt pattern既定、`denali_training_model` | **43,451 MB** | 44,249 MB | 7.23 s |
| **B. 30GBプリセット** | ckpt pattern `(.*layer1_\|.*layer2_\|.*layer3_)` | 27,241 MB | 28,267 MB | 7.77 s |
| **C. 全層checkpoint（下限）** | pattern=`null`（全`MythicConv2d`をcheckpoint） | 18,191 MB | 19,379 MB | 8.72 s |
| **D. checkpoint無効** | `activation_ckpt_config.enable=false` | OOM（1 iter目で失敗） | — | — |
| **E. ノイズ無効** | 既定ckpt、`noise_config=denali_no_noise` | 43,329 MB | 43,951 MB | 6.64 s |
| **F. 素のFP32学習（Mythic化なし）** | `steps=train_torch_fp`、同解像度、mmdet3dネイティブ | 9,599 MB | 10,879 MB | 1.85 s（data_time 1.15s含む。計算時間≈0.65s） |
| **G. to_acm（統計収集、GPU）** | `stat_n_samples_default=10`相当 | ピーク約23,863 MB（収集中に観測） | — | 全体で約3分37秒（約100バッチ） |

観測:

- **A は既存ログ（mini本番再学習, 2026-08-26実行）の43,466MB・7.16sと一致**（本実測は43,451MB・7.23s）。再現性を確認した。
- **D（checkpoint無効）は1イテレーション目でOOMする**（"this process has 44.36 GiB memory in use"）。L40S（44.39GiB）では完全無checkpointでは最初の1バッチも通らない。これはBEVFormer Retraining Guideが「48GB GPUでも既定のcheckpoint設定が必要」と案内している理由を裏付ける実測である。
- **E（ノイズ無効）はメモリをほぼ変えない**（43,451→43,329 MB、-0.3%）が、**速度は約8%改善する**（7.23s→6.64s）。ノイズ注入（`randn`呼び出し）は計算コストはあるが、backward用に保持するテンソル量にはほとんど効かない。つまりノイズはメモリの支配要因ではない。
- **F（素のFP32、Mythic化なし）が9,599MBなのに対し、A（Mythic化、既定checkpoint）は43,451MB——約4.5倍**。この比較から、Mythic化（アナログ量子化シミュレーション層）そのものが支配的な追加コストであることが数値で確認できる。
- **C（全MythicConv2dをcheckpoint）でも18,191MBとFの約1.9倍残る**。これはFPN・transformer・検出headなど「backbone以外のオフチップ部分」がcheckpoint対象外のまま残ること、およびcheckpoint境界ごとに保持する入力テンソル自体もMythic層では(signed-input分離等により)素の層より大きいことに起因すると考えられる**[推測]**。
- **計算時間でもMythic化の影響は大きい**: F の計算時間(data_time除く) ≈0.65s/iterに対し、A の7.23s/iterは**約11倍**。多項式近似の2回conv実行・SAR ADCシミュレーション・`torch.compile`経由の多項式評価などの追加計算が、メモリだけでなく学習時間も支配している。
- **to_acm（G）のピークメモリ（約24GB）はtrainの既定ckpt(約43.5GB)より小さい**。backward不要（統計収集はno_grad）なためで、ただしto_trainingと同じobserverの仕組みを使うため、サンプル数が多い本番設定（既定200）ではより長時間に渡って活性を保持し、§4と同種のOOMが理論上起こりうる**[推測・本実測では10サンプル相当に減らして確認、OOMは発生しなかった]**。

---

## 6. 必要スペックと所要時間の見積もり

### 6.1 GPU

- 既定の45GBプリセットは48GB級GPU（NVIDIA A6000/L40S）を前提に調整されている。Mythic社自身もBEVFormerの再学習を**8×NVIDIA A6000(48GB)**で実行している（Model Summary Report Rev 3.0, p.11-12）。
- 32GB級GPUなら30GBプリセット（§5のB相当、本実測27.2GB）、40GB級なら38GBプリセットが候補になる。
- 80GB級（A100/H100）を使えば、checkpointを緩めて（またはVRAM次第で無効化を試みて）速度を優先できる可能性がある**[推測、本実測はL40S 44.4GBのみで無checkpointはOOMしたため未確認]**。
- VRAMが足りない場合の最終手段はpattern=`null`の全層checkpoint（本実測18.2GB）だが、IterTimeが約20%増える。

### 6.2 所要時間

mini実測の `IterTime` を使い、フルnuScenes trainvalでの所要時間を外挿する。train split（700 scenes）のサンプル数は、full annotation生成ログの実測値（850 scenes全体で34,149 samples, [03_accuracy_simulation.md](03_accuracy_simulation.md) §4.7.5）からscene数で比例配分し、**約28,100 samples**と見積もる**[推測、実際のtrain split固有の分布により増減しうる]**。

| GPU数 | 1epochあたり | 24epochあたり |
|---|---|---|
| 1 GPU（A条件 7.23s/iter） | 28,100 × 7.23s ≈ 56.4時間 | ≈ 56.4時間 × 24 ≈ **56.4日** |
| 8 GPU（DDP、線形スケーリング仮定**[推測]**） | ≈ 7.1時間 | ≈ **7.1日** |

- DDPはGPUあたりのメモリを減らさないため、8GPU構成でも各GPUに45GB級VRAMが必要。
- `DataTime`はmini実測で0.03s/iter（`train.workers_per_gpu=0`でも無視できる水準）であり、CPU/データローダはボトルネックではない。F条件（mmdet3dネイティブ、同じworkers_per_gpu=0）ではdata_timeが1.15s/iterと大きいが、これは単一プロセスでのJPEGデコード・前処理コストであり、`train.workers_per_gpu`を増やせば並列化できる（ホストRAM/CPUコア数が許す範囲で）。
- CPU/RAMについてベンダーのYOLOガイドは「Intel Core i7以上、RAM 64GB、GPU VRAM 48GB」を要求仕様として明記している。BEVFormer専用の要求スペックは資料に無いが、画像入力の前処理負荷は同程度と見てよい**[推測]**。
- `to_training`の統計収集（CPU実行時)は約35〜42分（mini, §4）。`to_acm`は本実測で統計サンプル数を減らした場合約3.6分（§5-G）、既定の200サンプルでは単純比例で**約70分程度**になると予想される**[推測、サンプルあたりの処理時間が一定という仮定]**。

### 6.3 メモリ削減の手段のまとめ

| 手段 | 効果 | 代償 |
|---|---|---|
| activation checkpointパターンを狭める（既定→30GB） | メモリ -37%（43.5→27.2GB） | IterTime +7%（7.23→7.77s） |
| 全層checkpoint（pattern=null） | メモリ -58%（43.5→18.2GB、下限） | IterTime +21%（7.23→8.72s） |
| 解像度を800x450に下げる | 画素数が1/4になり大幅減（本ドキュメントでは未実測、ONNX再生成が必要） | mAP低下（精度とのトレードオフ） |
| `noise_config=denali_no_noise` | ほぼ無効（§5参照） | 精度シミュの意味が薄れる（ノイズ耐性を学習しない） |
| DDPでGPU数を増やす | 時間短縮のみ、メモリは不変 | GPU数倍のVRAM予算が必要 |

---

## 7. 参照

- ソース（コンテナ `/root/mythic_sdk/v26.05.2/mythic-model-zoo/` 相対、`.venv/lib/python3.12/site-packages/` 配下のパッケージは `munc/...` 等と表記）:
  - `mythic/model_zoo/bevformer/bevformer_train_mythic.py`（training loop, batch size, DDP, `Mem:`計測）
  - `configs/bevformer/bevformer_tiny.yaml`, `configs/bevformer/model_setup/tiny_1600x900.yaml`, `configs/common/base_config_generic.yaml`
  - `_torchnet.py:382-500,461-462`（forward, checkpoint wrapping, delete_unused_edges）
  - `_ace_model.py`, `_denali_ace_separable_model.py:441-447`（Mythic層の内部計算、fp16非対応）
  - `_observers.py:16-34`（`EdgeMetadataObserver`のclone保持、`HistogramObserver`との役割分担）
  - `_stats_collector.py:25-61,79-104,112-209,277-387`（`collect_edge_data`の全エッジhook、`StatsCollector.collect()`が毎回新規TorchNetを作る、観測結果を小さい`stats`辞書に圧縮するpost-processor群）
  - `munc/_base_op.py:157-174`（`collect_stats_if_needed`、`STATS_REQUIRED`なopは既存統計があっても必ず再収集）
  - `munc/ops/scale_all_nodes.py`（`ScaleAllNodes`の`requires_stat_collection: STATS_REQUIRED`、スケール伝播自体は`stats`辞書上のnumpy演算）
  - `_session.py:90,99`（`device_name`は`StatsCollector`にのみ伝播し、`train`/`eval_trained`の別TorchNetには影響しない）
  - `hydra_configs/training_model/denali.yaml`, `hydra_configs/noise_config/denali_training_model.yaml`
- ベンダー資料: `mythic_sdk/v26.05.2/doc/user-guides/BEVFormer Retraining Guide.pdf` §1.17（checkpointプリセット・48GB GPU前提）、`mythic_sdk/v26.05.0/doc/reports/Model Summary Report.pdf` p.11-12（BEVFormerは8×A6000で学習）、`doc/user-guides/YOLO Retraining Guide.pdf` §2（i7/RAM64GB/VRAM48GB要求仕様）
- 既存ドキュメント: [03_accuracy_simulation.md](03_accuracy_simulation.md) §4.7.4（to_trainingのOOM機構、28.04GiB活性の実測）、[conversion_steps/to_training.md](conversion_steps/to_training.md)、[conversion_steps/to_acm.md](conversion_steps/to_acm.md)
- 本ドキュメントの追加実測: §3-5の7条件と§4-3の`to_training`メモリ階段は、いずれもホストL40S・コンテナ`mythic_bevformer_train`で本ドキュメント執筆時に実行して得た。ログ・`nvidia-smi`トレースは実行後に削除済みで、本文の数値・表が実測結果の記録そのもの。再現する場合は§3-5・§4のコマンド断片と設定キーを参照。
