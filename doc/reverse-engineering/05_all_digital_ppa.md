# 全デジタル実行時の PPA 推定(BEVFormer-Tiny 実測)

状態: **調査完了(SDK v26.05.2)。P(性能)・P(電力)は推定可能で実測済み、算出式も `vnnmap` 逆アセンブルで確定(§2.2-§2.5, §5)。A(面積)は SDK から出ない。**

対象の問い: BEVFormer 等のモデルをアナログ ACE を一切使わず**すべてデジタル(v-MP)で**動かした場合の PPA を、SDK 内のツールで推定できるか。

結論の要約:

| 軸 | 推定可能性 | 根拠 |
|---|---|---|
| **P**erformance(cycles / fps / latency) | **可能** | `vnnmap --explore` がグラフ全体に対して cycle-accurate 相当の見積りを出す(§2)。算出式は逆アセンブルで確定(§2.2-§2.4) |
| **P**ower(mW) | **可能**(5nm typ、ダイナミック電力のみ) | `vnnmap` の per-component pJ モデル。係数は `[sys]` から上書き可能(§5) |
| **A**rea(mm²) | **不可(SDK 外の情報が必要)** | デジタル側に面積出力・面積シンボルが一切存在しない(§6) |

実測の結論: BEVFormer-Tiny をフルグラフで全デジタル実行すると、既定構成で **58.87 ms / 25.58 W@30fps**。ハイブリッド(m2072: 31.58 ms / 4.505 W@30fps、[PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) §2)に対し **latency 1.9 倍・power 5.7 倍**で、33 ms 制約を満たさない。オンチップ SRAM を 512 MB まで拡張した非現実的構成で初めて 33.19 ms / 18.27 W@30fps に届く(§4)。

---

## 1. なぜ「全デジタル」推定が成立するのか(アーキテクチャ上の根拠)

M2000 の PPA 推定は 2 系統に分かれており、**デジタル側はアナログ側から完全に独立している**([02_ppa_estimation.md](02_ppa_estimation.md) §1)。

```
アナログ経路: 学習済みモデル → dnn_compiler → final.l0.pb / perf_trace_dump.h5
                → perf_analysis.py(latency/fps/ACE util/area)
                → power_estimator.py(analog power)
                       ↑ ここに外部 JSON を注入するだけ
デジタル経路: VidModel → optimize() → quantize() → .vidir
                → vnnmap --explore(cycles / fps / Power@30fps / MAC util)
                → <Model>_proflings.json
```

アナログ側の estimator はデジタル NPU を**物理モデル化していない**。`--digital-npu-log-path` / `--npu-log-path` で渡された JSON の値をそのまま読み、`frame_latency_ns += 1e9/npu_fps` [perf:1303] と `calc_energy_digital_npu = power_mw_5nm_typ / fps / 1000` [pow:529-542] で合算するだけである([02_ppa_estimation.md](02_ppa_estimation.md) §4.11)。

したがって「全デジタル」の推定とは、**アナログ estimator を使わず、デジタル経路(`vnnmap --explore`)を全ノードに対して単独で走らせること**に等価であり、SDK に手を入れずに成立する。`vnnmap` が必要とするのは ONNX 由来のグラフ(`.vidir`)と `[sys]` 設定だけで、ACE への配置情報は一切参照しない。

### 1.1 アナログ/デジタルの op 分割は SDK 側で固定されている

通常フローでは Conv/Gemm/MatMul/Mul が `MythicConv2d`/`MythicLinear`(アナログ)に、depthwise conv が `__digital_onchip` に、非対応 op が off-chip に振られる([00_overview.md](00_overview.md) §3.5 レベル A)。この分割は `to_structural`/`to_training` で決まり、コンパイラの `auto_partition` は物理配置(レベル B)のみを担当する。

**全デジタル推定はこの分割機構を経由しない。** `vnnmap` は与えられた ONNX グラフの全ノードをデジタル実行対象として扱うので、アナログ/デジタルの op 分類を書き換える必要はない。

---

## 2. 推定に使う経路(実行の実体)

`vnnsdk_scripts/mythic_utils.py::run_vnn_flow()` が経路の全体。

| ステップ | 実装 | 備考 |
|---|---|---|
| `[sys]` 読み込み | `mythic_utils.py:185-188` | `nMPs` と `frequency` のみ Python 側で読む。他のキーは cfg ファイルごと `vnnmap` に渡される |
| `model.optimize()` | `mythic_utils.py:193` / `:200` | `skip_validation=True` で数値等価チェックをスキップ |
| `model.quantize(...)` | `mythic_utils.py:209` | `QuantizationConfig(calibration_dataset_size=1)` = ランダム 1 サンプルのダミー量子化 |
| `explore_model(...)` | `mythic_utils.py:211` | `ModelState.QUANTIZED` を要求。`.vidir` を `vnnmap` に渡す |
| JSON 出力 | `mythic_utils.py:221-226` | `{"system_config": <cfg 全セクション>, "profiling": <抽出値>}` |

`vnnmap` の実行コマンドは `vnnmap/run_vnnmap.py:227-238`:

```
vnnmap --csv_dir=<dir>/csv --output_prefix=<model> --network=<model>.vidir \
       --system_cfg=<cfg> --explore --edma
```

`--explore --edma`(= `advanced=True`)と `--codegen`(= `generate_vci=True`)は同時指定不可(`run_vnnmap.py:199-200`)。PPA 推定だけなら `--explore` 側を使う。

### 2.1 出力メトリクスの対応

`vnnmap` の stdout を `run_vnnmap.py::_extract_metrics` が正規表現で拾い、`mythic_utils.py::PROFILING_FIELDS` (`:22-28`) が JSON キーへ写す。

| stdout | `_extract_metrics` キー | JSON キー |
|---|---|---|
| `Total: <n>`(Cycles per inference) | `Total Cycles` | `total_cycles` |
| `eff. fps: <n>` | `Effective FPS` | `fps` |
| `Power@eff. fps: <n> mW` | `Power@eff. fps (mW)` | `power_mw_5nm_typ` |
| `Power@30fps: <n> mW` | `Power@30fps (mW)` | `power_mw_5nm_typ_30fps` |
| `efficiency: <n>%` | `Efficiency (%)` | `mac_utilization_pct` |
| `MACs:<n> bn` | `MACs (bn)` | `macs_bn` |
| `(<n> MB)` | `Model Size (MB)` | `model_size_mb` |

JSON に写らないが `_extract_metrics` が拾っている値(`run_vnnmap.py:104-133`)は探索上重要:
`Max DDR (kB)` / `Max OCR (kB)` / `MAC Cycles` / `Non MAC Cycles` / `Exposed DMA Cycles` / `Effective Latency (ms)` / `DDR Read (MB)` / `DDR Write (MB)`。
これらは `run_vnnmap()` の戻り値 dict から直接取れるので、`run_vnn_flow` を使わず `run_vnnmap` を直接呼ぶスイープでは全部使える(§3.3)。

### 2.2 レイテンシ / fps / 利用率の算出式(`vnnmap` 逆アセンブル)

[02_ppa_estimation.md](02_ppa_estimation.md) §3.8 のデジタル NPU レイテンシ(`profiling.fps` → `1e9/npu_fps`)の出所。
`vnnmap` は strip されていない(4396 シンボル)ため、DWARF なしでも `objdump -dC` + シンボル名で追える。
以下は `vNetwork::printFullProfile()`(`0x195eb0`)の該当ブロック(`0x197558`-`0x19774b`)から復元し、
gdb で実行時の構造体を読み出して数値一致を確認した結果である(検証手順は §2.5)。

```
eff. fps        = nBatch × frequency / (procCycles + exposedDmaCycles)
eff. latency ms = 1000 × (procCycles + exposedDmaCycles) / frequency
max. fps        = nBatch × frequency / procCycles                        # "DMAs hidden" 表示側
efficiency %    = MACs_bn × 1e9 × 100 / ((procCycles + exposedDmaCycles) × 64 × nMPs)
```

`vNetwork` の該当フィールド: `+0x120`=nMPs、`+0x124`=nMACs、`+0x130`=nBatch、`+0x138`=frequency、
`+0xd38`=総 MAC 数(bn, double)、`+0xd58`=procCycles(double)、`+0xd60`=exposedDmaCycles(int64)。

この式から従う性質は 3 件の cfg 摂動実験で確認済み(BEVFormer フルグラフ `.vidir`、§4.1 の baseline を基準):

| 摂動 | cycles | eff. latency | eff. fps | efficiency | P@30fps |
|---|---|---|---|---|---|
| `nBatch` 1→2 | 不変 | 不変 | ×2 (16.99→33.97) | 不変 | 不変 |
| `frequency` 2→1 GHz | ビット単位で不変 | ×2 | ×½ | 不変 | 不変 |
| `nMACs` 32→64 | 117.59 M→93.35 M | 短縮 | 16.99→21.41 | 41.67→52.52 % | — |

注意すべき点が 3 つある。

1. **`frequency` は cycle モデルに一切入らない。** cycles がビット単位で不変であることから、DDR レイテンシ
   (既定 200)も「ns ではなく cycle」で数えられている。周波数を上げると DDR アクセス時間も比例して
   短くなる扱いになるので、高周波側の見積りは楽観側に振れる。
2. **`efficiency` は `nMACs` を見ず 64 MAC/cycle/MP 固定で計算する。** 式の除数は `64 × nMPs`
   (定数 `0.015625` = 1/64、`0x245c10`)で、`vNetwork+0x124`(nMACs)を参照しない。一方 cycle モデル側の
   理想 MAC cycle は `2 × nMPs × nMACs` を使う(§2.3)。既定 `nMACs=32` では両者が一致する(2×32=64)が、
   `nMACs=64` にすると表示 52.52% に対し真の利用率は約 26% となる。**`nMACs` を既定から変えたスイープでは
   `efficiency` を利用率として読んではいけない。**
3. **`eff. latency` に `nBatch` は掛からない。** `nBatch>1` では `eff. fps` はバッチスループット、
   `eff. latency` は 1 バッチの時間になる。[02_ppa_estimation.md](02_ppa_estimation.md) §3.8 の
   `frame_latency_ns += 1e9/npu_fps` は fps の逆数なので 1 フレームあたりに正規化された値であり、
   `eff. latency` とは `nBatch` 倍ずれる。既定 `nBatch=1` では一致する。

### 2.3 cycle モデルの中身

**MAC cycle** — `vLayer::setMpCycles()`(`0x19c090`)。

```
getMacCycles(n, nMACs, x)   = (n / nMACs) * x          # 0x14d010、64bit 符号なし div → imul
getMacOverheadCycles(n)     = n / 9                    # 0x14d030、マジック乗数 0xe38e38e38e38e38f
vProfile+0x720 (MAC cycles) = getMacCycles(...) + getMacCycles(...)/9
```

MAC オーバーヘッドは**一律 +11.1%**(`n/9`)。レイヤ種別(`vLayer+0x570==1`)によって ×4 / ×3 の
倍率が掛かる分岐がある。

**DMA cycle** — `vLayer::readDMA(...)`(`0x1682d0`、本体 `0x168410`)。

```
DMA cycles = nbytes / readDiv<MEMTYPE> + readLatency<MEMTYPE>
```

`readDiv` は `vNetwork+0x170`、`readLatency` は `+0x158` の int テーブルで、`vBuf+0x74` のメモリ種別で
索引する。既定値は **DDR: nbytes/4 + 200 cycles、OCR: nbytes/8 + 1 cycle**。書き込み側は
`writeDiv*` / `writeLatency*`(既定すべて 1)。

算出した DMA cycle は `dmaThreshold`(`vNetwork+0x188`、既定 765)で露出判定に掛かる。閾値未満は
exposed 0(完全に隠蔽)、超過分だけが `sub`/`add`/`cmovae`(`0x1685e8`)で部分露出する。
§4.3 で `nMACs` や `OCRAM0` を変えると exposed DMA が大きく動くのはこの機構による
(MAC 時間を短くすると隠蔽できる量が減り、exposed DMA が増える)。

**`[sys]` の全キー** — `parseSystemCfg`(`0x68c90`)が `INIReader` で読む。既定値と `vNetwork` オフセット:

| キー | 既定 | off | キー | 既定 | off |
|---|---|---|---|---|---|
| `nMPs` | 8 | 0x120 | `OCRAM0` | 32 MB | 0x140 |
| `nMACs` | 32 | 0x124 | `OCRAM1` | 4 MB | 0x148 |
| `nSlots` | 2 | 0x128 | `DDR` | — | 0x150 |
| `systemWW` | 8 | 0x12c | `dmaThreshold` | 765 | 0x188 |
| `mCluster` | 1 | 0xd70 | `readLatencyDDR` | 200 | 0x15c |
| `pCluster` | 1 | 0xd74 | `readLatencyOCR` | 1 | 0x160 |
| `DMEM1` | 0x800 | 0x190 | `writeLatencyDDR` | 1 | 0x168 |
| `DMEM2` | 0x6000 | 0x194 | `writeLatencyOCR` | 1 | 0x16c |
| `DMEM3` | 0x8000 | 0x198 | `readDivDDR` | 4 | 0x174 |
| `frequency` | 1.2 GHz | 0x138 | `readDivOCR` | 8 | 0x178 |
| `nBatch` | 1 | 0x130 | `writeDivDDR` | 1 | 0x180 |
| `xTile` | 1 | 0x134 | `writeDivOCR` | 1 | 0x184 |
| `memPrefWGT` | — | 0x1c0 | `nomemFirstGraphInput` | bool | 0x1d4 |
| `OCR` / `DDRConfig` | — | — | `nomemGraphOutput` | bool | 0x1d5 |

`readLatency*` / `readDiv*` / `dmaThreshold` は cfg から書けるので、**DMA モデル自体をスイープ対象に
できる**(§9.2)。加えて全 `pow*Pj` 係数も `[sys]` キーである(§5.1)。

### 2.4 表示される `Total` はレイテンシの分母ではない

`vnnmap` は cycle の集計を **2 系統**持っており、値が一致しない。

**(a) `Cycles per inference:` ブロック**(`0x196f0f`-`0x196f9c`)— stdout に出る内訳。

```
MAC         = Σ vProfile+0x720
non MAC     = Σ vProfile+0x728
exposed DMA = p[0].0x710 + p[0].0x6f8 + Σ_i (p[i].0x6c8 + p[i].0x6e0)
Total       = MAC + non MAC + exposed DMA
```

同じ 4 項和は `vNetwork::getTotalDDRcycles`(`0x195710`)/ `getTotalProfile`(`0x195790`)にもある。
**このブロックはマッピング反復ごとに複数回 print される**(BEVFormer フルグラフでは 12 回)。
反復が進むと値が下がる(`Total` 117,605,925 → 117,591,123 → 117,588,241)。

**(b) `vSummaryProfile`**(`vNetwork+0xd40` に埋め込み、`vSummaryProfile::fill(vNetwork*)` = `0x1958d0`)—
`eff. fps` / `eff. latency` / `efficiency` が実際に読む値。全レイヤを走査して:

```
skip           : vLayer+0x1b8 != 0 のレイヤ(BEVFormer では 637 中 205 個)
ideal (+0xd50) += vLayer+0x228 (層 MACs, bn) × 1e9 / (2 × nMPs × nMACs)
actual(+0xd58) += max( vLayer+0x598 , 1.1 × ideal )          ← 1.1 は 0x245c00 の定数
DDR   (+0xd60) += getMaxDDRcycles(vLayer+0x240)              ← 0x195810、層内タイルの最大
params(+0xd40) += vLayer+0x230 ; (+0xd48) += vLayer+0x234
```

`vLayer+0x598` は `vLayer::calcMaxCycles()`(`0x19ed40`)が入れる「層内タイルの `vTile+0x820` の最大」。
`+0x18` の加算は **1 層あたり「理想 MAC 時間の 110%」を下限とするフロア**になっている。

BEVFormer フルグラフ(既定 cfg)で gdb から実測した突き合わせ:

| 量 | 値 |
|---|---|
| 走査レイヤ / skip / フロア発動 | 432 / 205 / **30 層** |
| ideal MAC cycles (`+0xd50`) | 49,064,362.5 |
| Σ `vLayer+0x598`(フロア適用前) | 69,616,769 |
| actual proc cycles (`+0xd58`、フロア適用後) | 69,746,355.667 |
| フロアによる増分 | **+129,586.667** |
| exposed DMA (`+0xd60`) | 47,989,156 |
| **レイテンシの分母** | **117,735,511.667** → 58.8678 ms / 16.9872 fps / 41.6734 % |

一方 stdout 最終ブロックの `Total` は 117,588,241(→ 58.794 ms / 17.0085 fps 相当)で、
**レイテンシの分母より 147,270.667 cycles(0.125%)少ない**。差の内訳は
フロア +129,586.667 と、集計系統の違い(`Σ vProfile+0x720/+0x728` 対 `Σ vLayer+0x598`)+17,684。

重要なのは `Σ vLayer+0x598` = 69,616,769 が **1 回目**の print ブロックの `MAC + non MAC`
(67,599,073 + 2,017,696)と完全に一致することである。つまり:

```
JSON total_cycles (117,605,925) = Σ vLayer+0x598 (69,616,769) + exposed DMA (47,989,156)
レイテンシの分母 (117,735,511.667) = 上記 + 1.1× フロア増分 (129,586.667)
stdout 最終ブロック (117,588,241)   = より後段のマッピング反復の値。どちらにも使われない
```

`_extract_metrics`(`run_vnnmap.py:113-116`)は `re.search` なので **最初の**ブロックを拾う。
結果として JSON の `total_cycles` / `MAC Cycles` / `Non MAC Cycles` は最終反復値ではないが、
**偶然 fps / latency の基礎になっている系統と整合する**。`eff. fps` / `Power@*` は 1 回しか print
されないので影響を受けない。§4.1 の cycle 内訳(MAC 67,599,073 等)は 1 回目のブロックの値である。

### 2.5 検証手順(gdb)

コンテナ内に `gdb` が入っているので、`vNetwork` を直接読める。ASLR 無効化は権限不足で失敗するが
`$rdi` から `this` を取れば足りる(その warning は無害)。`tools/digital_ppa/probe_vnnmap_cycles.py`
が §2.2-§2.4 の全数値を再現する:

```bash
docker exec mythic_digital_ppa bash -lc '
VN=/mythic/pyvnnsdk-env/lib/python3.12/site-packages/vnnmap/vnnmap
mkdir -p /work/probe && cd /work/probe
gdb -q -batch -x /work/probe_vnnmap_cycles.py --args $VN \
  --csv_dir=/work/probe/csv --output_prefix=P \
  --network=/work/out_full/BevformerTiny.vidir \
  --system_cfg=/work/system_configs/bevformer.cfg --explore --edma'
```

`printFullProfile` 到達時に `vNetwork` を読み、レイヤ配列(`+0x38`/`+0x40`)を独立に走査して
集計を再計算する。`PROBE-RECOMP` と `PROBE-STORED` が一致すれば §2.4 の集計式が正しい。出力例:

```
PROBE-CFG      nMPs=288 nMACs=32 nBatch=1 frequency=2000000000 peakMACs/cycle=18432
PROBE-LAYERS   ptrs=637 counted=432 skipped=205 mac_floor_applied=30
PROBE-RECOMP   ideal=49064362.5000 proc=69746355.6667 proc_unfloored=69616769.0000 floor_delta=129586.6667
PROBE-STORED   ideal=49064362.5000 proc=69746355.6667 dma=47989156 macs_bn=904.354330
PROBE-DERIVED  cycles=117735511.667 eff_fps=16.9872 eff_latency_ms=58.867756 efficiency=41.6734%
PROBE-HIDDEN   cycles=69746355.667 max_fps=28.6753 min_latency_ms=34.873178  (DMAs hidden)
```

再計算値は格納値と完全一致し、`PROBE-DERIVED` は `vnnmap` が print する
`eff. fps: 16.99 / eff. latency: 58.87 ms / efficiency: 41.67%` を再現する。
`vNetwork+0xd58` / `+0xd60` への静的な store 命令は存在せず(コンストラクタ `0x39560` のゼロ初期化のみ)、
書き込み元の特定には watchpoint が必要だった(`vSummaryProfile::fill` 内 `0x195936` / `0x19593f`)。
なお `vNetwork::printNetwork2csv()`(`0x135450`)が `+0xd50` / `+0xd58` を
`vLayer::appendCsvDescription(double,double)`(`0x90980`)へ渡すので、`<model>_flow.csv` からも
同じ 2 値を確認できる。

---

## 3. 再現手順

### 3.1 コンテナ

デジタル経路は compilerd コンテナ内の `/mythic/pyvnnsdk-env` に閉じている。SDK コンテナは不要。

```bash
docker run -d --name mythic_digital_ppa --memory=200g \
  -v <host_work_dir>:/work \
  gcr.io/mythic-devops/compilerd-bin:v26.05.2 sleep infinity
```

- Python は `/mythic/pyvnnsdk-env/bin/python`(システム python ではない)。
- SDK スクリプト群は `/mythic/vnnsdk/scripts`。`sys.path.insert(0, "/mythic/vnnsdk/scripts")` が必要。
- `vnnmap` バイナリは `/mythic/pyvnnsdk-env/lib/python3.12/site-packages/vnnmap/vnnmap`。直接叩くと stdout 全文(レイヤごとの中間出力含む)が見られる。
- メモリは要る。フルグラフの量子化で数十 GB のアクティベーションを扱う(`--memory=200g` で実行)。

### 3.2 フルグラフの全デジタル実行

`tools/digital_ppa/run_full_digital.py`(本リポジトリに格納)を使う。

```bash
docker exec mythic_digital_ppa \
  /mythic/pyvnnsdk-env/bin/python /work/run_full_digital.py /work/out_full
```

所要 ~4 分(ビルド/inline/shape inference ~10 秒、量子化統計収集 ~115 秒、capnp export ~1 秒、exploration ~7 秒)。出力:

```
out_full/BevformerTiny.vidi.onnx   165 KB   (initialized)
out_full/BevformerTiny.vidi.dat    228 MB   (weights)
out_full/BevformerTiny.vido.onnx   165 KB   (optimized)
out_full/BevformerTiny.vidir       257 MB   (capnp、vnnmap の入力)
out_full/BevformerTiny_proflings.json
```

### 3.3 `[sys]` パラメータのスイープ

`.vidir` は再利用できる。量子化をやり直す必要はない。`tools/digital_ppa/sweep_system_config.py`:

```bash
docker exec mythic_digital_ppa \
  /mythic/pyvnnsdk-env/bin/python /work/sweep_system_config.py \
  /work/out_full/BevformerTiny.vidir [cases.json]
```

1 ケース ~7 秒。`run_vnnmap()` を直接呼ぶので §2.1 の全メトリクス(DDR R/W、exposed DMA cycles 等)が取れる。

---

## 4. 実測値

### 4.1 出荷スクリプトの値はモデル全体ではない

`vnnsdk_scripts/bevformer/bevformer_tiny.py` は `TRANSFORMER_PART_ONLY: bool = True` を既定にしており、`_input_shapes()` は `img_features [1,256,1450,6]` を入力とする。**ResNet-50 backbone が含まれていない。**

したがって PPA アーティファクト内の `BevformerTiny_proflings.json`(例: `bevformer_m2072_high_2605_2.tar.gz` 内 `artifacts/firmware/vnn/`)の値は transformer 部分のみを表す。全デジタル値としては使えない。

| 項目 | transformer のみ | **フルグラフ(全デジタル)** | 比 |
|---|---|---|---|
| MACs | 16.529 bn | **904.354 bn** | 54.7× |
| Parameters | — | 57.195 mn | — |
| Model size | 13.622 MB | 54.611 MB | 4.0× |
| Total cycles | 10,116,764 | 117,605,925 | 11.6× |
| MAC cycles | — | 67,599,073 | — |
| non-MAC cycles | — | 2,017,696 | — |
| exposed DMA cycles | — | 47,989,156 | — |
| fps / latency | 216.15 / 4.63 ms | **16.99 / 58.87 ms** | 12.7× |
| MAC 利用率 | 9.69 % | 41.67 % | — |
| Power@eff.fps | 8.968 W | 14.485 W | 1.6× |
| Power@30fps | 1.245 W | **25.582 W** | 20.5× |
| max DDR | — | 439,778 kB | — |
| DDR Read / Write | — | 1951.2 MB / 1646.2 MB | — |

いずれも既定 `system_configs/bevformer.cfg`(`mCluster=1 pCluster=12 nMPs=288 OCRAM0=32MB OCRAM1=1MB DDR=100GB frequency=2GHz nBatch=1 xTile=2`)。

MACs が 54.7 倍なのに cycles が 11.6 倍で済んでいるのは、transformer 部分の MAC 利用率が 9.69% しかなく(Attention/Reshape/GridSample が支配的でデジタル MAC アレイを埋められない)、backbone の Conv がむしろ利用率を 41.67% まで押し上げているため。

### 4.2 ハイブリッドとの比較

| 構成 | Latency | Power@30fps | Area |
|---|---|---|---|
| ハイブリッド m2072(既知の可行点) | **31.58 ms** | **4.505 W** | 380 mm²(アナログ側、5.278 mm²/ACE × 72) |
| 全デジタル・既定構成 | 58.87 ms | 25.58 W | 不明(§6) |
| 全デジタル・最良構成(OCRAM0=512MB) | 33.19 ms | 18.27 W | 不明(かつ非現実的) |

**全デジタルは 33 ms 制約を既定構成では満たせない。** 満たせる構成も存在するが、要求されるオンチップ SRAM 容量が現実的でない。

### 4.3 `[sys]` スイープ結果(フルグラフ `.vidir`、n=16)

nMPs のみを振った系列:

| nMPs | freq | fps | latency | 利用率 | P@eff.fps | P@30fps | DDR R |
|---|---|---|---|---|---|---|---|
| 288 | 2 GHz | **16.99** | 58.87 ms | 41.67 % | 14.49 W | 25.58 W | 1951 MB |
| 576 | 2 GHz | 5.99 | 167.03 ms | 7.34 % | 5.50 W | 27.54 W | 2756 MB |
| 1152 | 2 GHz | 3.75 | 266.33 ms | 2.30 % | 4.19 W | 33.46 W | 5678 MB |
| 2304 | 2 GHz | 4.02 | 248.65 ms | 1.23 % | 5.00 W | 37.27 W | 5697 MB |
| 288 | 1 GHz | 8.49 | 117.74 ms | 41.67 % | 7.24 W | 25.58 W | 1951 MB |

**nMPs を増やすと悪化する。** MAC アレイを増やしても exposed DMA が支配的なため fps は下がり、DDR トラフィックが増えて `Power@30fps` も増える。デジタル実行のボトルネックは MAC 数ではなくメモリ帯域である。

メモリ・タイル構成を振った系列(すべて nMPs=288 / 2 GHz 基準):

| tag | 変更点 | fps | latency | 利用率 | P@eff.fps | P@30fps | DDR R | DMA cyc |
|---|---|---|---|---|---|---|---|---|
| baseline | — | 16.99 | 58.87 ms | 41.67 % | 14.49 W | 25.58 W | 1951 MB | 47.99 M |
| oc1_4M | OCRAM1 1→4 MB | 18.95 | 52.77 ms | 46.49 % | 16.10 W | 25.49 W | 1893 MB | 35.79 M |
| oc1_16M | OCRAM1 1→16 MB | 18.95 | 52.77 ms | 46.49 % | 16.10 W | 25.49 W | 1893 MB | 35.79 M |
| oc0_128M | OCRAM0 32→128 MB | 21.99 | 45.48 ms | 53.94 % | 15.76 W | 21.50 W | 842 MB | 23.94 M |
| **oc0_512M** | OCRAM0 32→512 MB | **30.13** | **33.19 ms** | **73.91 %** | 18.35 W | **18.27 W** | **5.1 MB** | — |
| oc0_512M_f4G | 上記 + 4 GHz | 60.26 | 16.59 ms | 73.91 % | 36.71 W | 18.27 W | 5.1 MB | — |
| oc0_128M_f3G | OCRAM0 128MB + 3 GHz | 32.98 | 30.32 ms | 53.94 % | 23.64 W | 21.50 W | 842 MB | — |
| oc0_128M_576 | OCRAM0 128MB + nMPs576 | 36.00 | 27.78 ms | 44.16 % | 26.30 W | 21.92 W | 742 MB | — |
| xt1 | xTile 2→1 | 15.40 | 64.92 ms | 37.79 % | 12.80 W | 24.92 W | 1935 MB | 69.33 M |
| xt4 | xTile 2→4 | 3.35 | 298.20 ms | 8.23 % | 3.68 W | 32.90 W | 5927 MB | 521.89 M |
| pc24 | pCluster 12→24, nMPs 576 | 16.67 | 59.98 ms | 20.45 % | 15.06 W | 27.11 W | 2066 MB | 82.24 M |

観察:

1. **効くのは OCRAM0(オンチップ SRAM)だけ。** 512 MB にすると DDR Read が 1951 MB → 5.1 MB、DDR Write が 1646 MB → 0 MB になり、モデル全体+アクティベーションがオンチップに載る。利用率 41.67% → 73.91%。`Power@30fps` も 25.58 → 18.27 W へ下がる(DDR アクセスエネルギーが消える分)。
2. **OCRAM1 は 4 MB で飽和。** 4 MB と 16 MB が完全同値(fps/power/DMA cycles すべて一致)なので、4 MB 以上は使われない。
3. **nMPs / pCluster / xTile は増やすと悪化する。** xTile=4 は exposed DMA が 521.89 M cycles(baseline の 10.9 倍)まで爆発する。
4. **周波数は素直にスケールする。** 2→3 GHz で fps 1.94×、latency 0.667×、利用率不変。`Power@eff.fps` は fps 比でほぼ線形に増え、`Power@30fps` は不変(§5.2 の外挿仕様どおり)。
5. **どの構成でも `Power@30fps` は 18.27 W が下限。** ハイブリッドの 4.505 W に対し最良でも約 4 倍。

---

## 5. デジタル側電力モデル

### 5.1 モデルの構成要素

`vnnmap` バイナリ内の文字列から、per-component の pJ モデルであることが確認できる:

```
powDdrReadPj / powDdrWritePj
powOcrMReadPj / powOcrMWritePj / powOcrCReadPj / powOcrCWritePj
powDmemReadPj / powDmemWritePj / powDmem1..3ReadPj / powDmem1..3WritePj
powImemReadPj / powImemWritePj
powMacUnitPjPerCycle / powNonMacUnitPjPerCycle
powBusExtDdrIfPj / powBusMatrixMPj / powBusMatrixCPj / powBusMatrixCIntPj / powBusMatrixCExtPj
powNocPj / powNocIntraPj / powNocInterPj
```

アナログ側の `power_estimator.py` とは**完全に別実装**で、DDR/OCRAM/DMEM/IMEM のアクセス回数 × pJ、MAC/non-MAC の cycle 数 × pJ/cycle、バス・NoC のトランザクション × pJ の総和という構造。

**これらは全て `[sys]` の設定キーである。** `parseSystemCfg`(`0x68c90`)内のラムダ(`0x66be0`)が
`INIReader::GetReal` で読み、`vNetwork` のフィールドへ格納する(例: `powDdrReadPj` → `+0xc60`、
`powDdrWritePj` → `+0xc68`)。cfg に書けば既定係数を**上書きできる**。

> 旧記述の訂正: 本節は以前「係数はバイナリ内に埋め込まれており Python 側からは変更できない」としていたが、
> 誤りである。既定値がバイナリ内にあるだけで、cfg から差し替え可能。

これは調査手段として有用で、**1 係数だけを 1.0 にし残りを 0 にすれば、対応するアクセス回数/cycle 数の
生の値を `Power@eff. fps` の数値として読み出せる**(§9.2)。電力内訳の分解に逆アセンブルは不要になる。

### 5.2 `Power@30fps` と `Power@eff.fps` の違い

`Power@eff.fps` は実際に出る fps(`eff. fps`)でのエネルギー消費率。`Power@30fps` は同じ 1 推論エネルギーを 30 fps に**外挿**した仮想値。

この構造は逆アセンブルで確定済み(`0x197768`-`0x197850`)。1 推論の総エネルギーを

```
energy_pJ = mem_energy_pJ() + bus_energy_pJ() + imem_energy_pJ()
          + mac_unit_energy_pJ() + nonmac_unit_energy_pJ()      # すべて vNetwork+0xa88 に対して
```

で求め(`vNetwork+0xa88` はネットワーク全体の `PowerProfile`。層ローカルは `+0x2a0`/`+0x300`)、

```
Power@eff. fps ∝ energy_pJ × eff_fps × 1e-9
Power@30fps    ∝ energy_pJ × 30      × 1e-9        # 30.0 は 0x245b50、1e-9 は 0x245b58
```

とする。**`Power@30fps` は per-inference エネルギーの単純な線形外挿であり、30 fps で動作した場合の
電力を物理モデル化したものではない**(§4.3 で周波数を変えても不変な理由)。
アクセス回数の記録は `PowerProfile::profile_transfer`(`0x18cbd0`)、帯域は
`vBandwidthMatrix::record_transfer`(`0x18cf40`)が `readDMA` から層ローカル/ネットワーク全体の
2 か所に対して呼ぶ。

`eff. fps < 30` の構成では `Power@30fps` は「そのハードでは実現不可能な動作点の電力」を表す。§4.3 で周波数を上げても `Power@30fps` が不変なのはこのため(1 推論のエネルギーが変わらない)。**基準 fps が異なる値を混在させて比較しない。** ハイブリッド側の 4.505 W@30fps と比較するなら `Power@30fps` 同士で比べる。

### 5.3 除外項

アナログ側 estimator と同様、leakage / clock tree / PCIe / D2D は "5nm typ" のダイナミック電力見積りに含まれていない([02_ppa_estimation.md](02_ppa_estimation.md) §4.11 でアナログ側について確認済みの構造)。

§5.2 の逆アセンブルで**確定**した。総和に入るのは mem / bus / imem / mac unit / non-mac unit の
5 項のみで、`[sys]` の `pow*Pj` キーにも leakage / clock tree / PCIe / D2D に相当するものは無い
(全キーはアクセス回数あたり pJ か cycle あたり pJ)。したがって `Power@30fps` は
**純粋なダイナミック電力**であり、リーク・クロックツリー・チップ間 I/O は含まない。
アナログ側と同じ除外構造なので、`Power@30fps` 同士の比較は同条件で成立する。

---

## 6. 面積(A)が出ない理由

`vnnmap` バイナリの文字列に **area / mm² / die に相当するシンボルは存在しない**(唯一のヒットは capnp の `asDataReader` で無関係)。出力も cycles / fps / power / メモリ使用量のみ。

- アナログ側: `perf_analysis.py` に面積計算があり、物理係数 **5.278 mm²/ACE**(= 380/72)が既知([PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) §4.1、ツール表示の 10.545 mm²/ACE はその 2 倍で物理値ではない)。
- デジタル側: v-MP コア 1 個あたりの面積が SDK のどこにも露出していない。`nMPs`(MAC アレイ数)・`pCluster`(processing cluster 数)は構成として与えられるが、それを面積に変換する係数がない。

したがって**全デジタル構成の面積は SDK 外の情報(v-MP コア面積 × コア数 + SRAM マクロ面積)でしか見積もれない**。特に §4.3 で最良だった OCRAM0=512 MB 構成は、5nm の SRAM マクロ面積を考えると面積・コストの観点で成立しない可能性が高い(512 MB SRAM は数百 mm² 規模)。**この点が全デジタル評価の最大の未確定要素である。**

---

## 7. フルグラフを通すために必要だった SDK 側の回避策

出荷状態の SDK では `TRANSFORMER_PART_ONLY=False` は通らない。2 箇所で落ちる。

### 7.1 ONNX local function の重複

```
onnx.onnx_cpp2py_export.checker.ValidationError: Model contains multiple local functions
with the same implementation id 'com.videantis.dynamic_functions.ResnetBlock0::ResnetBlock'
```

`bevformer_tiny.py::initialize_onnx()` は、onnxscript グラフが既に持っている local function の上に、キャッシュ済みの `com.videantis.dynamic_functions` 関数を**全部 append** してから `inline_selected_functions` を呼ぶ。transformer のみの場合は衝突しないが、backbone を含めると `ResnetBlock` 等が二重登録され checker が落ちる。

回避: `bevformer.bevformer_tiny.inline_selected_functions` をモンキーパッチし、`(domain, name)` で dedupe してから元の inliner に渡す(`tools/digital_ppa/run_full_digital.py`)。

### 7.2 Cap'n Proto の blob 上限超過

```
capnp.lib.capnp.KjException: capnp/layout.c++:1724: failed: text blob too big
  at vnnmap/network.py:287  (tensor.data = data_bytes)
  via vnnort/utils/vnnmap_export.py:143  (self._network.add_tensor(...))
```

量子化後の capnp export で `x_0__1 [1,64,450,4800]` = **552,960,000 バイト**(fp32)のテンソルを 1 個の blob に書こうとして、Cap'n Proto の 512 MB 上限を超える。

重要なのは**これが重みではなくアクティベーション(dynamic)**であること。`vnnmap/network_schema.capnp:248-249` は `Tensor.data` を

> For TensorType static this contains the weight data of the corresponding tensor. Otherwise this is not set.

と定義しているが、`vnnmap_export.py::_add_tensors` (`:105-153`) は `_tensor_quant_infos` の全エントリについて無条件に `data=tensor.data` を渡しており、キャリブレーション 1 サンプル分のアクティベーションが dynamic テンソルにも載ってしまう。フルグラフの入力は `images [1,3,900,9600]`(6 カメラを横連結)で、最初の Conv 出力が 553 MB になる。

回避: `CapnprotoNetwork.add_tensor` をモンキーパッチし、`tensor_type != static` の場合に `data=None` にする(`tools/digital_ppa/run_full_digital.py`)。schema の定義どおりの挙動に戻すだけなので、`vnnmap` 側の解釈には影響しない(exploration は正常完了し、レイヤごとの cycle/DDR 見積りも全レイヤ分出ている)。

**[未検証]** SDK 本体側の修正としては `vnnmap_export.py:143` で `tensor_type` を見て data を渡し分けるのが筋。ベンダーへの報告候補。

---

## 8. 数値の留保事項

| 留保 | 内容 |
|---|---|
| 精度は未検証 | `skip_validation=True` + `calibration_dataset_size=1`(ランダム 1 サンプル)で実行。得られた値は cycles/power の見積りにのみ使える。全デジタル実行時の精度は別途評価が必要 |
| 量子化は INT8/INT16 前提 | v-MP のデータ型([Datasheet v0.3](../../mythic_sdk/v26.05.0/doc/datasheets/Mythic_M2000_NPU_Datasheet_v0.3.pdf))。アナログ ACE の ANA8 とはノイズ特性が全く異なるので、精度はハイブリッドより良くなる方向 |
| `Power@30fps` は外挿値 | §5.2。16.99 fps しか出ない構成の 25.58 W@30fps は実現不可能な動作点の値 |
| 除外項はダイナミック電力のみ | §5.3。leakage / clock tree / PCIe / D2D は非含有(確定) |
| `efficiency` は `nMACs` 非依存 | §2.2。64 MAC/cycle/MP 固定で計算するので、`nMACs` を既定 32 から変えると表示値が真の利用率とずれる |
| `Total` cycles ≠ レイテンシの分母 | §2.4。両者は別系統の集計で 0.125% ずれる。cycles と latency を掛け合わせた再計算はしない |
| DDR レイテンシは cycle 単位 | §2.3。`frequency` を上げると DDR アクセス時間も比例して縮む扱いになり、高周波側は楽観側に振れる |
| 面積が出ない | §6。全デジタルの A は SDK 外の情報が必須 |
| バッチ 1 固定 | `vnnmap` は dynamic batch 非対応。`vnnmap_export.py:136-138` が batch=-1 を 1 に固定する |
| `--edma` の効果は未分離 | `advanced=True` は `--explore --edma` を同時に付ける(`run_vnnmap.py:236-238`)。EDMA 有効/無効の差分は測っていない |

---

## 9. 次に調査する場合の起点

### 9.1 すぐ再現できる状態

- スクリプト: `tools/digital_ppa/run_full_digital.py`(フルグラフ実行)、`tools/digital_ppa/sweep_system_config.py`(`[sys]` スイープ)、`tools/digital_ppa/probe_vnnmap_cycles.py`(gdb で cycle 集計を読み出し §2.2-§2.4 を検証)
- コンテナ: §3.1 の 1 コマンドで作れる。`compilerd-bin:v26.05.2` のみで完結
- `.vidir` を一度作れば(~4 分)スイープは 1 ケース 7 秒

### 9.2 未解決の課題(優先順)

1. **面積の外部見積り(§6)。** これが埋まらないと PPA の A が空欄のままで、ハイブリッドとの正当な比較ができない。必要なのは (a) v-MP コア 1 個の 5nm 面積、(b) OCRAM0/OCRAM1 の SRAM マクロ面積/MB。いずれも SDK 外。Datasheet の 12 タイル構成と 370 TOPS から逆算する余地はあるが、v-MP 単体の面積分離は資料次第。
2. **OCRAM0 の現実的上限の確定。** §4.3 の結論は「512 MB あれば 33 ms を満たす」だが、実チップの OCRAM0 は 32 MB(`bevformer.cfg` の既定値)。32 MB と 512 MB の間(64/128/256 MB)での Pareto を取れば、「どこまで SRAM を積めば全デジタルが成立するか」の定量的な答えになる。128 MB で 45.48 ms(§4.3)なので、33 ms 到達は 256〜512 MB のどこか。
3. **他モデルへの適用。** YOLOPX は `system_configs/yolopx.cfg`(nMPs=1 / 1 GHz)で transformer 分割の問題がなく、フルグラフがそのまま通る可能性が高い(出荷値: macs_bn=0.089 / 544.84 fps / 39.0 mW@eff / 2.15 mW@30fps / 利用率 75.79%)。BEVFormer より全デジタル化の見込みがある。ハイブリッドでは 48 ACE が最適([PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md))なので、全デジタルとの比較対象としても適切。
4. **exposed DMA が支配的な理由の分解。** `<model>_flow.csv`(`vnnmap/csv/` に出る)にレイヤごとの cycle/DDR 内訳がある。どのレイヤが DDR トラフィックを生んでいるかを特定すれば、モデル側の改変(backbone の軽量化・チャネル削減)で全デジタルが成立するかを判断できる。BEVFormer フルグラフの exposed DMA は 47.99 M cycles / 総 117.6 M cycles = 41%。§2.3 の DMA モデル(`readDivDDR` / `readLatencyDDR` / `dmaThreshold`)が cfg から可変なので、**DMA モデル自体のスイープで感度を直接測れる**。特に `dmaThreshold`(既定 765)は隠蔽量を決めるので、`nMPs` を増やすと悪化する現象(§4.3 観察 3)の主因かどうかを切り分けられる。
5. **電力内訳の分解。** §5.1 のとおり `pow*Pj` は `[sys]` キーなので、1 係数だけ 1.0・残り 0 の cfg を投げれば各コンポーネントの生アクセス回数が `Power@eff. fps` として読み出せる。25.58 W@30fps のうち DDR / OCRAM / MAC / bus が何 W ずつかを確定でき、「OCRAM0=512 MB で 18.27 W まで下がる」の内訳も検証できる。逆アセンブル不要。
6. **精度評価。** INT8 デジタル実行の精度は、アナログ ANA8 + ノイズ([03_accuracy_simulation.md](03_accuracy_simulation.md))より良いはずだが未測定。全デジタルが「電力 4 倍だが精度が大幅に良い」なら、トレードオフの議論として意味を持つ。

`vnnmap` の逆アセンブル(旧項目 5)は完了。結果は §2.2-§2.5 / §5.1-§5.3。

### 9.3 参照

- [02_ppa_estimation.md](02_ppa_estimation.md) — アナログ側 estimator の完全解析。§3.8 にデジタル JSON の合算方法(本文書 §2.2 がその `npu_fps` の出所)、§5 に面積係数
- [00_overview.md](00_overview.md) §3.5 — アナログ/デジタル分割のレベル A / B
- [PLAN_bevformer_ppa_exploration.md](PLAN_bevformer_ppa_exploration.md) — ハイブリッド側の探索結果(m2072 が唯一の可行 SKU)
- [PLAN_yolopx_ppa_exploration.md](PLAN_yolopx_ppa_exploration.md) — 同 YOLOPX(48 ACE が最適)
- `mythic_sdk/v26.05.0/_extracted_compiler/vnnsdk_scripts/mythic_utils.py` — `run_vnn_flow` / `PROFILING_FIELDS`
- `mythic_sdk/v26.05.0/_extracted_compiler/vnnmap/run_vnnmap.py` — `_extract_metrics` / コマンドライン組み立て
- `mythic_sdk/v26.05.0/_extracted_compiler/vnnmap/network_schema.capnp` — `Tensor` / `TensorType` 定義
- `mythic_sdk/v26.05.0/_extracted_compiler/vnnort/utils/vnnmap_export.py` — capnp export(§7.2 の問題箇所)
- `mythic_sdk/v26.05.0/_extracted_compiler/vnnsdk_scripts/bevformer/bevformer_tiny.py` — `TRANSFORMER_PART_ONLY`
- `mythic_sdk/v26.05.0/doc/datasheets/Mythic_PPA_Estimator_Datasheet_v0.4.pdf` — §8.1/8.3 に v-MP 上のデジタルノードに関する記述
- `mythic_sdk/v26.05.0/_extracted_compiler/vnnmap/vnnmap` — cycle / 電力モデル本体。strip されておらず(4396 シンボル)DWARF は無い。`objdump -dC` + シンボル名で追える(§2.2-§2.5)。主要アドレス: `printFullProfile` `0x195eb0` / `vSummaryProfile::fill(vNetwork*)` `0x1958d0` / `setMpCycles` `0x19c090` / `readDMA` `0x1682d0` / `parseSystemCfg` `0x68c90`
