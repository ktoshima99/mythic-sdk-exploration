# Mythic M2000 SDK: 性能・電力・面積(PPA)推定器の完全解析

本ドキュメントは Mythic M2000 AI アクセラレータ SDK に含まれる 2 つの推定スクリプトを、
式レベル・行番号レベルで完全にリバースエンジニアリングしたものである。

対象ソース(読み取り専用):

- 性能: `mythic_sdk/v26.05.0/_extracted_compiler/perf_analysis.py`(全 1329 行)
- 電力: `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/m2000_power_estimator/power_estimator.py`(全 649 行)
- protobuf 定義: `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/irs/l0/ir_pb2.py`, `parameters_pb2.py`

以降、「[perf:NNN]」は `perf_analysis.py` の行番号、「[pow:NNN]」は `power_estimator.py` の行番号を指す。

---

## 1. 概要

M2000 は「アナログ NPU(ACE アレイ、28nm)」+「デジタル NPU(オプション、5nm)」+ デジタル
補助ロジック(SRAM・アクセサ・制御・NOC)からなる混載チップである。PPA 推定は 2 本のスクリプトに分離している。

| 推定器 | 入力 | 出力 | 主な仮定 |
|---|---|---|---|
| `perf_analysis.py`(性能・面積) | HDF5 トレース `perf_trace_dump.h5` | レイテンシ / fps / ACE 利用率 / SRAM・SIMD 時間 / ダイ面積 | 各 timestep の律速要素の総和がレイテンシ |
| `power_estimator.py`(電力) | L0 protobuf(`Crate`)+ 任意の packet/event/NPU ログ | W(消費電力)/ TOPS/W | 1 推論エネルギー × 推論レート(fps)= 電力 |

両者は独立プログラムであり、性能側の面積推定は ACE アレイのみを対象とする。電力側はアナログ
(28nm 固定)+デジタル(5/12/28nm 選択)を分けて算出する。

> **入力の粒度に関する注意(重要)**: 上表の「入力」は各 **スクリプト単体** から見た入力である。
> PPA 推定フロー全体(`ppa-estimator --estimate-performance`)の視点では区別が要る:
> - **`final.l0.pb`(L0 protobuf)はコンパイラの出力**。コンパイル時点で確定する静的 IR で、
>   PPA 推定フローに **外部から与えられる入力**。`power_estimator.py` はこれを直接読む。
> - **`perf_trace_dump.h5` はコンパイラの出力ではない**。`--estimate-performance` 実行時に
>   フロー内部で `funcsim`(機能シミュレータ、CPU 逐次イベントドリブン)がコンパイル成果物を
>   実行して **その場で生成する中間トレース**であり、`artifacts/ppa/` 配下に書き出される。
>   したがって **PPA 推定フロー全体から見れば `.h5` は外部入力ではなく内部生成物**であり、
>   フローへの外部入力はあくまでコンパイル成果物(tar)である。`.h5` が「入力」なのは、
>   その内部生成物を読む **`perf_analysis.py` スクリプト単体の視点に限った話**である。
>
> まとめると、フローへの外部入力は「コンパイル成果物」1 つで、`funcsim` が `.h5` を生成 →
> `perf_analysis.py` が `.h5` を、`power_estimator.py` が `final.l0.pb` を読む、という直列関係。

---

## 2. 入力データ(HDF5 構造)

> 本節の「入力」は `perf_analysis.py` スクリプト単体から見た入力である。この HDF5
> (`perf_trace_dump.h5`)は `--estimate-performance` フロー内で `funcsim` が生成する中間トレース
> であり、コンパイラの出力でもフローへの外部入力でもない(§1 の注意、及び §9 を参照)。

`perf_analysis.py` の docstring [perf:33-70] と実パース処理から、HDF5 の全データセット構造は次の通り。

```
/
├── ace_calcs/
│   └── ace_tile_N/          (N=タイル番号)
│       ├── trans_id
│       ├── timestep
│       ├── num_inputs        ← ACE 演算の入力数(MAC 数算出に使用)
│       └── num_outputs       ← ACE 演算の出力数
├── simd_calcs/
│   ├── ace_tile_N/
│   │   ├── trans_id
│   │   ├── timestep
│   │   ├── num_input_bytes           ← SIMD 時間算出に使用
│   │   ├── num_secondary_input_bytes
│   │   └── num_output_bytes
│   └── host_interface_tile/          (同上フィールド)
└── sram_accesses/
    ├── ace_tile_N/
    │   ├── trans_id
    │   ├── timestep
    │   ├── address
    │   ├── size                  ← アクセスバイト数(バイト法に使用)
    │   ├── estimated_final_size   (制御構造アクセスの推定最終サイズ)
    │   ├── access_type            (read=0, write=1)
    │   ├── initiator              (文字列)
    │   ├── hwu_id
    │   ├── is_rmw                 (read-modify-write)
    │   ├── category   (0=なし, 1=データバッファ, 2=制御フロー, 3=管理, 4=デバッグ)
    │   ├── control_flow_type
    │   └── num_accesses           ← アクセス法に使用(新形式トレースのみ)
    └── host_interface_tile/       (同上フィールド)
```

実際にコードが読むフィールド:

- SRAM [perf:364-371]: `access_type`, `size`, `num_accesses`(`USING_NUM_ACCESSES=True` [perf:250] のとき。旧形式では `[1]*len` [perf:369]), `category`, `timestep`
- ACE [perf:430-431]: `num_inputs`, `num_outputs`
- SIMD [perf:451,456]: `timestep`, `num_input_bytes`

注意: docstring には `num_accesses` は明記されていないが、コード [perf:367] は新形式トレースで
このデータセットを必須として読む。`category` は 1(データ)と 2(制御)のみ集計対象で、0/3/4 は
`other_bytes_per_tile` に加算されるのみで律速計算には使われない [perf:392-411]。

### デジタル NPU JSON(任意)[perf:264-280]

`--digital-npu-log-path` 指定時に読む。参照キー:
`profiling.fps`, `profiling.total_cycles`, `profiling.mac_utilization_pct`,
`profiling.macs_bn`, `system_config.sys.frequency`, `system_config.sys.nmps`。

---

## 3. 性能推定アルゴリズム(式)

### 3.1 定数(トポロジ)[perf:286-288]

```
PER_CHIP_ACE_COUNT     = args.num_aces  (デフォルト 24, 4 の倍数を強制 [perf:92-97,113-117])
PER_ACE_TILE_COUNT     = 4              (1 タイルあたり ACE 数)
PER_CHIP_ACE_TILE_COUNT= num_aces // 4  (デフォルト 24//4 = 6 タイル)
```

### 3.2 パース段階での集計

**SRAM 集計** [perf:373-411]: 各アクセス行 i について:

- `ts_accesses[tile][timestep]["all_accesses"] += num_accesses[i]` [perf:375]
- read(access_type=0)/write(=1) × control(category=2)/data(category=1) の 4 象限で
  バイト数(`+= size[i]`)とアクセス数(`+= num_accesses[i]`)を別々に蓄積 [perf:376-389]
- 同時にタイル単位の総計 `*_per_tile` も蓄積 [perf:392-411]

**ACE 集計** [perf:426-436]: ACE の各演算について

```
ace_operations += 1                              [perf:435]
ace_mac_count  += int(num_inputs) * int(num_outputs)   [perf:436]
```

MAC 数は「入力数 × 出力数」で近似(アナログクロスバーの積和数)。

**SIMD 集計** [perf:447-456]: timestep ごとに `simd_operations += 1`、
`simd_input_bytes += num_input_bytes[i]`。

### 3.3 timestep ごとの所要時間(2 手法並行)[perf:511-755]

各 timestep で「全タイルにわたる最大値」を取る(合算ではない)[perf:498-499,517]。
これは「1 つの timestep 内では全タイルが並列動作し、最も遅いタイルが律速する」という仮定。

#### バイト法(bytes method)

各タイルの当該 timestep の read/write バイトを data+control で合算した「タイル別最大」を求める。

```
max_total_read_bytes  = max_read_data_bytes  + max_read_control_bytes    [perf:612-615]
max_total_write_bytes = max_write_data_bytes + max_write_control_bytes   [perf:616-619]

sram_read_estimated_duration_ns  = max_total_read_bytes  / SRAM_BYTES_READ_PER_CYCLE  * CLOCK_PERIOD_NS   [perf:620-622]
sram_write_estimated_duration_ns = max_total_write_bytes / SRAM_BYTES_WRITTEN_PER_CYCLE * CLOCK_PERIOD_NS  [perf:637-640]
```

コード内コメント [perf:610-611] に「これはおそらく正しくない。本来は各タイルで先に合算してから
max を取るべき」と作者(ptoth)自身が注記している(=既知の近似の限界)。

#### アクセス法(accesses method)

read/write を区別せず全アクセス数の「タイル別最大」を使う。RTL は read/write に関わらず
`SRAM_ACCESS_PORTS` 個まで並列に処理できるため、より正確とされる [perf:657-660]。

```
sram_estimated_duration_accesses_ns = max_accesses_by_any_tile / TOTAL_SRAM_ACCESSES_PER_CYCLE * CLOCK_PERIOD_NS   [perf:661-664]
```

**バイト法 vs アクセス法の違い(要点):**

| 観点 | バイト法 | アクセス法 |
|---|---|---|
| 単位 | バイト数(`size`) | アクセス回数(`num_accesses`) |
| read/write | 別々に時間算出し個別 max | 合算(ポート共有前提) |
| 帯域係数 | `SRAM_PERFORMANCE_SCALING_FACTOR=0.5` | `SRAM_ACCESSES_PERFORMANCE_SCALING_FACTOR=0.6` |
| ポートモデル | 128 B/cycle の理想帯域を 0.5 倍 | 8 ポート × 0.6 |
| 精度評価 | 旧来手法 | RTL に近く「より正確」とコメント [perf:657-660] |
| 適用 | 旧形式トレース対応 | 新形式(`num_accesses` 必須) |

新形式トレースでは両方が計算・出力される(`USING_NUM_ACCESSES=True` [perf:250])。

#### SIMD 所要時間 [perf:682-698]

```
simd_estimated_duration_ns = simd_input_bytes / SIMD_VECTOR_WIDTH * CLOCK_PERIOD_NS   [perf:691-693]
   (SIMD_VECTOR_WIDTH = 8 レーン)
longest_simd_estimated_duration_ns = 各 SIMD 名にわたる max   [perf:695-698]
total_simd_usage_ns += simd_estimated_duration_ns  (全タイル・全timestepの合算) [perf:694]
```

#### timestep 所要時間(律速)[perf:705-747]

```
# バイト法
timestep_duration_ns = max(ACE_DURATION_NS,                       # =160ns 下限
                           sram_read_estimated_duration_ns,
                           sram_write_estimated_duration_ns,
                           longest_simd_estimated_duration_ns)     [perf:705-710]
total_duration_ns += timestep_duration_ns                          [perf:715]

# アクセス法
timestep_duration_accesses_ns = max(ACE_DURATION_NS,
                                    sram_estimated_duration_accesses_ns,
                                    longest_simd_estimated_duration_ns)  [perf:724-728]

# SIMD を無視した参考値
timestep_duration_accesses_no_simd_ns = max(ACE_DURATION_NS, sram_estimated_duration_accesses_ns)  [perf:736-739]
```

各 timestep は最低でも `ACE_DURATION_NS=160ns` かかる(ACE 演算 1 回=1 timestep)。
「超過分(excess)」は律速要素別に `total_excess_*` に振り分けられる [perf:717-722, 749-755]
(どの max 項が採用されたかで分類。デバッグ出力専用の内訳)。

### 3.4 最終ボトルネックレイテンシ [perf:757-895]

epoch ベースの `total_duration_ns`(timestep 総和)とは別に、「チップ全体の律速」を次で求める
(これが公表レイテンシ)。

**ACE クリティカルパス** [perf:771]:
```
total_ace_duration_ns = len(timesteps) * ACE_DURATION_NS
```
= timestep 数 × 160ns(全 ACE 演算が直列に 1 timestep ずつ進む最小パス)。

**タイル別最大 SRAM 時間(バイト法)** [perf:844-859]:
```
max_bytes_read_by_any_tile  = max_tile( data_bytes_read_per_tile[t] + control_bytes_read_per_tile[t] )
max_sram_read_duration_ns   = max_bytes_read_by_any_tile  / SRAM_BYTES_READ_PER_CYCLE  * CLOCK_PERIOD_NS
max_sram_write_duration_ns  = max_bytes_written_by_any_tile / SRAM_BYTES_WRITTEN_PER_CYCLE * CLOCK_PERIOD_NS
```

**タイル別最大 SRAM 時間(アクセス法)** [perf:869-876]:
```
max_accesses_by_any_tile        = max_tile( data_accesses_per_tile[t] + control_accesses_per_tile[t] )
max_sram_duration_accesses_ns   = max_accesses_by_any_tile / TOTAL_SRAM_ACCESSES_PER_CYCLE * CLOCK_PERIOD_NS
```

**タイル別最大 SIMD 時間** [perf:879-881]:
```
max_simd_usage_by_any_tile = max(simd_bytes_per_tile.values()) / SIMD_VECTOR_WIDTH * CLOCK_PERIOD_NS
```

**ボトルネック(公表レイテンシ)** [perf:883-895]:
```
maximum_bottleneck_ns = max(total_ace_duration_ns,
                            max_sram_read_duration_ns,
                            max_sram_write_duration_ns,
                            max_simd_usage_by_any_tile)              # バイト法 [perf:883-888]

maximum_bottleneck_accesses_ns = max(total_ace_duration_ns,
                                     max_sram_duration_accesses_ns,
                                     max_simd_usage_by_any_tile)     # アクセス法 [perf:891-895]
```

### 3.5 fps とレイテンシ [perf:918-946]

```
frame_latency_ns = maximum_bottleneck_ns              (または _accesses_ns)  [perf:925,940]
fps = 1e9 / maximum_bottleneck_ns                                            [perf:926,941]
```

`format_time` [perf:908-910] は総時間 < 1e6 ns なら us、以上なら ms 表記に切替 [perf:901-906]。
バイト法とアクセス法で 2 回出力される(それぞれ `sram_method` ログ属性でフィルタ [perf:135-162])。

### 3.6 ACE 利用率・理論時間 [perf:1085-1115]

```
total_ace_ops = Σ 全 ACE の ace_operations                                       [perf:1083]
total_macs    = Σ 全 ACE の ace_mac_count                                        [perf:1084]

最大理論 ACE 実行時間(1 ACE, 並列化なし)= total_ace_ops * ACE_DURATION_NS       [perf:1087-1090]
最小理論 ACE 実行時間(PER_CHIP_ACE_COUNT で均等並列)
                                  = total_ace_ops * ACE_DURATION_NS / PER_CHIP_ACE_COUNT   [perf:1091-1095]

ACE Utilization = (total_ace_ops * ACE_DURATION_SEC / PER_CHIP_ACE_COUNT)
                  / (maximum_bottleneck_ns / 1e9) * 100                          [perf:1097-1102]
```
= 最小理論時間 ÷ 推定処理時間 × 100(%)。バイト法・アクセス法それぞれで算出 [perf:1107-1115]。
`ACE_DURATION_SEC = 160/1e9` [perf:216]。

### 3.7 その他の報告値

- 全チップ SRAM 総バイト/総アクセス、及びそれに基づく総時間(**参考情報、レイテンシには不使用**)
  [perf:773-832, 1121-1185]。`total_sram_read_duration_ns = total_bytes_read / SRAM_BYTES_READ_PER_CYCLE * CLOCK_PERIOD_NS` [perf:819-822] 等。
- 平均 SRAM/SIMD 時間 = 総時間 / `num_ace_tiles`(=`len(ace_names)`)[perf:952, 974-994]。
- タイル別実測時間はデバッグ出力のみ [perf:1189-1278]。

### 3.8 アナログ+デジタル合算 [perf:1292-1306]

デジタル NPU JSON 提供時のみ。デジタルは直列加算(パイプライン重なりなしの保守的仮定):
```
frame_latency_ns += 1e9 / npu_fps                                    [perf:1303]
Combined fps = 1e9 / frame_latency_ns                                [perf:1305]
デジタル MAC = macs_bn * 1e9                                         [perf:1295]
```
PCIe 転送オーバーヘッドは含まないと明記 [perf:1308-1316]。

---

## 4. 電力推定(式)

`power_estimator.py`。基本原理: `電力 = 1 推論あたりエネルギー × 推論レート(inf_rate)` [pow:564]。

### 4.1 OpEnergy の 6 成分 [pow:93-138]

`OpEnergy` dataclass は 1 演算のエネルギーを 6 成分に分解:

| 成分 | 意味 | 主たる算出元 |
|---|---|---|
| `ace_active` | ACE アクティブ電力×時間 | `ace_op_energy_time`(物理電流モデル) |
| `ace_sleep` | ACE 非アクティブ(snooze/sleep)分 | 同上の inactive_energy |
| `sram` | 実データバッファアクセス | `calc_accessor_energy` |
| `accessor` | ディスクリプタ処理+データ処理 | `calc_accessor_energy` |
| `control` | トークン更新+オペレーションカウンタ | `calc_operation_energy` / `calc_accessor_energy` |
| `noc` | インターコネクト(演算内) | **常に 0**(`# TODO - ADD NOC ENERGY` [pow:343 等]) |

派生プロパティ [pow:104-117]:
```
total   = ace_active + ace_sleep + sram + accessor + control + noc      [pow:107]
digital = sram + accessor + control + noc                               [pow:112]
ace     = ace_active + ace_sleep                                        [pow:117]
```

### 4.2 制御エネルギー `calc_operation_energy` [pow:170-178]

```
n_iterations = get_num_op_control_iterations(buffer.iteration_spec)
             = ceil( get_num_kernel_iterations(iter_spec) / 4.0 )       [pow:159-162]
  ※ コメント「コンパイラに未実装」[pow:161]

get_num_kernel_iterations = prod(num_iterations.dims) * prod(wq_iterations.dims) * stay_count   [pow:152-156]

control += n_iterations * TOKEN_UPDATE_ENERGY * average_token_list_length          [pow:174]
control += n_iterations * OPERATION_COUNTER_BYTES(=96) * SRAM_BYTE_READ_ENERGY
         + n_iterations * OPERATION_COUNTER_WRITEABLE_BYTES(=24) * SRAM_BYTE_WRITE_ENERGY   [pow:175-176]
```

### 4.3 アクセサエネルギー `calc_accessor_energy` [pow:181-191]

```
n_bytes      = get_kernel_bytes(iter_spec) = prod(filter.dims)          [pow:165-167,184]
n_iterations = get_num_kernel_iterations(iter_spec)                     [pow:185]

control  += n_iterations * TOKEN_UPDATE_ENERGY * average_token_list_length                       [pow:186]
accessor += n_iterations * ACCESSOR_DESCRIPTOR_BYTES(=136) * (SRAM_BYTE_READ_ENERGY + SRAM_BYTE_WRITE_ENERGY)   [pow:187]
accessor += n_iterations * n_bytes * ACCESSOR_PROCESSING_ENERGY                                   [pow:188]
sram     += n_iterations * n_bytes * (SRAM_BYTE_WRITE_ENERGY if write else SRAM_BYTE_READ_ENERGY) [pow:189]
```

`TOKEN_UPDATE_ENERGY` は EnergyConstants 生成時に導出 [pow:81-86]:
```
TOKEN_UPDATE_ENERGY = 3 * 8 * (SRAM_BYTE_READ_ENERGY + SRAM_BYTE_WRITE_ENERGY)
   (8 バイト/トークン、3 アクセス=read+modify+write)
```

### 4.4 ACE 物理電流モデル `ace_op_energy_time` [pow:198-252]

これが電力推定の核。50°C typical コーナーでの電流(A)を仮定 [pow:205-212]:

| 記号 | 値 | 意味 |
|---|---|---|
| `i_adc_global` | 9.9e-6 A | GDAC・バイアス |
| `i_common_mode` | 2.2 × 2e-6 A | ADC 1 個あたりコモンモード(pFSR=5 で 2.2 倍) |
| `i_adc_core` | 83e-6 + 2×`i_common_mode` A | ADC 1 個あたりコア |
| `i_aidac_global_1p0` | 2.65e-4 A | AIDAC グローバル(1.0V) |
| `i_aidac_global_1p8` | 1.32e-3 A | AIDAC グローバル(1.8V) |
| `i_aidac_core_1p8` | 11.6e-6 A | AIDAC 1 個あたりコア(1.8V) |
| `i_aidac_core_1p0` | 4.6e-6 A | AIDAC 1 個あたりコア(1.0V) |

非アクティブ電流(sleep 引数で分岐)[pow:216-223]:
```
if sleep:  # ディープスリープ
    i_adc_core_inactive = 0
    i_aidac_core_inactive_1p0 = 0
    i_aidac_core_inactive_1p8 = 0
else:      # snooze モード
    i_adc_core_inactive = 1.6e-5 A
    i_aidac_core_inactive_1p0 = 2.4e-6 A
    i_aidac_core_inactive_1p8 = 5.4e-6 A
```

パワーゲーティング [pow:225-231]:
```
input_powergate_groupsize  = 1  (AIDAC 個別オフ可)
output_powergate_groupsize = 1  (ADC 個別オフ可)
n_inputs_active  = ceil(n_inputs / 1) * 1 = n_inputs
n_outputs_active = n_outputs
n_inputs_inactive  = MAX_ACE_INPUTS(=1280)  - n_inputs_active
n_outputs_inactive = MAX_ACE_OUTPUTS(=272)  - n_outputs_active
```

電力(W)= 電圧 × 電流の総和 [pow:232-246]。電源電圧: ADC=1.0V, AIDAC=1.0V と 1.8V の 2 系統。
```
active_power  = 1.0 * i_adc_global
              + 1.0 * i_adc_core * n_outputs_active
              + 1.0 * i_aidac_global_1p0
              + 1.8 * i_aidac_global_1p8
              + 1.0 * i_aidac_core_1p0 * n_inputs_active
              + 1.8 * i_aidac_core_1p8 * n_inputs_active                    [pow:238-243]

inactive_power = 1.0 * i_adc_core_inactive   * n_outputs_inactive
               + 1.0 * i_aidac_core_inactive_1p0 * n_inputs_inactive
               + 1.8 * i_aidac_core_inactive_1p8 * n_inputs_inactive       [pow:244-246]
```

時間とエネルギー [pow:247-252]:
```
t_adc = 1e-9 * [0,160,160,160,160,160,160,160,160][n_output_bits]   → n_output_bits>=1 で常に 160ns
energy          = active_power   * t_adc
inactive_energy = inactive_power * t_adc
return (energy, inactive_energy, t_adc)
```
コメント [pow:247] 「ADC 計算時間は常に 160ns、常に 8bit 出力」。

### 4.5 ACE 全体集計 `calc_energy_ace` [pow:319-357]

L0 の `mma_dot` launcher を走査 [pow:321-323]。各 op について:
```
n_iterations = get_num_kernel_iterations(op.input.iteration_spec)              [pow:329]
active_energy, inactive_energy, op_time = ace_op_energy_time(
    n_inputs  = 1 + prod(op.input.iteration_spec.wq_sub_filter.dims),
    n_outputs = prod(op.output.iteration_spec.wq_sub_filter.dims),
    n_input_bits  = 1 + op.input_end_bit - op.input_start_bit,
    n_output_bits = op.adc_cycles.dims[0],
    sleep=True )                                                               [pow:330-335]

profile.ace_active += active_energy   * n_iterations                           [pow:336]
profile.ace_sleep  += inactive_energy * n_iterations                           [pow:337]
total_time         += op_time         * n_iterations                           [pow:338]

# デジタルエネルギー(入力読み+出力書き+制御)
profile += calc_accessor_energy(op.base, op.input,  ..., write=False)          [pow:340]
profile += calc_accessor_energy(op.base, op.output, ..., write=True)           [pow:341]
profile += calc_operation_energy(op.base, op.input, ...)                       [pow:342]
```

余剰スリープ時間の埋め合わせ [pow:345-353]:
```
needed_sleep_time = num_aces / inf_rate - total_time
sleep_energy, sleep_sleep_energy, sleep_time = ace_op_energy_time(0,0,8,8, sleep=True)
profile.ace_sleep += (sleep_energy + sleep_sleep_energy) * needed_sleep_time / sleep_time   [pow:353]
```
= 1 推論周期(`num_aces/inf_rate` 秒)のうち演算していない時間を、n_inputs=n_outputs=0 の
アイドル電力で満たす。コメント [pow:345]「snooze と sleep の区別が甘い」と注記。

### 4.6 他演算 COPY/SIMD/PAD/INFEED/OUTFEED [pow:359-439]

いずれも ACE 電流モデルを使わず、`calc_accessor_energy` + `calc_operation_energy` のみ:

| 演算 | launcher | accessor 呼出 |
|---|---|---|
| COPY [pow:359-374] | `copy` | input(read)+output(write)+control |
| SIMD [pow:376-391] | `salu` | input(read)+output(write)+control |
| PAD [pow:393-407] | `pad` | output(write)+control |
| INFEED [pow:409-423] | `infeed` | output(write)+control |
| OUTFEED [pow:425-439] | `outfeed` | input(read)+control |

全てに `# TODO - ADD NOC ENERGY` があり NOC 成分は未算入。

### 4.7 インターコネクト(NOC)電力 `calc_energy_interconnect` [pow:441-482]

packet log(JSON)提供時のみ。各エントリで:
```
global_bytes = value["inter_tile_traffic"]["bytes_transferred"]
global_energy = global_bytes * INTERCON_GLOBAL_BYTE_XFER_ENERGY                [pow:453-455]
local_bytes  = value["total_bytes_transferred"] - global_bytes
local_energy = local_bytes * INTERCON_LOCAL_BYTE_XFER_ENERGY                   [pow:462-463]
total_energy = total_global_energy + total_local_energy                        [pow:470]
```
packet log 未提供なら 0.0 を返す [pow:443-445]。
注意: `hasattr(self,"packet_log")` は `packet_log` 属性が dataclass field で常に定義済みのため、
実際には None チェックにならない微妙な挙動(**推測**: 未提供時は for ループが None で例外になりうるが、
呼び出し側の警告ログ設計から packet log なし運用が主想定)。

### 4.8 PCIe 電力 `calc_pcie_power` [pow:484-493]

```
active_power = PCIE_POWER["P0"][num_pcie_lanes] * pcie_activity_factor
idle_power   = PCIE_POWER["P1"][num_pcie_lanes] * (1 - pcie_activity_factor)
power(mW)    = active_power + idle_power  →  /1000 で W                         [pow:489-493]
```
`PCIE_POWER` は 28nm IP databook 由来 [pow:30-37]。P0/P0S/P1/P2/POWER_DOWN × {1,2,4 レーン}。

### 4.9 Die-to-Die 電力 `calc_d2d_power` [pow:495-527]

```
デフォルト power = D2D_POWER["POWER_DOWN"](=1.7mW) * NUM_D2D_INSTANCES(=4)      [pow:505]

num_die_in_system > 1 のとき:
  num_active_phy = num_die_in_system - 1
  active_power = D2D_POWER["ACTIVE"][lanes] * D2D_ACTIVITY_FACTOR(0.3) * num_active_phy
  idle_power   = D2D_POWER["IDLE"][lanes]   * (1 - 0.3)               * num_active_phy
  active_phy_power   = (active_power + idle_power) * num_active_phy        [pow:510-512]
  inactive_phy_power = D2D_POWER["POWER_DOWN"] * (NUM_D2D_INSTANCES - num_active_phy)
  power = active_phy_power + inactive_phy_power                            [pow:517]
  → /1000 で W
```
`D2D_POWER` [pow:39-44] は 16/12/8 Tx/Rx レーンで ACTIVE=214.3mW を 1.0/0.75/0.5 倍、
IDLE=71.5mW を同様にスケール。
注意(**推測**): [pow:512] は `num_active_phy` を 2 回乗じており、次元的に過大の可能性(バグ疑い)。
ただしこの関数は total に算入されない(4.11 参照)ため公表値に影響しない。

### 4.10 デジタル NPU 電力 `calc_energy_digital_npu` [pow:529-542]

物理モデルは持たず、NPU プロファイル JSON の値を流用:
```
power = npu_log["profiling"]["power_mw_5nm_typ"]
fps   = npu_log["profiling"]["fps"]
return (power / fps) / 1000   # 1 フレームあたり W                              [pow:542]
```

### 4.11 total への算入/未算入 `calc_power` [pow:544-602]

```
operation_power    = total_energy.total * inf_rate                              [pow:564]
interconnect_power = calc_energy_interconnect() * inf_rate                       [pow:565]
total_power        = operation_power + interconnect_power                        [pow:571]
```

| 項目 | 算入? | 根拠 |
|---|---|---|
| Functional Unit(ACE+SRAM+accessor+control+全演算) | ✅ | `operation_power` [pow:564,588] |
| Interconnect(NOC、packet log 由来) | ✅ | `interconnect_power` [pow:565,589] |
| Leakage Power | ❌ | コメントアウト [pow:566-567,591] |
| Clock Tree Power | ❌ | コメントアウト [pow:568,592] |
| PCIe Power | ❌ | コメントアウト [pow:569,593](関数は存在 [pow:484-493]) |
| Die-to-Die Power | ❌ | コメントアウト [pow:570,594](関数は存在 [pow:495-527]) |
| OpEnergy.noc(演算内 NOC) | ❌ | 全 `calc_energy_*` で `# TODO`、常に 0 |
| Digital NPU | 別建て | `digital_npu_power` を別途出力 [pow:573-575,597] |

`# TODO - Add leakage, clock tree and chip I/O power` [pow:566] 及び末尾 NOTE [pow:601]
「leakage/clock tree/chip I/O は将来版で追加」。leakage/clock 係数(未使用)は
`LEAKAGE_POWER_POWER_FACTOR=0.5`, `CLOCK_TREE_POWER_FACTOR=0.5`(いずれも digital 電力に対する係数)
[pow:27-28,567-568]。

デジタル NPU 提供時のみ合算表示 [pow:599]:`total_power + digital_npu_power`。

### 4.12 TOPS/W 算出 [pow:547-555]

```
ace_op_energies[i] = ace_op_energy_time(n_input_bits=i, n_output_bits=i,
                        n_inputs=MAX_ACE_INPUTS(1280), n_outputs=MAX_ACE_OUTPUTS(272),
                        sleep=False)[0:1] の合計  (=active energy のみ)
                        for i in reversed(range(2, 9))                          [pow:547-553]

TOPS/W = 2 * MAX_ACE_INPUTS * MAX_ACE_OUTPUTS / energy / 1e12                    [pow:554]
       = 2 * 1280 * 272 / energy / 1e12   (=696,320 演算/E, 2 は積+和で 2 OP)
```
デバッグ出力のみ。フル ACE(1280 入力×272 出力)を各 ADC ビット精度で動かした場合の効率。

---

## 5. 面積推定

`perf_analysis.py` のみが算出。ACE アレイのタイルのみ対象(デジタル・I/O は含まない)。

定数 [perf:217-221]:
```
ACE_TILE_X_DIM_MM    = 8.24 mm   (shrink 前 X)
ACE_TILE_Y_DIM_MM    = 6.32 mm   (shrink 前 Y)
shrink 係数 0.9      → post-shrink 寸法
ACE_TILE_X_DIM_PS_MM = 8.24 * 0.9 = 7.416 mm
ACE_TILE_Y_DIM_PS_MM = 6.32 * 0.9 = 5.688 mm
ACE_TILE_AREA_PS_MM2 = 7.416 * 5.688 = 42.18 mm^2  (1 タイル)
```
`0.9` は「プロセスシュリンク係数」(**推測**: 28nm から縮小した実装ノードへの面積縮小率、コメントは
"post-shrink" のみ)。

最終面積 [perf:1283-1286]:
```
Estimated Die Area = ACE_TILE_AREA_PS_MM2 * PER_CHIP_ACE_TILE_COUNT
                   = 42.18 * (num_aces/4)
                   = 42.18 * 6 = 253.1 mm^2   (デフォルト 24 ACE = 6 タイル)
```
注: 正確には 8.24×0.9×6.32×0.9 = 42.182 mm²（本文中の ≈42.16 mm² は概算）。

---

## 6. ハードウェア定数一覧表

### 6.1 perf_analysis.py

| 定数 | 値 | 単位 | 意味 | 導出/根拠(行) |
|---|---|---|---|---|
| `CLOCK_PERIOD_NS` | 1 | ns | デジタルクロック周期(1 GHz) | [perf:211] |
| `ACE_DURATION_NS` | 160 | ns | ACE 計算 1 回の代表所要時間 | [perf:215] |
| `ACE_DURATION_SEC` | 160e-9 | s | 同上(秒) | =160/1e9 [perf:216] |
| `ACE_TILE_X_DIM_MM` | 8.24 | mm | タイル X(shrink 前) | [perf:217] |
| `ACE_TILE_Y_DIM_MM` | 6.32 | mm | タイル Y(shrink 前) | [perf:218] |
| shrink 係数 | 0.9 | – | プロセスシュリンク | [perf:219-220] |
| `ACE_TILE_AREA_PS_MM2` | 42.18 | mm² | 1 タイル面積(shrink 後) | X_ps×Y_ps [perf:221] |
| `CYCLES_IN_A_TIMESTEP` | 160 | cycle | 1 timestep のサイクル数 | =160/1 [perf:223] |
| `SIMD_VECTOR_WIDTH` | 8 | lane | SIMD ベクタレーン数 | [perf:227] |
| `IDEAL_SRAM_BYTES_READ_PER_CYCLE` | 128 | B/cycle | 理想 read 帯域 | [perf:233] |
| `IDEAL_SRAM_BYTES_WRITTEN_PER_CYCLE` | 128 | B/cycle | 理想 write 帯域 | [perf:234] |
| `SRAM_PERFORMANCE_SCALING_FACTOR` | 0.5 | – | バイト法の帯域現実化係数 | [perf:235] |
| `SRAM_BYTES_READ_PER_CYCLE` | 64 | B/cycle | =128×0.5 | [perf:236] |
| `SRAM_BYTES_WRITTEN_PER_CYCLE` | 64 | B/cycle | =128×0.5 | [perf:237] |
| `SRAM_BYTES_READ_PER_ACE_OPERATION` | 10240 | B | =64×160/1 | [perf:238] |
| `SRAM_BYTES_WRITTEN_PER_ACE_OPERATION` | 10240 | B | =64×160/1 | [perf:239] |
| `SRAM_ACCESS_PORTS` | 8 | port | SRAM ポート数(コメント「実際は 9、GNOC/TOM 分未計上」) | [perf:243] |
| `SRAM_ACCESSES_PERFORMANCE_SCALING_FACTOR` | 0.6 | – | バンク競合考慮(コメント「暫定推測」) | [perf:244] |
| `TOTAL_SRAM_ACCESSES_PER_CYCLE` | 4.8 | acc/cycle | =8×0.6 | [perf:245] |
| `TOTAL_SRAM_ACCESSES_PER_TIMESTEP` | 768 | acc | =4.8×160 | [perf:246] |
| `USING_NUM_ACCESSES` | True | – | 新形式トレースフラグ | [perf:250] |
| `PER_ACE_TILE_COUNT` | 4 | ACE/tile | 1 タイル ACE 数 | [perf:287] |
| `num_aces`(既定) | 24 | ACE | チップ ACE 数(6 タイル) | [perf:113-117,286] |

帯域係数の根拠(コメント):`0.5`=理想帯域に対する性能スケーリング(明示的根拠コメントなし、
保守的係数)[perf:235]。`0.6`=SRAM バンク競合を見込んだ暫定推測("a guess for now")[perf:244]。

### 6.2 power_estimator.py

| 定数 | 値 | 単位 | 意味 | 行 |
|---|---|---|---|---|
| `MAX_ADC_BITS` / `MIN_ADC_BITS` | 8 / 2 | bit | ADC 出力ビット範囲 | [pow:17-18] |
| `MAX_AIDAC_BITS` / `MIN_AIDAC_BITS` | 8 / 2 | bit | AIDAC 入力ビット範囲 | [pow:19-20] |
| `MAX_ACE_INPUTS` | 1280 | – | ACE 最大入力(AIDAC 数) | [pow:21] |
| `MAX_ACE_OUTPUTS` | 272 | – | ACE 最大出力(ADC 数) | [pow:22] |
| `ACCESSOR_DESCRIPTOR_BYTES` | 136 | B | アクセサディスクリプタ | [pow:23] |
| `OPERATION_COUNTER_BYTES` | 96 | B | オペレーションカウンタ(read) | [pow:24] |
| `OPERATION_COUNTER_WRITEABLE_BYTES` | 24 | B | 同(write) | [pow:25] |
| `DEFAULT_AVERAGE_TOKEN_LIST_LENGTH` | 5.0 | – | 平均トークンリスト長 | [pow:26] |
| `CLOCK_TREE_POWER_FACTOR` | 0.5 | – | クロックツリー係数(**未使用**) | [pow:27] |
| `LEAKAGE_POWER_POWER_FACTOR` | 0.5 | – | リーク係数(**未使用**) | [pow:28] |
| `NUM_D2D_INSTANCES` | 4 | – | D2D PHY 数 | [pow:45] |
| `D2D_ACTIVITY_FACTOR` | 0.3 | – | D2D 稼働率 | [pow:46] |
| ACE 電流群 | 表 4.4 参照 | A | 物理電流モデル | [pow:205-223] |
| 電源電圧 | 1.0 / 1.8 | V | ADC=1.0, AIDAC=1.0&1.8 | [pow:233-235] |
| `t_adc` | 160e-9 | s | ADC 計算時間(固定) | [pow:248] |

### 6.3 ENERGY_TABLE 全値 [pow:53-75]（単位: J/byte 又は J/op）

| キー | 28nm | 12nm | 5nm |
|---|---|---|---|
| `SRAM_BYTE_READ_ENERGY` | 1.0e-12 | 0.6e-12 | 0.5e-12 |
| `SRAM_BYTE_WRITE_ENERGY` | 1.1e-12 | 0.7e-12 | 0.6e-12 |
| `INTERCON_GLOBAL_BYTE_XFER_ENERGY` | 1.6e-12 | 1.6e-12 | 1.2e-12 |
| `INTERCON_LOCAL_BYTE_XFER_ENERGY` | 1.6e-12 | 1.6e-12 | 1.2e-12 |
| `ACCESSOR_PROCESSING_ENERGY` | 1.0e-12 | 0.6e-12 | 0.6e-12 |
| `TOKEN_UPDATE_ENERGY`(導出) | 3×8×(R+W) = 5.04e-11 | 3.12e-11 | 2.64e-11 |

12nm の LOCAL コメント [pow:65]「0.25fJ/um × (0.9V)² × 1000um × 8bit/byte = 1.6pJ/byte per mm」、
GLOBAL コメント [pow:64]「FF エネルギー 2fF/toggle なので複数ホップでも無視できる」。
`TOKEN_UPDATE_ENERGY` は R+W の和に 24(=3×8)を掛けて生成 [pow:81-86]。

---

## 7. バイト法とアクセス法(まとめ)

- **バイト法(bytes)**: HDF5 の `size`(実バイト)を read/write × data/control で集計し、
  理想帯域 128B/cycle を `0.5` 倍(=64B/cycle)で割って時間化。read と write を独立に扱い各々の
  タイル別最大を律速に使う [perf:620-640, 844-859]。旧形式トレースでも動作。
- **アクセス法(accesses)**: `num_accesses`(アクセス回数)を集計し、`8 ポート × 0.6`(=4.8 acc/cycle)
  で割る。read/write を区別せずポート共有前提で合算 → RTL に近く「より正確」[perf:657-660, 661-664, 869-876]。
  新形式トレース(`num_accesses` 必須)で有効。
- 両者は独立して `maximum_bottleneck_ns` / `maximum_bottleneck_accesses_ns` を生成し、
  それぞれ別ログ(`sram_method` 属性)で fps・利用率まで二重出力される [perf:135-162, 918-946, 1097-1115]。
- ログハンドラは既定で `accesses` のみ通す(`SramCalculationMethodFilter("accesses")` [perf:159])。
  つまり通常運用ではアクセス法の結果が表示され、バイト法は DEBUG 用途に近い。

---

## 8. 未算入項目と限界

**性能(perf_analysis.py):**
- PCIe 転送オーバーヘッド未含(明記 [perf:1308-1316])。
- アナログ+デジタルは単純直列加算(重なりを考慮しない保守的近似)[perf:1303]。
- SRAM 律速のタイル別最大は「data+control の max を合算」しており、作者自身が
  「本来はタイルで合算後に max を取るべき」と注記(近似の既知の不正確さ)[perf:610-611]。
- SRAM ポート数はコメント上「実際は 9、GNOC/TOM 分未計上」[perf:243]。
- `0.6` のバンク競合係数は "a guess for now" [perf:244]。
- タイル別/全チップ SRAM 総時間は参考情報でレイテンシに不使用 [perf:778-779, 1187-1188]。

**電力(power_estimator.py):**
- Leakage / Clock Tree / PCIe / D2D いずれも total に**未算入**(全てコメントアウト [pow:566-570])。
  関数(`calc_pcie_power`, `calc_d2d_power`)と係数(0.5)は定義済みだが呼ばれない。
- NOC(演算内)エネルギーは全 `calc_energy_*` で `# TODO`、常に 0 [pow:343 等]。
  ただし packet log 経由の interconnect_power は算入 [pow:565]。
- `get_num_op_control_iterations` は「コンパイラに未実装」[pow:161]で ceil(iter/4) の暫定式。
- ACE 電流は 50°C typical 固定、キャリブレーション時間ゼロ・連続実行前提 [pow:194-196, 205]。
- snooze/sleep の区別が甘い(コメント [pow:345])。
- `calc_d2d_power` の `num_active_phy` 二重乗算はバグ疑い(**推測**、[pow:512])だが total 非算入のため無害。
- Combined 電力はデジタル NPU 提供時のみ、単純加算 [pow:599]。

---

## 9. 参照ファイル一覧

| ファイル | 役割 |
|---|---|
| `mythic_sdk/v26.05.0/_extracted_compiler/perf_analysis.py` | 性能・面積推定(HDF5 トレース解析、全 1329 行) |
| `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/m2000_power_estimator/power_estimator.py` | 電力推定(L0 protobuf 解析、全 649 行) |
| `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/irs/l0/ir_pb2.py` | L0 IR 定義(`Crate`,`BaseLauncher`,`MmaDot`,`ParameterInfo`,`iteration_spec`,`adc_cycles`,`input_start_bit`/`input_end_bit`) |
| `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/irs/l0/parameters_pb2.py` | パラメータ/`LauncherRole` 定義 |
| `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/irs/l0/shape_pb2.py` | `Shape`/`dims` 定義 |
| `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/irs/l0/vector_processing_pb2.py` | SIMD(salu)関連 |
| `mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg/target_spec/target_pb2.py`, `resources_pb2.py` | ターゲット/リソース定義(ir が import) |

入力データファイル(実行時)。**生成元でスクリプト入力とフロー外部入力を区別する**:
- `perf_trace_dump.h5`(HDF5 トレース、`--hdf5-path`) — `perf_analysis.py` の入力だが、**コンパイラ出力
  でもフロー外部入力でもない**。`--estimate-performance` 実行時に `funcsim` がコンパイル成果物を
  実行して `artifacts/ppa/` 配下に生成する**内部中間トレース**(§1・§2 の注意参照)。
- L0 protobuf(`--l0-pb-path`、`Crate` としてパース [pow:284-285]) — `power_estimator.py` の入力で、
  **コンパイラの出力(`final.l0.pb`)= フローへの外部入力**。`.h5` と異なり funcsim 実行前に確定済み。
- 任意 JSON: packet log / event log(`.h5` と同じく `funcsim`/PPA フローが生成)/ digital NPU log。

---

*本ドキュメントは 26.05 SDK 抽出コードの静的解析に基づく。数式・行番号は上記ソースから直接引用。
「推測」と明記した箇所以外はコード・コメントの記述に基づく事実である。*
