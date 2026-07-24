# `gemm.cu` 代码结构说明

本文对应 `examples/gemm/gemm.cu`，用于解释 Qwen2.5 GEMM 测试程序的
模块边界、数据流、模板选择和实际执行路径。修改数据类型、命令行参数、
候选模板、trans/batch 语义或调优配置格式时，应同步更新本文。

## 1. 程序目标

程序使用 CUTLASS TensorOp GEMM 模拟 Qwen2.5 推理中捕获到的 cuBLAS
调用形式，支持：

- Q/K/V/O projection、MLP、LM Head；
- Attention QK^T 和 PV；
- cuBLAS 视角的 NN、NT、TN、TT；
- 普通 GEMM 和 strided-batched GEMM；
- FP16 A/B/C/D、默认 FP32 accumulator；
- CUDA Event 或 Chrono 计时；
- 完整候选比较或 optimal-only；
- 可选 Reference 校验；
- cuBLASLt 派生候选和实测 optimal 配置。

当前默认精度路径为：

```text
A/B/C/D       = FP16（用于近似 BF16）
Accumulator   = FP32
Epilogue      = FP32 compute
D write-back  = FP16
```

## 2. 顶层执行路径

当前 `main()` 的实际调用链为：

```text
main
├── 检查编译 CUDA 版本
├── Options::parse
│   └── apply_operation_defaults
├── cudaSetDevice / cudaGetDeviceProperties
├── 根据 trans 选择 LayoutA/LayoutB
└── profile_cublas_layout
    ├── optimal-only
    │   ├── profile_optimal_only_candidate
    │   └── profile_unmapped_single_candidate
    └── 普通模式
        ├── Alignment=8 → profile_all_candidates
        └── 其他 → Alignment=1 single fallback
```

源码中还保留大 M、完整 Alignment=8/4/2/1 和 cuBLASLt 派生候选模块，
但它们目前没有全部接入上述普通主路径。阅读代码时必须区分“已定义”和
“实际会从 main 调用”。

## 3. 编译期开关

文件开头将构建宏转换为常量：

| 宏 | C++ 常量 | 作用 |
|---|---|---|
| `GEMM_CONCISE_LOG` | `kConciseLog` | 抑制非最佳候选的详细输出 |
| `GEMM_USE_CHRONO` | `kUseChronoTimer` | 使用 Chrono，否则使用 CUDA Event |
| `GEMM_OPTIMAL_ONLY` | `kOptimalOnly` | 只执行 optimal 或一个 fallback |
| `GEMM_NCU_EXACT_ONLY` | `kNcuExactOnly` | 只执行 NCU 精确映射，未匹配直接失败 |
| `GEMM_SKIP_VERIFICATION` | `kVerifyResults=false` | 跳过 Reference 和结果拷贝 |
| `GEMM_ACCUMULATOR_TYPE` | `ElementAccumulator` | 选择 FP16/FP32 accumulator |

这些都是编译期选项，切换后必须重新编译。

## 4. 类型与硬件配置

基础类型：

```cpp
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = GEMM_ACCUMULATOR_TYPE;
using ElementCompute = ElementAccumulator;
```

基础硬件配置：

```cpp
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag = cutlass::arch::Sm80;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
```

`ArchTag=Sm80` 是 CUTLASS kernel 的最低架构标签，实际 `nvcc -arch`
可以指定更新的兼容架构。

## 5. `GemmConfiguration`

`GemmConfiguration` 是所有 GEMM 模板的统一入口，封装：

- ElementA/B/C 和 accumulator；
- LayoutA/B/C；
- ThreadblockShape；
- WarpShape；
- InstructionShape；
- Epilogue；
- ThreadblockSwizzle；
- Stages；
- AlignmentA/B/C。

候选模板只负责给这些参数提供具体值。

## 6. 内置候选模板

### `LinearGemm`

```text
TB32x256x32_W32x64x32_S4
Identity
```

主要面向 Decode Linear 和 LM Head 等瘦矩阵。

### `AttentionQKGemm`

```text
TB32x128x32_W16x64x32_S2
Identity
```

主要面向短 K 的 QK^T。

### `AttentionPVStreamKGemm`

```text
TB32x128x32_W16x64x32_S4
Stream-K
```

主要面向 PV、长 K 或 CTA 数量不足的情况。

### FP32 accumulator 候选

`Fp32CompactGemm` 和 `Fp32LargeM64x64Gemm` 用于评估 FP32 accumulator
导致的寄存器压力变化。

### 大 M 候选

源码保留：

```text
128x128, 128x256, 256x128,
64x128, 128x64, 64x256
```

用于 Prefill 调优。当前普通主路径没有直接执行完整大 M 候选集合。

## 7. 日志类型名称

`ConfigName<T>` 生成面向用户的名称，如 `float`、`row-major`、
`Stream-K`；`ConfigToken<T>` 生成可写入 `.inc` 的 C++ 类型名称，如
`cutlass::layout::RowMajor`。

修改日志字段或生成配置格式时，需要同时检查 Python 解析脚本。

## 8. `Options` 与 Qwen 语义

`Options` 保存：

```text
m/n/k, alpha/beta, iterations,
split_k_slices, batch_count,
operation, operand_a, operand_b, trans
```

`apply_operation_defaults()` 提供标准映射：

| operation | cuBLAS A | cuBLAS B | trans |
|---|---|---|---|
| `q_proj` | W_Q | HiddenStates | TN |
| `k_proj` | W_K | HiddenStates | TN |
| `v_proj` | W_V | HiddenStates | TN |
| `o_proj` | W_O | AttentionOutput | TN |
| `up_proj` | W_up | MLPInput | TN |
| `gate_proj` | W_gate | MLPInput | TN |
| `down_proj` | W_down | SwiGLUOutput | TN |
| `lm_head` | W_lm_head | LastHiddenState | TN |
| `qk` | K | Q | TN |
| `pv` | V | P | NN |

显式 `--operand-a`、`--operand-b`、`--trans` 分别覆盖 operation 默认值。
operand 名称用于语义标注；真正改变内存访问的是 trans 对应的 layout。

## 9. trans 到 Layout 的映射

程序使用 cuBLAS 列主序调用视角：

| trans | LayoutA | LayoutB | LayoutC |
|---|---|---|---|
| NN | ColumnMajor | ColumnMajor | ColumnMajor |
| NT | ColumnMajor | RowMajor | ColumnMajor |
| TN | RowMajor | ColumnMajor | ColumnMajor |
| TT | RowMajor | RowMajor | ColumnMajor |

因此“模型公式中的 K 被转置”不等于 `transb=T`；必须同时看 A/B 分别
对应哪个 Qwen 张量。

## 10. `Tensors`

`Tensors<LayoutA,LayoutB,LayoutC>` 保存：

```text
A, B, C, D, reference
batch_stride_a/b/c
```

一个结构同时支持普通 GEMM 和 strided-batched GEMM。每个 batch 使用
连续、互不重叠的存储区域。

## 11. `initialize_tensors`

初始化流程：

1. 根据 M/N/K 和 layout 创建单个矩阵视图；
2. 计算单 batch 容量和 stride；
3. 按 `batch_count` 扩展底层分配；
4. 校验模式下随机初始化第一个 batch；
5. 将第一个 batch 复制到其余 batch；
6. 同步 A/B/C/D/reference 到设备。

关闭校验时直接在设备端清零，不生成 Host 随机数据。

## 12. `compute_reference`

Reference 使用与被测 kernel 相同的：

- M/N/K；
- LayoutA/B/C；
- accumulator 和 Epilogue compute；
- alpha/beta；
- batch stride。

函数逐 batch 调用 CUTLASS Device Reference GEMM，随后同步并用于比较。

## 13. `ArgumentFactory`

`ArgumentFactory` 将 `Options` 和 `Tensors` 转成 `Gemm::Arguments`。

模式选择：

```text
batch_count == 1 → GemmUniversalMode::kGemm
batch_count > 1  → GemmUniversalMode::kBatched
```

普通 Swizzle 与 Stream-K 使用两个特化版本，因为 Stream-K 构造参数包含
额外的 SM 数控制参数。

## 14. `run_tensorop_gemm`

执行顺序：

```text
清理并检查既有 CUDA error
→ Gemm::can_implement
→ 分配 workspace
→ gemm.initialize
→ warm-up 一次
→ timed iterations
→ 计算 avg_time 和 GFLOPS
→ D2H
→ 逐元素比较全部 batch
```

错误位置通过以下阶段标识：

```text
preexisting_cuda_error
can_implement
initialize
warmup_launch
timed_launch
timer
verification_copy
verification_compare
```

GFLOPS 计算：

```text
2 * M * N * K * batch_count / avg_time
```

## 15. Attention Batched GEMM

Attention 不再把 Head 数展平进 M/N：

```text
Decode QK^T: M=L,        N=1, K=head_dim
Decode PV:   M=head_dim, N=1, K=L

Prefill QK^T: M=S,        N=S, K=head_dim
Prefill PV:   M=head_dim, N=S, K=S

batch_count = batch_size * attention_heads
```

Batch size 只进入 `batch_count`，不会同时进入 N。

当前固定 batch stride 相当于 K/V 已按 Q Head 组织或完成 `repeat_kv`
物化，不支持未物化 GQA 的非线性 KV Head 指针复用。

## 16. 结果输出

### 人类可读输出

`print_configuration()` 和 `print_result()` 输出 operation、A/B、trans、
物理尺寸、batch count、layout、Tile、Stages、Status、时间和 GFLOPS。

### 机器记录

`print_cutlass_record()` 输出：

```text
CUTLASS_CANDIDATE ...
CUTLASS_BEST ...
```

记录包含 M/N/K/trans/batch_count、accumulator、layout、alignment、
Tile、Stages、split-K、时间和 GFLOPS。调优脚本依赖这些字段。

## 17. 基础候选比较

`profile_all_candidates()`：

1. 初始化一次张量；
2. 计算一次 Reference；
3. 运行 Linear、QK、PV 和 FP32 Compact 候选；
4. 排除失败或校验失败项；
5. 选择平均时间最低者；
6. 输出 `CUTLASS_BEST`。

## 18. Optimal-only

`profile_optimal_only_candidate()` 读取
`examples/gemm/optimal_configurations.inc`。

新的 trans-aware 记录按以下维度匹配：

```text
accumulator + M/N/K + trans + batch_count
```

命中后只执行对应模板；未命中时执行一个确定性 pseudo-random fallback。
不同 accumulator、trans 或 batch_count 不应复用同一最佳结果。

## 19. cuBLASLt 派生候选

`profile_cublaslt_template()` 和
`profile_cublaslt_generated_candidate()` 读取
`cublaslt_generated_candidates.inc`。

该模块保存 cuBLASLt 元数据转换得到的 CUTLASS 候选。目前
`profile_cublas_layout()` 没有调用生成候选分派，因此普通运行不会将它
加入比较。若重新接入，必须保证生成文件按 trans 和 batch_count 匹配。

## 19.1 NCU-exact 分派

`--ncu-exact` 定义 `GEMM_NCU_EXACT_ONLY=1`，并要求存在
`examples/gemm/ncu_exact_configurations.inc`。它按 accumulator、M/N/K、
trans 和 batch_count 完全匹配。

输入的 `ncu_tb_m/ncu_tb_n/ncu_tb_k/ncu_stages` 原样进入
`GemmShape<TB_M,TB_N,TB_K>` 和 `kStages`，不经过 cuBLASLt-derived 的
`TB_K=32` 或 Stages 2～4 映射。未匹配时直接失败，不进入 fallback。
WarpShape、Layout、Alignment、Swizzle、Split-K 是显式提供的 CUTLASS
补充字段，不能标为 NCU 直接值。

普通非 `--ncu-exact` 构建也会在文件存在时调用
`profile_ncu_generated_candidate()`。匹配配置输出
`CUTLASS_CANDIDATE source=ncu-exact`，与基础候选和 cuBLASLt-derived
候选一起进入多轮日志。最终 optimal 生成器读取所有候选记录，按同一 shape
和配置的中位时间选择 winner；因此 NCU-exact 是候选来源，不享有优先权。

目标机自动采集由 `capture_ncu_candidates.sh` 和
`parse_ncu_candidates.py` 完成。前者逐 manifest 用例保存 `.ncu-rep` 与 raw
CSV；后者只解析已验证 kernel 命名规则，使用 NCU block threads 校验
warps/CTA 并枚举 WarpShape。未识别名称写入 metadata 的 skipped 列表，
不得进入候选。

采集命令使用 Application Replay 和 `--profile-from-start off`。
`cublaslt_profiler --profile-winner-only` 先在采集区间外完成 heuristic
筛选，仅在 `cudaProfilerStart/Stop` 区间启动一次 winner，防止 NCU 对全部
筛选 kernel 执行 replay。

## 20. Alignment 分派

连续维由 layout 决定：

```text
ColumnMajor A → M
RowMajor A    → K
ColumnMajor B → K
RowMajor B    → N
ColumnMajor C → M
```

当前主路径：

```text
A/B/C连续维均满足8 → Alignment=8，比较基础候选
否则               → Alignment=1、Stages=2、单个fallback
```

源码保留完整 Alignment=8/4/2/1 和大 M 分派函数，但目前未从普通主路径
完整调用。

## 21. 配套文件

| 文件 | 与 `gemm.cu` 的关系 |
|---|---|
| `build_gemm.sh` | 设置编译宏、accumulator、timer 和架构 |
| `run_gemm.sh` | 生成 Qwen operation、M/N/K/trans/batchCount |
| `optimal_configurations.inc` | optimal-only 的正式模板映射 |
| `cublaslt_generated_candidates.inc` | 可选 cuBLASLt 派生候选 |
| `ncu_exact_configurations.inc` | 可选 NCU CTA/Stages 原样映射 |
| `tools/qwen_gemm/generate_qwen_shapes.py` | 生成全模型 shape 清单 |
| `tools/qwen_gemm/generate_cutlass_candidates.py` | 生成 cuBLASLt 派生候选 |
| `tools/qwen_gemm/generate_optimal_configurations.py` | 生成 optimal 映射 |
| `tools/qwen_gemm/generate_ncu_exact_configurations.py` | 由 CSV 生成 NCU-exact 映射 |
| `tools/qwen_gemm/capture_ncu_candidates.sh` | 目标机 NCU 抓取 |
| `tools/qwen_gemm/parse_ncu_candidates.py` | NCU 名称解析与 CUTLASS 补全候选生成 |
| `tools/qwen_gemm/estimate_qwen_gemm.py` | 使用运行结果估算 forward |

## 22. 修改同步检查表

修改 `gemm.cu` 时按下表检查配套文件：

| 修改内容 | 同步检查 |
|---|---|
| 新增 operation | `run_gemm.sh`、shape 生成器、本文映射表 |
| 修改 M/N/K 或 batch 语义 | run/shape/估算脚本、日志解析、README |
| 修改日志字段 | 两个生成脚本和结果收集脚本 |
| 修改 accumulator | build、cuBLASLt profiler、optimal 键 |
| 修改候选模板 | 配置名称、调优生成器、optimal 文件 |
| 修改 trans/layout | cuBLASLt profiler、alignment、Reference |
| 修改输出目录 | README、调优脚本和 `.gitignore` |

## 23. 后续拆分建议

若继续扩展，建议将单文件拆为：

```text
qwen_gemm_options.h       # 参数和 operation/trans 语义
qwen_gemm_tensors.h       # 张量、初始化、Reference
qwen_gemm_candidates.h    # 模板候选和 alignment
qwen_gemm_runner.h        # 参数构造、计时、错误处理
qwen_gemm_reporting.h     # 人类日志和机器记录
gemm.cu                   # main 与顶层分派
```

拆分前应先补充最小编译/运行回归：

- NN、NT、TN、TT 各一例；
- Linear batch_count=1；
- QK/PV batch_count>1；
- FP16/FP32 accumulator；
- verification 开启/关闭；
- optimal-only 命中/未命中。
