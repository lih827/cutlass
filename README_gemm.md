# Qwen2.5 Decode / Prefill GEMM 测试

本测试使用 CUTLASS TensorOp GEMM，根据不同 Qwen2.5 模型参数生成 Decode 或 Prefill 阶段的 `M/N/K`，按 `M/N/K` 去重后运行，并在小 M 与大 M 候选配置中分别选择最佳结果。

## 目录结构与文件用途

Qwen GEMM 扩展不是 CUTLASS 上游原生产物。为避免污染 CUTLASS 根目录，
根目录只保留 README 和三个常用入口；辅助工具、生成结果和仓外交付物分类
存放：

```text
cutlass/
├── README_gemm.md
├── README_gemm_runtime.md
├── build_gemm.sh
├── run_gemm.sh
├── tune_optimal_cutlass.sh
├── examples/gemm/
│   ├── gemm.cu
│   ├── cublaslt_profiler.cu
│   ├── optimal_configurations.inc
│   ├── ncu_exact_configurations.csv.example
│   ├── ncu_exact_configurations.inc       # 按需生成
│   └── cublaslt_generated_candidates.inc
├── tools/qwen_gemm/
│   ├── generate_qwen_shapes.py
│   ├── generate_cutlass_candidates.py
│   ├── generate_optimal_configurations.py
│   ├── generate_ncu_exact_configurations.py
│   ├── capture_ncu_candidates.sh
│   ├── parse_ncu_candidates.py
│   ├── collect_gemm_results.py
│   ├── estimate_qwen_gemm.py
│   └── tune_cutlass_from_cublaslt.sh
├── docs/qwen_gemm/
│   ├── GEMM_CODE_GUIDE.md
│   ├── qwen2.5推理中gemm算子使用.docx
│   └── qwen2_5_0_5b_prefill_decode_flow.svg
└── outputs/qwen_gemm/                 # 生成目录，不提交 Git
    ├── binaries/
    ├── manifests/
    ├── logs/
    ├── cublaslt_tuning/
    └── optimal_tuning/

../../qwen_gemm_artifacts/             # CUTLASS 仓库外
├── qwen_decode_gemm_package.tar.gz
├── gemm_performance_comparison.xlsx
├── up_gate_test.log
└── up_gate_test.xlsx
```

| 文件 | 用途 |
|---|---|
| `README_gemm.md` | 完整说明，包括 cuBLASLt 采样、候选生成和 optimal 调优 |
| `README_gemm_runtime.md` | 仅构建、运行、结果收集和推理估算的精简说明 |
| `build_gemm.sh` | 编译 CUTLASS GEMM 与可选 cuBLASLt profiler |
| `run_gemm.sh` | 按 Qwen2.5 参数生成并运行 Decode/Prefill 用例 |
| `tune_optimal_cutlass.sh` | 全模型、指定 accumulator 的最终模板调优入口 |
| `gemm.cu` | 支持 trans、Qwen 操作数语义和 strided-batched Attention 的测试程序 |
| `cublaslt_profiler.cu` | 获取相同 M/N/K/trans/batchCount 的 cuBLASLt 启发式结果 |
| `optimal_configurations.inc` | optimal-only 编译使用的实测最佳 CUTLASS 模板 |
| `ncu_exact_configurations.csv.example` | NCU-exact 输入表格式示例 |
| `ncu_exact_configurations.inc` | 可选的 NCU CTA/Stages 原样 CUTLASS 映射 |
| `cublaslt_generated_candidates.inc` | 可选的 cuBLASLt 派生 CUTLASS 候选 |
| `generate_qwen_shapes.py` | 生成所有 Qwen2.5 参数规模的去重用例清单 |
| `generate_cutlass_candidates.py` | 将 cuBLASLt 元数据转换为合法 CUTLASS 候选 |
| `generate_optimal_configurations.py` | 按实测中位时间生成 optimal 配置 |
| `generate_ncu_exact_configurations.py` | 原样保留 NCU CTA M/N/K、Stages 并生成严格分派 |
| `capture_ncu_candidates.sh` | 在目标 GPU 上逐用例运行 NCU 并保留原始报告 |
| `parse_ncu_candidates.py` | 解析已验证 kernel 命名规则并枚举合法 WarpShape |
| `collect_gemm_results.py` | 解析日志并回填仓外 XLSX |
| `estimate_qwen_gemm.py` | 从实测最佳时间估算 GEMM-only forward 延迟 |
| `tune_cutlass_from_cublaslt.sh` | 单模型 cuBLASLt 候选采样辅助流程 |
| `GEMM_CODE_GUIDE.md` | `gemm.cu` 的模块、调用链、数据流和维护同步说明 |
| `qwen2.5推理中gemm算子使用.docx` | 0.5B Prefill/Decode 实测 cuBLAS 接口、M/N/K、trans 和 batchCount 参考 |
| `qwen2_5_0_5b_prefill_decode_flow.svg` | 0.5B Prefill/Decode 算子流程及 GEMM 位置示意图 |

`docs/qwen_gemm` 保存本测试依赖的 Qwen 实测参考资料副本；仓外原件不移动。
`outputs/qwen_gemm` 中的日志、清单、二进制和调优中间结果均可重新生成，
不属于 CUTLASS 原生源码。`../../qwen_gemm_artifacts` 中的压缩包、XLSX
和历史测试结果是仓外交付物，不得提交到 CUTLASS 仓库。

`build_gemm.sh` 和 `run_gemm.sh` 都必须从 CUTLASS 根目录执行。

脚本既支持直接执行，也支持显式通过 Bash 调用；使用 `bash` 时不要求脚本文件具有执行权限：

```bash
bash build_gemm.sh --arch sm_89
bash run_gemm.sh --model 7b
bash tools/qwen_gemm/tune_cutlass_from_cublaslt.sh --model 7b --arch sm_89
```

## 环境要求

- Linux
- Bash 4.0 或更高版本
- Python 3 和 `openpyxl`
- CUDA Toolkit
- CUTLASS 源码
- SM80 或更高架构的 NVIDIA GPU

当前 kernel 使用：

```text
ArchTag         = Sm80
OperatorClass   = OpClassTensorOp
InstructionShape = 16x8x16
```

安装 XLSX 回填依赖：

```bash
python3 -m pip install openpyxl
```

## 添加执行权限

进入 CUTLASS 根目录：

```bash
cd /path/to/cutlass
chmod +x build_gemm.sh run_gemm.sh
```

## 编译

默认针对 `sm_89` 编译：

```bash
./build_gemm.sh
```

指定其他 GPU 架构：

```bash
./build_gemm.sh --arch sm_80
./build_gemm.sh --arch sm_86
./build_gemm.sh --arch sm_89
```

默认版本打印所有候选配置及其结果。若只需要最终胜出的完整配置、Status、avg_time 和 GFLOPS，可使用编译期开关：

```bash
./build_gemm.sh --arch sm_89 --concise-log
./run_gemm.sh --model 7b
```

精简日志仍会在内部执行所有 CUTLASS 候选以及命中的 cuBLASLt 派生候选，只抑制非最佳项的输出，不会改变选优过程。要恢复完整日志，重新执行不带 `--concise-log` 的 build。

默认使用 CUDA Event 计时。也可以在编译时切换为 `std::chrono::steady_clock`：

```bash
./build_gemm.sh --arch sm_89 --timer cuda-event
./build_gemm.sh --arch sm_89 --timer chrono
```

选择累加类型（A/B/C/D 始终为 FP16）：

```bash
./build_gemm.sh --arch sm_89 --accumulator fp16
./build_gemm.sh --arch sm_89 --accumulator fp32
```

默认使用 FP32 accumulator：A/B/C/D 保持 FP16（用于近似 BF16），
GEMM 主循环使用 FP32 累加，Epilogue 使用 FP32 compute，结果 D 转换并
写回 FP16。FP16 accumulator 仅作为显式选择的对照模式。
`--accumulator` 是编译选项，修改后必须重新编译。源码统一通过
`GEMM_ACCUMULATOR_TYPE` 选择类型，后续可以扩展其他 CUTLASS 类型，但还需
同步确认输入类型、InstructionShape、OperatorClass 与目标架构支持该组合。

`chrono` 模式会在计时区间前后执行 `cudaDeviceSynchronize()`，因此测量的是整段迭代的主机墙钟时间；适合与主机侧计时程序保持一致。短 kernel 的精确性能比较优先使用默认的 `cuda-event`。

已知部分 shape 的最优模板时，可编译单候选模式以减少执行时间：

```bash
./build_gemm.sh --arch sm_89 --optimal-only
```

该模式只读取 `examples/gemm/optimal_configurations.inc`。每条映射同时包含
accumulator 类型；只有 accumulator 与精确 `M/N/K` 均命中时才执行对应模板，
不会跨类型复用最优结果。仓库当前自带的是先前环境实测得到的 FP16 映射；
FP32 在完成对应调优前使用确定性 fallback。运行阶段不会重新调优或比较候选。

后续新增最优配置时，在 `.inc` 中增加一条 `GEMM_OPTIMAL_ENTRY` 并重新编译即可，无需修改 `gemm.cu`。每条记录依次指定：accumulator C++ 类型、`M/N/K`、A/B/C Layout、A/B/C Alignment、ThreadblockShape、WarpShape、Swizzle 和 Stages。

### NCU-exact：原样使用 NCU CTA 与 Stages

NCU-exact 与 cuBLASLt-derived 是两条独立路径。前者不固定 `TB_K=32`，也
不把 Stages 限制为 2～4；输入的 `ncu_tb_m/n/k` 和 `ncu_stages` 会原样
成为 CUTLASS `ThreadblockShape` 和 `kStages`。

```bash
cp examples/gemm/ncu_exact_configurations.csv.example ncu_exact.csv
python3 tools/qwen_gemm/generate_ncu_exact_configurations.py \
  --input ncu_exact.csv \
  --output examples/gemm/ncu_exact_configurations.inc

bash build_gemm.sh --arch sm_90 --accumulator fp32 --ncu-exact
./examples/gemm/gemm \
  --m=128 --n=3584 --k=3584 --trans=NN --batch-count=1
```

匹配键是 `accumulator + M/N/K + trans + batch_count`。没有完全匹配时直接
失败，不执行 optimal 或随机 fallback。日志中的 `tb_m/tb_n/tb_k/stages`
来自实际实例化的 CUTLASS 类型，可与输入表交叉检查。

NCU 准确提供且保持不变的字段标记为：

```text
ncu_tb_m / ncu_tb_n / ncu_tb_k    direct:ncu-kernel-name
ncu_stages                         direct:ncu-kernel-name
```

仅凭 CTA 和 Stages 不能唯一确定 `WarpShape、Layout、Alignment、Swizzle、
Split-K`，这些字段必须在 CSV 中显式补齐，并标记为
`provided:cutlass-completion`。生成器不会修正 NCU 数值；若组合不满足
CUTLASS 或目标 GPU 约束，应让编译或运行明确失败。

NCU-exact 也可以参与完整 optimal 调优，而不独占执行：

```bash
bash tune_optimal_cutlass.sh \
  --models all \
  --arch sm_90 \
  --accumulator fp32 \
  --ncu-exact-csv ncu_exact.csv
```

普通候选构建发现 `ncu_exact_configurations.inc` 后，会对精确匹配的 NCU
模板计时并输出 `CUTLASS_CANDIDATE source=ncu-exact`。同一轮中的基础
CUTLASS、cuBLASLt-derived 和 NCU-exact 记录使用相同 iterations、计时器
与输入条件。`generate_optimal_configurations.py` 按多轮 `avg_time` 中位数
统一选择 winner，因此可运行且更快的 NCU-exact 模板会被写入最终
`optimal_configurations.inc`。不可编译的配置会在构建期明确失败；能够编译
但不能运行或校验失败的配置记录为无效，不会成为 winner。

推荐在目标机器上直接自动抓取，而不是手工填写 CSV：

```bash
bash tune_optimal_cutlass.sh \
  --models all \
  --arch sm_90 \
  --accumulator fp32 \
  --capture-ncu
```

如 `ncu` 不在 `PATH`：

```bash
bash tune_optimal_cutlass.sh \
  --models all --arch sm_90 --accumulator fp32 --capture-ncu \
  --ncu /opt/nvidia/nsight-compute/ncu
```

自动流程会在 `outputs/qwen_gemm/optimal_tuning/<accumulator>/` 保存：

```text
ncu_capture/reports/*.ncu-rep       NCU原始报告
ncu_capture/csv/*.csv               NCU raw page
ncu_capture/capture_index.csv       报告与M/N/K/trans/batchCount对应关系
ncu_exact_candidates.csv            可参与CUTLASS编译的补全候选
ncu_exact_metadata.json             字段来源及未解析kernel原因
```

解析器只接受已经验证的命名形式，例如：

```text
cutlass_...gemm_<CTA_M>x<CTA_N>_<CTA_K>x<Stages>_<nn|nt|tn|tt>_align<N>
```

完整 kernel 名称和 block threads 是 NCU 直接数据；CTA M/N/K、Stages 是从
已验证命名规则解析的数据。WarpShape 根据 `block_threads/32` 对合法 CUTLASS
形状进行枚举，Layout/Alignment 根据 trans 和连续维得到，Swizzle/Split-K
当前补全为 Identity/1。任何不匹配已验证命名规则的 kernel 都只写入 metadata
的 `skipped`，不会猜测 CTA 或 stages。

同一个 NCU CTA/Stages 可以对应多个 WarpShape 补全候选。普通调优会全部
实测；`--ncu-exact` 严格运行模式也会只在 NCU-exact 集合内部比较这些候选，
选择其中实测最快的一项，不会进入基础或 cuBLASLt-derived fallback。

正式性能测试已经确认模板正确时，可关闭正确性校验及其数据准备/拷贝：

```bash
./build_gemm.sh --arch sm_89 --optimal-only --skip-verification
```

该模式直接在 Device 端清零 A/B/C，跳过 Host 随机初始化和 H2D、reference 张量、reference GEMM、reference/D 输出的 D2H 以及 `TensorEquals`。结果显示 `verification: disabled` 和 `Status: Not verified`。内存分配、kernel 初始化、一次 warmup 和正式 iterations 仍会执行。默认不加该选项时继续进行完整校验。

指定 `nvcc` 路径：

```bash
./build_gemm.sh --nvcc /usr/local/cuda/bin/nvcc
```

编译产物为：

```text
examples/gemm/gemm
examples/gemm/cublaslt_profiler
```

## 使用 cuBLASLt 缩小 CUTLASS 搜索范围

在最终运行机器上执行：

```bash
chmod +x tools/qwen_gemm/tune_cutlass_from_cublaslt.sh
./tools/qwen_gemm/tune_cutlass_from_cublaslt.sh \
  --model 7b --arch sm_89 --accumulator fp16 --iterations 20
```

该脚本依次完成：

1. 构建 CUTLASS 测试程序和 cuBLASLt profiler。
2. 对去重后的全部 Decode、Prefill `M/N/K` 请求 cuBLASLt heuristic 候选并逐一计时。
3. 记录每个 shape 的最优 Algo ID、Tile、Stages、Split-K、Swizzle 和 workspace。
4. 运行 `generate_cutlass_candidates.py`，把 cuBLASLt Tile/Stages 转换成合法的 SM80 CUTLASS Threadblock/Warp 模板。
5. 对现有候选没有覆盖的合法 Tile 自动新增模板，并重新编译 `gemm`。
6. 后续运行命中同一 `M/N/K` 时，把生成模板与 CUTLASS 自身候选一起实测比较并输出真正的最佳项；没有映射的 shape 只比较 CUTLASS 自身候选。

`cublaslt_generated_candidates.inc` 是可选文件。文件尚未生成或被删除时，`gemm.cu` 仍可正常编译，`run_gemm.sh` 会使用原有候选集合；执行调优脚本后才启用精确 shape 映射。

调优脚本会在第一次 build 前自动隔离旧 `.inc`，并把它备份到当前累加器独立目录。新映射先写入临时文件；如果生成模板编译失败，失败源码会保存为 `.failed`，脚本删除活动映射并自动重新构建基础候选版本，因此旧文件或非法新配置不会破坏后续普通 build/run。

生成文件包括：

```text
examples/gemm/cublaslt_generated_candidates.inc
outputs/qwen_gemm/cublaslt_tuning/fp16/cublaslt_decode.log
outputs/qwen_gemm/cublaslt_tuning/fp16/cublaslt_prefill.log
outputs/qwen_gemm/cublaslt_tuning/fp16/cublaslt_cutlass_mapping.csv
```

cuBLASLt 的 Tile/Stages 不是 CUTLASS 模板的完整描述。生成器遵循以下约束：

- `.inc` 在每个 shape 前保留原始 cuBLASLt Algo ID、Tile ID/尺寸、Stages ID、Split-K、Reduction、Swizzle、Workspace，仅供追溯参考；其后保存实际参与编译的 CUTLASS 映射。
- Tile 在合法目录中存在时精确采用；现有候选未覆盖但可合法构造的 Tile 会新增 CUTLASS 模板；其余情况选择最近的合法 TensorOp Tile，并在 CSV 中标记 `tile_mapping`。
- cuBLASLt 单缓冲 Stages 会提升为 CUTLASS 合法的最小 Stages 2。
- 自动生成仅使用已经通过 SM80 编译检查的 WarpShape（每个维度至少 32、每个 CTA 不超过 8 个 warp）；非常规 cuBLASLt Tile 会映射到最近的安全 CUTLASS Tile。
- Threadblock K 固定为已验证的 32；cuBLASLt 的内部 stage-size 不直接当作 CUTLASS Threadblock K。Stages 限制在 2～4，以避免非法单缓冲或过大的共享内存配置。
- cuBLASLt 的 Split-K 数量会写入生成模板，并由 CUTLASS `GemmUniversal` 使用对应的 split-K slices。
- alignment=1 不做 padding，固定使用 Threadblock K=32、同步 `MmaPipelined` Stages 2。
- 生成结果与 GPU、CUDA/cuBLASLt 版本、workspace 和矩阵布局相关，换机器或升级 CUDA 后应重新执行调优脚本。

## 调优数据来源与可信度

调优数据必须按来源分栏保存，不允许用推导值覆盖直接观测值。当前使用以下三级标记：

```text
direct   API、计时器或分析器直接返回的值
parsed   从实际执行的 kernel 名按已知命名规则解析的值
derived  为构造合法 CUTLASS 候选而计算或选择的值
```

### cuBLASLt直接取得的字段

`cublaslt_profiler` 直接设置或通过 cuBLASLt API 查询以下字段：

```text
M/N/K
accumulator / compute type
algo_id
tile_id
stages_id
split_k
reduction
CTA swizzle
custom option
workspace
CUDA Event avg_time
GFLOPS
```

这些字段在报告中标记：

```text
cublaslt_config_origin      = direct:cublaslt-api
cublaslt_measurement_origin = direct:cuda-event
```

`tile_id` 和 `stages_id` 是准确的 cuBLASLt 内部 ID，但不是 CUTLASS 的完整
ThreadblockShape 和 Stages，不能把 ID 本身直接改名为 CUTLASS 配置。

### 当前由生成器推导的字段

`generate_cutlass_candidates.py` 为了生成可编译候选，会产生：

```text
cutlass_threadblock
cutlass_warp
cutlass_stages
cutlass_split_k
alignment
layout
```

其来源分别写入 CSV：

```text
cutlass_threadblock_origin
cutlass_warp_origin
cutlass_stages_origin
cutlass_split_k_origin
alignment_origin
layout_origin
```

当前规则为：

```text
Threadblock M/N  由 tile_id 查表或选择最近合法tile
Threadblock K    固定为已验证的32
WarpShape        从合法CUTLASS catalog选择，并非cuBLASLt真实WarpShape
Stages           由 stages_id 映射并限制为2～4
Split-K          复制cuBLASLt API直接返回值
Alignment        根据M/K及连续维度计算
Layout           根据本测试程序的存储策略选择
```

生成的 `.inc` 也分别使用注释：

```text
[direct:cublasLt-api]
[derived:cutlass-candidate]
```

因此 `cuBLASLt-derived TB..._W..._S...` 只表示候选来源，不表示完整还原了
cuBLASLt 的真实 kernel。

### 可选NCU联合采样字段

若额外使用 NCU 捕获最终 winner，应把结果放入单独字段：

```text
ncu_kernel_name             direct
ncu_grid_x/y/z              direct
ncu_block_x/y/z             direct
ncu_registers_per_thread    direct
ncu_shared_memory_bytes     direct
ncu_threadblock_m/n/k       parsed:kernel-name
ncu_stages                  parsed:kernel-name
ncu_kernel_layout           parsed:kernel-name
ncu_kernel_alignment        parsed:kernel-name
ncu_warps_per_cta           derived:block-threads/32
```

例如已知格式：

```text
...gemm_32x32_128x2_nn_align8
```

可解析为 `TB32x32x128_S2`、`nn`、`align8`，但必须保留完整原始 kernel
名称和 `parsed:kernel-name` 标记。CUDA/cuBLAS 升级后名称不匹配已知规则时，
这些字段应写 `unknown`，不能退回到未标注的猜测。

NCU 的 `Block=(128,1,1)` 可直接得到 threads/CTA，再推导4 warps/CTA；
它仍不能唯一确定 WarpShape。任何可能的 WarpShape 只能保存为
`derived_warp_candidates`，最终必须由 CUTLASS 实测选择。

### 完整optimal调优过程

`tune_optimal_cutlass.sh` 对每种 accumulator 独立执行：

1. 生成所有目标模型、Decode/Prefill长度的去重 `M/N/K` manifest。
2. 按相同 accumulator 构建 CUTLASS 与 cuBLASLt profiler。
3. 对每个 shape 实测 cuBLASLt heuristic winner，保存直接API字段。
4. 将 cuBLASLt 信息转换成带来源标记的 CUTLASS候选并重新构建。
5. 对基础候选、FP32专属候选和 cuBLASLt-derived候选执行多轮实测。
6. 仅保留每轮都成功的配置，按 `avg_time` 中位数选出最优。
7. 将 winner 写入 accumulator感知的 `optimal_configurations.inc`。
8. 与另一 accumulator 的已有映射合并，禁止跨类型复用。
9. 构建 optimal-only并运行覆盖检查；任何 fallback 都使验证失败。

最终报告中的：

```text
configuration_origin = direct:measured-cutlass-candidate
selection_origin     = derived:minimum-median-time-across-rounds
```

表示模板参数来自实际执行的 CUTLASS候选，而“选择它为winner”是由多轮实测
中位数规则推导出的结论。

## 为全部 Qwen2.5 参数量生成目标机器 optimal

完整调优覆盖 `0.5b,1.5b,3b,7b,14b,32b,72b`，默认 Batch=1，并对所有默认 Decode L、Prefill S 的 `M/N/K` 全局去重：

```bash
bash tune_optimal_cutlass.sh \
  --models all \
  --arch sm_89 \
  --accumulator fp16 \
  --rounds 3 \
  --iterations 100
```

FP32 必须独立调优：

```bash
bash tune_optimal_cutlass.sh \
  --models all \
  --arch sm_89 \
  --accumulator fp32 \
  --rounds 3 \
  --iterations 100
```

两种累加器分别写入 `outputs/qwen_gemm/optimal_tuning/fp16` 和
`outputs/qwen_gemm/optimal_tuning/fp32`，
候选测量日志、cuBLASLt compute type、报告与 metadata 均不混用。生成器会将
新类型映射和 `optimal_configurations.inc` 中另一类型的映射合并到同一个文件。
普通全候选模式也会在 FP32 构建时额外比较较小 CTA/较低寄存器压力的
`FP32-Compact` 与 `FP32-Large-M` 候选。

该脚本先实测 cuBLASLt，并生成 `cublaslt_generated_candidates.inc`；随后把 cuBLASLt-derived 模板和 CUTLASS 基础模板全部执行多轮，按每个候选的 `avg_time` 中位数选出赢家。最终生成：

```text
examples/gemm/optimal_configurations.inc
outputs/qwen_gemm/optimal_tuning/<fp16|fp32>/optimal_configurations_report.csv
outputs/qwen_gemm/optimal_tuning/<fp16|fp32>/optimal_configurations_metadata.json
```

生成映射保存布局、alignment、ThreadblockShape、WarpShape、Swizzle、Stages 和 Split-K。调优开始前，原文件会备份到 `outputs/qwen_gemm/optimal_tuning/<fp16|fp32>/optimal_configurations.inc.previous`；新文件生成成功后再原子替换活动配置。`cublaslt_generated_candidates.inc` 是调优阶段候选，不能替代最终 optimal 文件。

调优脚本结束时已经构建 optimal-only 可执行文件并进行完整命中检查。之后无需再次调优，可直接运行任意标准模型：

```bash
bash run_gemm.sh --model 32b
```

如果重新拉取代码或重新编译，只要保留生成文件即可直接构建 optimal-only：

```bash
bash build_gemm.sh --arch sm_89 --optimal-only --concise-log
bash run_gemm.sh --model 0.5b
bash run_gemm.sh --model 72b
```

只调部分模型时可使用 `--models 7b,14b,32b`，但生成文件只覆盖所选模型，并会替换当前活动 optimal；需要兼容所有标准参数量时应使用 `--models all`。使用自定义 H、I、V、Batch 或 L/S 产生的新 shape 若未覆盖，将使用确定性 fallback。

## 运行 Decode / Prefill 用例

不指定 `--stage` 时依次运行全部 Decode 和 Prefill 用例：

```bash
./run_gemm.sh --model 7b
```

也可以显式使用 `--stage all`。任一阶段出现失败时组合命令返回非 0；两个阶段会分别输出 summary。

运行 Qwen2.5-7B：

```bash
./run_gemm.sh --model 7b --stage decode
```

默认上下文长度为：

```text
128, 129, 130, 131, 133, 137, 256, 257, 512, 513, 1024, 1025, 2048, 2049
```

脚本先生成算子形状，再严格按 `M/N/K` 去重：与 `L` 无关的 Linear GEMM 只运行一次；只有 `Attention QK^T` 和 `Attention PV` 会随 `L` 生成新形状。相同形状的来源算子会合并记录，例如 `Q / Attention Out`、`K / V` 和 `MLP Up / MLP Gate`。Q 与 Attention Out、K 与 V、Up 与 Gate 在模型中分别独立调用，因此整模型估算时每组时间均乘 2；去重只减少基准测试次数，不减少模型调用次数。

默认列表既保留常用 L 边界，也包含各默认 Prefill `S` 对应的首次 Decode `L=S+1`。Qwen2.5-7B 最终运行 32 个唯一 Decode GEMM，而不是原始展开后的调用。每个唯一 GEMM 都会比较：

- `Linear TB..._W..._S...`
- `Attention-QK TB..._W..._S...`
- `Attention-PV-StreamK TB..._W..._S...`

运行 Qwen2.5-7B Prefill：

```bash
./run_gemm.sh --model 7b --stage prefill
```

Prefill 默认序列长度为：

```text
128, 256, 512, 1024, 2048, 129, 130, 132, 136
```

Prefill 的层内 GEMM 随 `S` 变化，但 LM Head 只为提示词最后一个位置生成 logits，因此始终使用 `1 x V x H`，与 Decode LM Head 是同一个全局去重 shape，并且只在所有 Transformer 层之后执行一次。Qwen2.5-7B Prefill 去重后包含 54 个唯一 GEMM；同一 `S` 下的 `Q / Attention Out`、`K / V`、`MLP Up / MLP Gate` 以及相同形状的 Attention 算子会合并记录。

本测试用于估算推理性能时采用以下口径：Batch=1；Q/K/V 与 Up/Gate 不融合；Attention QK^T/PV 使用 strided-batched GEMM；A/B/C/D 使用 FP16 近似 BF16 存储，默认使用 FP32 accumulator 和 FP32 Epilogue compute，最终写回 FP16。该结果不包含 embedding、RMSNorm、RoPE、softmax/mask、激活、残差、KV 管理、kernel 间隙和多卡通信，因此不能当作完整端到端延迟。

大 M Prefill 默认比较六组安全配置：

```text
128x128x32，Warp 64x64x32
128x256x32，Warp 64x64x32
256x128x32，Warp 64x64x32
64x128x32，Warp 32x64x32
128x64x32，Warp 64x32x32
64x256x32，Warp 32x64x32
```

ColumnMajor 用例分别计算各操作数的最大合法 FP16 向量宽度：A/C 的连续维度是 M，因此 Alignment A/C 由 M 决定；B 的连续维度是 K，因此 Alignment B 由 K 决定。例如 `M%4=0、M%8!=0、K%8=0` 使用 `4/8/4`，而 `M%8=0、K%4=0、K%8!=0` 使用 `8/4/8`。不做 padding；A 或 B 任一侧为 Alignment 1 时使用 Stages 2 同步 fallback，否则大 M 使用 Stages 3。默认 Prefill S=129/130/132 覆盖 Alignment 1/2/4。

## 常用运行参数

只输出将要运行的命令：

```bash
./run_gemm.sh --model 7b --stage prefill --dry-run
```

指定 Decode L 或 Prefill S：

```bash
./run_gemm.sh --model 7b --stage prefill --lengths 128,129,256
```

指定迭代次数：

```bash
./run_gemm.sh --model 7b --iterations 100
```

运行其他模型：

```bash
./run_gemm.sh --model 0.5b
./run_gemm.sh --model 1.5b
./run_gemm.sh --model 3b
./run_gemm.sh --model 14b
./run_gemm.sh --model 32b
./run_gemm.sh --model 72b
```

也可以覆盖模型参数：

```bash
./run_gemm.sh \
  --model 7b \
  --stage prefill \
  --h 4096 \
  --heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --intermediate 22016 \
  --vocab 152064
```

查看帮助：

```bash
./build_gemm.sh --help
./run_gemm.sh --help
```

## 输出

每个 GEMM 用例会输出：

- `M/N/K`
- 实际 kernel 配置
- `Status`
- `avg_time`
- `gflops`
- 最佳配置

失败定位字段同时写入普通 `Results` 和结构化 `CUTLASS_CANDIDATE/CUTLASS_BEST` 记录：

```text
failure_stage: event_sync_stop
failure_location: examples/gemm/gemm.cu:...
cuda_error: 700 (cudaErrorIllegalAddress: an illegal memory access was encountered)
```

`failure_stage` 可能为 `preexisting_cuda_error`、`can_implement`、`initialize`、`warmup_launch`、`warmup_launch_cuda`、`warmup_sync`、`timed_launch`、`chrono_pre_sync`、`chrono_post_sync`、`event_create_start/stop`、`event_record_start/stop`、`event_sync_stop`、`event_elapsed_time`、`event_destroy_start/stop`、`verification_copy` 或 `verification_compare`。成功时为 `none`、源码行为0、CUDA错误为 `cudaSuccess`。`preexisting_cuda_error` 表示前一个候选的异步错误在当前候选开始前才被观察到，应优先检查日志中的前一个候选。

全部用例完成后输出：

```text
Decode GEMM summary
  total: 32
  passed: 32
  failed: 0
```

Prefill 默认输出：

```text
Prefill GEMM summary
  total: 54
  passed: 54
  failed: 0
```

## 汇总 GEMM-only 模型估算

先运行包含目标长度的用例并保存日志，再按模型层数和实际调用次数汇总最佳 kernel 时间：

```bash
./run_gemm.sh --model 7b --stage prefill --lengths 128 | tee cutlass_prefill.log
python3 tools/qwen_gemm/estimate_qwen_gemm.py --log cutlass_prefill.log --model 7b --stage prefill --length 128

./run_gemm.sh --model 7b --stage decode --lengths 129 | tee cutlass_decode.log
python3 tools/qwen_gemm/estimate_qwen_gemm.py --log cutlass_decode.log --model 7b --stage decode --length 129
```

若 prompt 长度为 `P`，Prefill 使用 `S=P`；完成 Prefill 并开始下一次 Decode forward 时，Attention KV 长度使用 `L=P+1`。估算器会对 `Q/Out`、`K/V`、`Up/Gate` 各乘 2，对层内 GEMM 乘模型层数，LM Head 只计一次。请勿平均各用例 GFLOPS，整模型延迟必须累加 `avg_time × 调用次数`。

### 一个 forward 的含义与计算

`estimate_qwen_gemm.py` 每次只估算一个 forward：

- `--stage prefill --length S`：一次完整 Prefill forward。该 forward 处理全部 S 个 prompt token，并由最后位置 LM Head 产生第一个输出 token。
- `--stage decode --length L`：一次 Decode forward。该 forward 读取长度为 L 的 KV Cache，并产生一个新 token。

单个 forward 的 GEMM-only 时间按下式计算：

```text
per_layer = 2*T(Q/Out) + 2*T(K/V) + T(QK^T) + T(PV)
          + 2*T(Up/Gate) + T(Down)
forward   = num_layers*per_layer + T(LM_Head_M1)
```

其中每个 `T(shape)` 都来自日志中该 `M/N/K` 的实测最佳 `avg_time`。当前估算器是严格实测模式：缺少任一所需 shape 就报错，不会使用理论峰值、FLOPs 比例、相邻 L 插值或外推值补齐。

例如 `S=128、G=128`，如果 G 包含 Prefill 直接生成的第一个 token，则完整生成过程需要一次 Prefill `S=128`，再执行127次 Decode，`L=129...255`：

```text
T_total = T_prefill(128) + sum(T_decode(L), L=129...255)
```

当前脚本没有 `S/G` 自动循环汇总接口；要得到完全基于实测的总时间，需要日志覆盖上述每个 L，再逐个执行估算并求和。

### 基于全量实测数据估算 S/G

下面示例估算 Qwen2.5-7B、prompt 长度 `S=128`、总生成 token 数 `G=128`。这里 G **包含 Prefill 直接产生的第一个 token**，因此需要实测一次 Prefill `S=128` 和127次 Decode `L=129...255`。

```bash
MODEL=7b
S=128
G=128
FIRST_L=$((S + 1))
LAST_L=$((S + G - 1))
DECODE_LENGTHS=$(seq -s, "$FIRST_L" "$LAST_L")

./run_gemm.sh \
  --model "$MODEL" \
  --stage prefill \
  --lengths "$S" \
  | tee "prefill_s${S}.log"

./run_gemm.sh \
  --model "$MODEL" \
  --stage decode \
  --lengths "$DECODE_LENGTHS" \
  | tee "decode_s${S}_g${G}.log"
```

然后逐个 forward 读取实测最佳 `avg_time` 并累加：

```bash
PREFILL_MS=$(
  python3 tools/qwen_gemm/estimate_qwen_gemm.py \
    --log "prefill_s${S}.log" \
    --model "$MODEL" \
    --stage prefill \
    --length "$S" \
  | awk '/GEMM-only lower-bound latency:/ {print $4}'
)

DECODE_MS=$(
  for L in $(seq "$FIRST_L" "$LAST_L"); do
    python3 tools/qwen_gemm/estimate_qwen_gemm.py \
      --log "decode_s${S}_g${G}.log" \
      --model "$MODEL" \
      --stage decode \
      --length "$L"
  done \
  | awk '/GEMM-only lower-bound latency:/ {sum += $4} END {printf "%.6f", sum}'
)

awk -v prefill="$PREFILL_MS" -v decode="$DECODE_MS" \
  'BEGIN {
    total = prefill + decode
    printf "Prefill GEMM-only: %.6f ms\n", prefill
    printf "Decode GEMM-only:  %.6f ms\n", decode
    printf "Total GEMM-only:   %.6f ms\n", total
  }'
```

上述结果全部来自目标机器日志中的实测 kernel 时间，没有对缺失 L 做插值。如果 `G=1`，只有 Prefill，不需要运行 Decode。若 G 表示“Prefill 之后额外生成的 token 数”，则 Decode 应实测 `L=S+1...S+G`，而不是 `S+G-1`。

## CUTLASS 与 HGEMM 结果对比

仓库提供：

```text
collect_gemm_results.py           CUTLASS 日志解析和结果回填脚本
gemm_performance_comparison.xlsx  Excel 对比模板
```

先运行 CUTLASS 用例并保存完整日志。脚本按 `M/N/K` 定位 Excel 模板中的行，只更新 CUTLASS 最佳配置和 GFLOPS；HGEMM 数据、加速比公式和格式均会保留。

直接更新原始 XLSX：

```bash
./run_gemm.sh --model 7b --stage decode | tee cutlass_decode.log
python3 tools/qwen_gemm/collect_gemm_results.py \
  --log cutlass_decode.log \
  --workbook ../../qwen_gemm_artifacts/gemm_performance_comparison.xlsx

./run_gemm.sh --model 7b --stage prefill | tee cutlass_prefill.log
python3 tools/qwen_gemm/collect_gemm_results.py \
  --log cutlass_prefill.log \
  --workbook ../../qwen_gemm_artifacts/gemm_performance_comparison.xlsx
```

如需保留空白模板，可输出到新 XLSX：

```bash
python3 tools/qwen_gemm/collect_gemm_results.py \
  --log cutlass_prefill.log \
  --workbook ../../qwen_gemm_artifacts/gemm_performance_comparison.xlsx \
  --output gemm_performance_comparison_filled.xlsx
```

`--output` 必须使用 `.xlsx` 后缀；省略时原地更新 `--workbook`。如果日志中的 `M/N/K` 不存在于模板中，脚本会报错且不会写入不完整结果。

HGEMM（自研）和 HGEMM（CUDA）数据来自其他测试程序，输出格式与 `gemm.cu` 不同，因此不由脚本解析。请按相同的 `M/N/K` 分别人工筛选两类 HGEMM 最佳 GFLOPS，再填写 Excel 模板中的人工录入列。

更新后的 `gemm_performance_comparison.xlsx` 包含：

- 阶段、相关 L/S 和来源算子名称（相同 `M/N/K` 合并）
- `M/N/K`
- CUTLASS 最佳配置与 GFLOPS（脚本自动更新）
- HGEMM（自研）GFLOPS（人工筛选，已有内容会被保留）
- HGEMM（CUDA）GFLOPS（人工筛选，已有内容会被保留）
- 两类 HGEMM 分别相对 CUTLASS 的加速比和人工筛选备注

Excel 模板按 `M/N/K` 全局去重，预置 Qwen2.5-7B 的 85 个唯一 GEMM；其中 Decode 与 Prefill 共用 `1 x V x H` 的最后位置 LM Head。蓝色列由脚本直接回填 CUTLASS 数据；黄色和橙色列分别用于人工录入 HGEMM（自研）及 HGEMM（CUDA）结果；绿色列自动计算两者相对 CUTLASS 的加速比。

## 注意事项

- 为了与 cuBLAS 的 ColumnMajor 物理布局比较，kernel 按 M 选择布局：
  - `M=1`：A/C 的 RowMajor 与 ColumnMajor 物理存储等价，使用 A/B/C alignment `8/8/8`。
  - 小 M 且对齐的 Decode：A/B/C 均使用 ColumnMajor；A/C 根据 M 选择 alignment `8/4/2`，B使用 alignment 8。
  - `M>=128`：A/C 按 M、B 按 K 独立选择 alignment 8/4/2/1；任一输入为 alignment 1 时使用 Stages 2，否则使用 Stages 3。
  - 默认 Prefill 的 S=129/130/132 分别覆盖 alignment 1/2/4；所有用例均不做 padding。
- 例如 `M=28` 使用 A/B/C alignment `4/8/4`，`M=14` 使用 `2/8/2`，`M=40` 使用 `8/8/8`。
- FP16 Alignment 1 只有 2 字节，不能用于 SM80 `cp.async`；Alignment 1 fallback 固定使用 Stages 2，以选择 CUTLASS 的同步 `MmaPipelined` 双缓冲主循环。
- LM Head 固定为 `M=1`，只生成最后位置 logits；部分 MLP 用例仍会分配较大的矩阵，需要足够的主机内存和 GPU 显存。
- `run_gemm.sh` 默认运行当前 CUTLASS 根目录下的 `examples/gemm/gemm`。
## Qwen2.5 / cuBLAS trans 与操作数映射

测试程序按传统 cuBLAS 的列主序调用语义接收 `M/N/K`，并支持
`--trans=NN|NT|TN|TT`。`trans` 不能脱离 A/B 的身份单独修改，因此
`run_gemm.sh` 会同时传入 `--operation`、`--operand-a` 和 `--operand-b`。
手动运行时，标准 `--operation` 会自动提供 A/B 和 trans 默认值；显式
传入的 `--operand-a`、`--operand-b` 或 `--trans` 分别覆盖对应默认值。

默认 Qwen2.5 映射如下：

| Qwen 计算 | cuBLAS A | cuBLAS B | trans |
|---|---|---|---|
| Q/K/V/O projection | 对应 projection weight | activation | TN |
| MLP Up/Gate/Down | 对应 MLP weight | MLP activation | TN |
| LM Head | `lm_head.weight` | last hidden state | TN |
| Attention QK^T | K | Q | TN |
| Attention PV | V | softmax probability P | NN |

Linear 的模型行主序计算为 `X * W^T`，cuBLAS 侧模拟其转置
`W * X^T`，因此 A/B 交换为 `A=Weight,B=Activation`。QK^T 同理以
`A=K,B=Q,TN` 表示；PV 以 `A=V,B=P,NN` 表示。程序输出会同时显示
Qwen operation、A/B 身份、trans、物理 A/B 尺寸以及 CUTLASS layout。

手动运行示例：

```bash
# Qwen Linear：模型输出 [tokens, out]，cuBLAS M=out、N=tokens
./examples/gemm/gemm --m=896 --n=6 --k=896 \
  --operation=q_proj --operand-a=W_Q --operand-b=HiddenStates --trans=TN

# Attention PV 的 cuBLAS 表达
./examples/gemm/gemm --m=64 --n=6 --k=6 --operation=pv
# 上式自动采用 A=V、B=P、trans=NN

# 也可显式测试其他调用形式
./examples/gemm/gemm --m=64 --n=64 --k=64 --trans=NT
```

`generate_qwen_shapes.py` 生成的清单按 `M/N/K/trans/batch_count` 去重，并
额外记录 `operand_a`、`operand_b`。Attention QK^T/PV 使用单 Head 的
M/N/K 和 `batch_count=batch*heads`，不再把 Head 数展平到 M/N。
Batch size 只进入 `batch_count`：Decode 的单 Head GEMM 固定 `N=1`，
Prefill 固定 `N=S`，避免在 N 与 batch_count 中重复计算 batch。

例如 Qwen2.5-0.5B Decode、L=7：

```bash
./examples/gemm/gemm --m=7 --n=1 --k=64 \
  --batch-count=14 --operation=qk
./examples/gemm/gemm --m=64 --n=1 --k=7 \
  --batch-count=14 --operation=pv
```

前者自动使用 `A=K,B=Q,TN`，后者使用 `A=V,B=P,NN`。GFLOPS 按
`2*M*N*K*batch_count/avg_time` 计算；初始化、reference 和结果比较均
覆盖全部 batch。当前采用固定 strided batch，等价于 K/V 已按 Q Head
组织或物化后的调用，不模拟未物化 GQA 的非线性 KV Head 指针复用。
