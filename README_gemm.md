# Qwen2.5 Decode / Prefill GEMM 测试

本测试使用 CUTLASS TensorOp GEMM，根据不同 Qwen2.5 模型参数生成 Decode 或 Prefill 阶段的 `M/N/K`，按 `M/N/K` 去重后运行，并在小 M 与大 M 候选配置中分别选择最佳结果。

## 放置目录

将文件放到 CUTLASS 源码目录中，形成以下结构：

```text
cutlass/
├── README_gemm.md
├── build_gemm.sh
├── run_gemm.sh
├── tune_cutlass_from_cublaslt.sh
├── generate_cutlass_candidates.py
├── include/
├── tools/
└── examples/
    └── gemm/
        ├── gemm.cu
        ├── cublaslt_profiler.cu
        └── cublaslt_generated_candidates.inc
```

`build_gemm.sh` 和 `run_gemm.sh` 都必须从 CUTLASS 根目录执行。

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
chmod +x tune_cutlass_from_cublaslt.sh
./tune_cutlass_from_cublaslt.sh --model 7b --arch sm_89 --iterations 20
```

该脚本依次完成：

1. 构建 CUTLASS 测试程序和 cuBLASLt profiler。
2. 对去重后的全部 Decode、Prefill `M/N/K` 请求 cuBLASLt heuristic 候选并逐一计时。
3. 记录每个 shape 的最优 Algo ID、Tile、Stages、Split-K、Swizzle 和 workspace。
4. 运行 `generate_cutlass_candidates.py`，把 cuBLASLt Tile/Stages 转换成合法的 SM80 CUTLASS Threadblock/Warp 模板。
5. 对现有候选没有覆盖的合法 Tile 自动新增模板，并重新编译 `gemm`。
6. 后续运行命中同一 `M/N/K` 时只执行生成的模板；没有映射的 shape 自动回退到原有候选比较。

`cublaslt_generated_candidates.inc` 是可选文件。文件尚未生成或被删除时，`gemm.cu` 仍可正常编译，`run_gemm.sh` 会使用原有候选集合；执行调优脚本后才启用精确 shape 映射。

生成文件包括：

```text
examples/gemm/cublaslt_generated_candidates.inc
cublaslt_tuning/cublaslt_decode.log
cublaslt_tuning/cublaslt_prefill.log
cublaslt_tuning/cublaslt_cutlass_mapping.csv
```

cuBLASLt 的 Tile/Stages 不是 CUTLASS 模板的完整描述。生成器遵循以下约束：

- Tile 在合法目录中存在时精确采用；现有三候选未覆盖但可合法构造的 Tile 会新增 CUTLASS 模板；其余情况选择最近的合法 TensorOp Tile，并在 CSV 中标记 `tile_mapping`。
- cuBLASLt 单缓冲 Stages 会提升为 CUTLASS 合法的最小 Stages 2。
- 自动生成仅使用已经通过 SM80 编译检查的 WarpShape（每个维度至少 32、每个 CTA 不超过 8 个 warp）；非常规 cuBLASLt Tile 会映射到最近的安全 CUTLASS Tile。
- Threadblock K 固定为已验证的 32；cuBLASLt 的内部 stage-size 不直接当作 CUTLASS Threadblock K。Stages 限制在 2～4，以避免非法单缓冲或过大的共享内存配置。
- cuBLASLt 的 Split-K 数量会写入生成模板，并由 CUTLASS `GemmUniversal` 使用对应的 split-K slices。
- alignment=1 不做 padding，固定使用 Threadblock K=32、同步 `MmaPipelined` Stages 2。
- 生成结果与 GPU、CUDA/cuBLASLt 版本、workspace 和矩阵布局相关，换机器或升级 CUDA 后应重新执行调优脚本。

## 运行 Decode / Prefill 用例

运行 Qwen2.5-7B：

```bash
./run_gemm.sh --model 7b --stage decode
```

默认上下文长度为：

```text
128, 256, 512, 1024, 2048
```

脚本先生成算子形状，再严格按 `M/N/K` 去重：与 `L` 无关的 Linear GEMM 只运行一次；只有 `Attention QK^T` 和 `Attention AV` 会随 `L` 生成新形状。相同形状的来源算子会合并记录，例如 `Q / Attention Out` 和 `K / V`。

默认 Qwen2.5-7B 最终运行 14 个唯一 GEMM，而不是原始展开后的 45 次调用。每个唯一 GEMM 都会比较：

- Linear
- Attention QK^T
- Attention AV Stream-K

运行 Qwen2.5-7B Prefill：

```bash
./run_gemm.sh --model 7b --stage prefill
```

Prefill 默认序列长度为：

```text
128, 256, 512, 1024, 2048, 129, 130, 132, 136
```

Prefill 中所有算子的形状都会随 `S` 变化。Qwen2.5-7B 去重后包含 62 个唯一 GEMM；同一 `S` 下的 `Q / Attention Out`、`K / V` 以及相同形状的 Attention 算子会合并记录。

大 M Prefill 会比较三组配置：

```text
128x128x32，Warp 64x64x32
128x256x32，Warp 64x64x32
256x128x32，Warp 64x64x32
```

M、K 都能被 8 整除时使用 Alignment A/B/C `8/8/8` 和 Stages 3；否则不做 padding，改用 Alignment `1/1/1`、Stages 2 的同步双缓冲 fallback。

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

全部用例完成后输出：

```text
Decode GEMM summary
  total: 14
  passed: 14
  failed: 0
```

Prefill 默认输出：

```text
Prefill GEMM summary
  total: 62
  passed: 62
  failed: 0
```

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
python3 collect_gemm_results.py \
  --log cutlass_decode.log \
  --workbook gemm_performance_comparison.xlsx

./run_gemm.sh --model 7b --stage prefill | tee cutlass_prefill.log
python3 collect_gemm_results.py \
  --log cutlass_prefill.log \
  --workbook gemm_performance_comparison.xlsx
```

如需保留空白模板，可输出到新 XLSX：

```bash
python3 collect_gemm_results.py \
  --log cutlass_prefill.log \
  --workbook gemm_performance_comparison.xlsx \
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

Excel 模板在原有 sheet 中预置了 Qwen2.5-7B 的 14 个 Decode 和 62 个 Prefill 唯一 GEMM。蓝色列由脚本直接回填 CUTLASS 数据；黄色和橙色列分别用于人工录入 HGEMM（自研）及 HGEMM（CUDA）结果；绿色列自动计算两者相对 CUTLASS 的加速比。

## 注意事项

- 为了与 cuBLAS 的 ColumnMajor 物理布局比较，kernel 按 M 选择布局：
  - `M=1`：A/C 的 RowMajor 与 ColumnMajor 物理存储等价，使用 A/B/C alignment `8/8/8`。
  - 小 M 且对齐的 Decode：A/B/C 均使用 ColumnMajor；A/C 根据 M 选择 alignment `8/4/2`，B使用 alignment 8。
  - `M>=128` 且 M/K 均能被8整除：使用大 M 配置、alignment `8/8/8`、Stages 3。
  - 其他非对齐 M/K：不做 padding，使用大 M 配置、alignment `1/1/1`、Stages 2。
- 例如 `M=28` 使用 A/B/C alignment `4/8/4`，`M=14` 使用 `2/8/2`，`M=40` 使用 `8/8/8`。
- FP16 Alignment 1 只有 2 字节，不能用于 SM80 `cp.async`；Alignment 1 fallback 固定使用 Stages 2，以选择 CUTLASS 的同步 `MmaPipelined` 双缓冲主循环。
- LM Head 和部分 MLP 用例会分配较大的矩阵，需要足够的主机内存和 GPU 显存。
- `run_gemm.sh` 默认运行当前 CUTLASS 根目录下的 `examples/gemm/gemm`。
