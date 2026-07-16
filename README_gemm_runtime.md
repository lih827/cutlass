# Qwen2.5 Decode / Prefill GEMM 运行说明

本测试使用 CUTLASS TensorOp GEMM，根据 Qwen2.5 模型参数生成 Decode 和 Prefill 阶段的 `M/N/K`，按 `M/N/K` 去重后运行，并从内置候选配置中选择实测性能最佳的结果。

## 文件放置

将交付文件放入 CUTLASS 源码根目录，形成以下结构：

```text
cutlass/
├── README_gemm_runtime.md
├── build_gemm.sh
├── run_gemm.sh
├── collect_gemm_results.py
├── gemm_performance_comparison.xlsx
├── include/
├── tools/
└── examples/
    └── gemm/
        ├── gemm.cu
        └── cublaslt_generated_candidates.inc  # 可选
```

`build_gemm.sh` 和 `run_gemm.sh` 必须从 CUTLASS 根目录执行。

## 环境要求

- Linux
- Bash 4.0 或更高版本
- CUDA Toolkit 9.0 或更高版本
- CUTLASS 源码
- SM80 或更高架构的 NVIDIA GPU
- Python 3 与 `openpyxl`，仅在需要回填 XLSX 时使用

当前 GEMM 使用：

```text
Element A/B/C    = half
Accumulator      = half
OperatorClass    = OpClassTensorOp
ArchTag          = Sm80
InstructionShape = 16x8x16
```

## 可选配置文件

`examples/gemm/cublaslt_generated_candidates.inc` 保存按精确 `M/N/K` 匹配的附加 CUTLASS 模板配置。

- 文件存在且当前 `M/N/K` 有对应记录时：附加模板会与程序内置 CUTLASS 候选共同运行，最终按实测时间选择最佳项。
- 文件不存在、内容为空或当前 shape 没有记录时：程序直接使用内置 CUTLASS 候选。
- 附加模板无法运行或结果校验失败时：程序忽略该模板，继续使用内置候选。
- `.inc` 只参与编译；生成可执行文件后，运行阶段不再读取它。

因此 `.inc` 可有可无。需要明确使用内置候选时，可以删除该文件后重新编译：

```bash
rm -f examples/gemm/cublaslt_generated_candidates.inc
bash build_gemm.sh --arch sm_89
```

## 编译

默认针对 `sm_89`：

```bash
bash build_gemm.sh
```

指定 GPU 架构：

```bash
bash build_gemm.sh --arch sm_80
bash build_gemm.sh --arch sm_86
bash build_gemm.sh --arch sm_89
```

指定 `nvcc`：

```bash
bash build_gemm.sh --nvcc /usr/local/cuda/bin/nvcc --arch sm_89
```

默认编译版本会输出所有候选配置及结果。只输出最终最佳配置时使用：

```bash
bash build_gemm.sh --arch sm_89 --concise-log
```

精简日志只抑制非最佳候选的输出，不会减少实际测试的候选数量。恢复完整日志需要重新执行不带 `--concise-log` 的 build。

GEMM 可执行文件为：

```text
examples/gemm/gemm
```

## 运行全部用例

不指定 `--stage` 时依次运行 Decode 和 Prefill：

```bash
bash run_gemm.sh --model 7b
```

也可以显式指定：

```bash
bash run_gemm.sh --model 7b --stage all
```

两个阶段分别输出 summary；任一阶段存在失败时，组合命令返回非 0。

## 单独运行 Decode 或 Prefill

只运行 Decode：

```bash
bash run_gemm.sh --model 7b --stage decode
```

Decode 默认上下文长度：

```text
128, 256, 512, 1024, 2048
```

只运行 Prefill：

```bash
bash run_gemm.sh --model 7b --stage prefill
```

Prefill 默认序列长度：

```text
128, 256, 512, 1024, 2048, 129, 130, 132, 136
```

Qwen2.5-7B 默认包含14个唯一 Decode GEMM和62个唯一 Prefill GEMM。相同 `M/N/K` 的来源算子会合并，只执行一次。

## 支持的模型

```bash
bash run_gemm.sh --model 0.5b
bash run_gemm.sh --model 1.5b
bash run_gemm.sh --model 3b
bash run_gemm.sh --model 7b
bash run_gemm.sh --model 14b
bash run_gemm.sh --model 32b
bash run_gemm.sh --model 72b
```

## 常用参数

指定迭代次数：

```bash
bash run_gemm.sh --model 7b --iterations 100
```

指定 Decode L 或 Prefill S：

```bash
bash run_gemm.sh --model 7b --stage decode --lengths 128,512,2048
bash run_gemm.sh --model 7b --stage prefill --lengths 128,129,256
```

只打印将要执行的命令：

```bash
bash run_gemm.sh --model 7b --dry-run
```

覆盖模型参数：

```bash
bash run_gemm.sh \
  --model 7b \
  --h 4096 \
  --heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --intermediate 22016 \
  --vocab 152064
```

查看帮助：

```bash
bash build_gemm.sh --help
bash run_gemm.sh --help
```

## 内置候选

小 M 用例会比较：

```text
Linear
Attention QK^T
Attention AV Stream-K
```

大 M 用例会比较：

```text
Threadblock 128x128x32，Warp 64x64x32
Threadblock 128x256x32，Warp 64x64x32
Threadblock 256x128x32，Warp 64x64x32
Threadblock 64x128x32， Warp 32x64x32
Threadblock 128x64x32， Warp 64x32x32
Threadblock 64x256x32， Warp 32x64x32
```

对齐的较大 M 用例使用 alignment `8/8/8`、Stages 3。非对齐用例不做 padding，使用 alignment `1/1/1`、同步双缓冲 Stages 2。

## 输出

完整日志会为每个候选输出：

- 实际 ThreadblockShape、WarpShape、InstructionShape
- A/B/C布局与 alignment
- Stages、Split-K 和 Swizzle
- Status
- avg_time
- GFLOPS

每个用例最后输出：

```text
Best configuration: ...
  avg_time: ... ms
  gflops: ...
```

## 回填性能对比表

安装依赖：

```bash
python3 -m pip install openpyxl
```

运行并保存日志：

```bash
bash run_gemm.sh --model 7b --stage decode | tee cutlass_decode.log
bash run_gemm.sh --model 7b --stage prefill | tee cutlass_prefill.log
```

更新原始 XLSX：

```bash
python3 collect_gemm_results.py \
  --log cutlass_decode.log \
  --workbook gemm_performance_comparison.xlsx

python3 collect_gemm_results.py \
  --log cutlass_prefill.log \
  --workbook gemm_performance_comparison.xlsx
```

输出到新文件：

```bash
python3 collect_gemm_results.py \
  --log cutlass_prefill.log \
  --workbook gemm_performance_comparison.xlsx \
  --output gemm_performance_comparison_filled.xlsx
```

脚本只更新 CUTLASS 最佳配置和 GFLOPS，保留 HGEMM 数据、公式与格式。

## 注意事项

- `M=1` 时 A/C使用 RowMajor，B使用 ColumnMajor；A/B/C alignment 为 `8/8/8`。
- `M>1` 时 A/B/C使用 ColumnMajor。
- alignment=1 的 FP16 访问为2字节，不能使用 SM80 `cp.async`，因此固定采用同步 `MmaPipelined` Stages 2。
- LM Head 和部分 MLP 用例需要较大的主机内存与 GPU显存。
- 更换 GPU、CUDA版本或 `.inc` 后必须重新执行 build。
