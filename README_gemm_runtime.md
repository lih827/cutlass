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
        └── optimal_configurations.inc  # 可选的精确 M/N/K 最优映射
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

默认使用 CUDA Event 计时。编译时可切换为 `std::chrono::steady_clock`：

```bash
bash build_gemm.sh --arch sm_89 --timer cuda-event
bash build_gemm.sh --arch sm_89 --timer chrono
```

`chrono` 模式在计时前后同步 GPU，输出仍使用相同的 `avg_time` 和 `gflops` 字段。短 kernel 的精确性能比较建议保留默认的 `cuda-event`。

只运行一个模板以缩短测试时间：

```bash
bash build_gemm.sh --arch sm_89 --optimal-only
```

`--optimal-only` 只读取 `examples/gemm/optimal_configurations.inc`。该文件可以是交付包自带配置，也可以替换为目标环境已经生成的同格式配置。命中精确 `M/N/K` 时只执行对应模板；文件不存在或 shape 未命中时确定性地选择 fallback。编译和运行过程本身不依赖任何配置生成程序。

已确认模板正确后，可关闭 reference 和结果校验以减少整用例耗时：

```bash
bash build_gemm.sh --arch sm_89 --optimal-only --skip-verification
```

关闭后直接在 Device 端清零 A/B/C，不执行 Host 随机初始化/H2D、reference GEMM、校验相关 D2H 和 `TensorEquals`；输出为 `verification: disabled`、`Status: Not verified`。默认 build 仍保留完整正确性校验。

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

Qwen2.5-7B 默认包含 14 个唯一 Decode GEMM 和 62 个唯一 Prefill GEMM。相同 `M/N/K` 的来源算子会合并，只执行一次；`MLP Up` 与 `MLP Gate` 均采用 `N=I`，并作为相同 shape 合并统计。

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
Linear TB..._W..._S...
Attention-QK TB..._W..._S...
Attention-PV-StreamK TB..._W..._S...
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
