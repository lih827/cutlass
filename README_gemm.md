# Qwen2.5 Decode GEMM 测试

本测试使用 CUTLASS TensorOp GEMM，根据不同 Qwen2.5 模型参数生成 Decode 阶段的 `M/N/K`，依次运行所有用例并比较三组 kernel 配置。

## 放置目录

将文件放到 CUTLASS 源码目录中，形成以下结构：

```text
cutlass/
├── README_gemm.md
├── build_gemm.sh
├── run_gemm.sh
├── include/
├── tools/
└── examples/
    └── gemm/
        └── gemm.cu
```

`build_gemm.sh` 和 `run_gemm.sh` 都必须从 CUTLASS 根目录执行。

## 环境要求

- Linux
- Bash 4.0 或更高版本
- CUDA Toolkit
- CUTLASS 源码
- SM80 或更高架构的 NVIDIA GPU

当前 kernel 使用：

```text
ArchTag         = Sm80
OperatorClass   = OpClassTensorOp
InstructionShape = 16x8x16
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
```

## 运行全部 Decode 用例

运行 Qwen2.5-7B：

```bash
./run_gemm.sh --model 7b
```

默认上下文长度为：

```text
128, 256, 512, 1024, 2048
```

每个长度包含9个算子，共45个用例。每个用例都会比较：

- Linear
- Attention QK^T
- Attention AV Stream-K

## 常用运行参数

只输出将要运行的命令：

```bash
./run_gemm.sh --model 7b --dry-run
```

指定上下文长度：

```bash
./run_gemm.sh --model 7b --lengths 128,512,2048
```

指定迭代次数：

```bash
./run_gemm.sh --model 7b --iterations 100
```

当前 kernel 的 Attention 路径针对 `M=28`，因此默认用于 Qwen2.5-7B。可覆盖其他模型参数，但需要保证 `batch × heads = 28`：

```bash
./run_gemm.sh \
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
  total: 45
  passed: 45
  failed: 0
```

## 原生与自研环境结果对比

仓库提供：

```text
collect_gemm_results.py           日志解析和结果合并脚本
gemm_performance_comparison.xlsx  Excel 对比模板
```

先在原生环境保存完整日志：

```bash
./run_gemm.sh --model 7b | tee native.log
python3 collect_gemm_results.py \
  --environment native \
  --log native.log \
  --output gemm_comparison.csv
```

再在自研环境保存日志，并回填同一个 CSV：

```bash
./run_gemm.sh --model 7b | tee custom.log
python3 collect_gemm_results.py \
  --environment custom \
  --log custom.log \
  --output gemm_comparison.csv
```

两个环境也可以只负责生成日志，然后把 `native.log` 和 `custom.log` 复制到同一台机器上执行上述两个合并命令。

生成的 `gemm_comparison.csv` 包含：

- 上下文长度和算子名称
- `M/N/K`
- 原生最佳配置与 GFLOPS
- 自研最佳配置与 GFLOPS
- 自研相对原生的加速比

Excel 模板预置了 Qwen2.5-7B 的45个 Decode 用例。可将 CSV 中对应列的数据粘贴到模板的原生和自研结果列，加速比会自动计算并高亮。

## 注意事项

- 为了与 cuBLAS 的 ColumnMajor 物理布局比较，kernel 按 M 选择布局：
  - `M=1`：A/C 的 RowMajor 与 ColumnMajor 物理存储等价，使用 A/B/C alignment `8/8/8`。
  - `M=28`：A/B/C 均使用 ColumnMajor，A/C leading dimension 为28，使用 alignment `4/8/4`。
- 当前可执行程序接受 `M=1` 或 `M=28`；其他 M 会被参数校验拒绝。
- LM Head 和部分 MLP 用例会分配较大的矩阵，需要足够的主机内存和 GPU 显存。
- `run_gemm.sh` 默认运行当前 CUTLASS 根目录下的 `examples/gemm/gemm`。
