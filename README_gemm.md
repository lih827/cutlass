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

脚本先生成算子形状，再严格按 `M/N/K` 去重：与 `L` 无关的 Linear GEMM 只运行一次；只有 `Attention QK^T` 和 `Attention AV` 会随 `L` 生成新形状。相同形状的来源算子会合并记录，例如 `Q / Attention Out` 和 `K / V`。

默认 Qwen2.5-7B 最终运行 14 个唯一 GEMM，而不是原始展开后的 45 次调用。每个唯一 GEMM 都会比较：

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

## CUTLASS 与 HGEMM 结果对比

仓库提供：

```text
collect_gemm_results.py           CUTLASS 日志解析和结果回填脚本
gemm_performance_comparison.xlsx  Excel 对比模板
```

先运行 CUTLASS 用例并保存完整日志。脚本按 `M/N/K` 定位 Excel 模板中的行，只更新 CUTLASS 最佳配置和 GFLOPS；HGEMM 数据、加速比公式和格式均会保留。

直接更新原始 XLSX：

```bash
./run_gemm.sh --model 7b | tee cutlass.log
python3 collect_gemm_results.py \
  --log cutlass.log \
  --workbook gemm_performance_comparison.xlsx
```

如需保留空白模板，可输出到新 XLSX：

```bash
python3 collect_gemm_results.py \
  --log cutlass.log \
  --workbook gemm_performance_comparison.xlsx \
  --output gemm_performance_comparison_filled.xlsx
```

`--output` 必须使用 `.xlsx` 后缀；省略时原地更新 `--workbook`。如果日志中的 `M/N/K` 不存在于模板中，脚本会报错且不会写入不完整结果。

HGEMM（自研）和 HGEMM（CUDA）数据来自其他测试程序，输出格式与 `gemm.cu` 不同，因此不由脚本解析。请按相同的 `M/N/K` 分别人工筛选两类 HGEMM 最佳 GFLOPS，再填写 Excel 模板中的人工录入列。

更新后的 `gemm_performance_comparison.xlsx` 包含：

- 相关上下文长度和来源算子名称（相同 `M/N/K` 合并）
- `M/N/K`
- CUTLASS 最佳配置与 GFLOPS（脚本自动更新）
- HGEMM（自研）GFLOPS（人工筛选，已有内容会被保留）
- HGEMM（CUDA）GFLOPS（人工筛选，已有内容会被保留）
- 两类 HGEMM 分别相对 CUTLASS 的加速比和人工筛选备注

Excel 模板预置了 Qwen2.5-7B 的 14 个唯一 Decode GEMM。蓝色列由脚本直接回填 CUTLASS 数据；黄色和橙色列分别用于人工录入 HGEMM（自研）及 HGEMM（CUDA）结果；绿色列自动计算两者相对 CUTLASS 的加速比。

## 注意事项

- 为了与 cuBLAS 的 ColumnMajor 物理布局比较，kernel 按 M 选择布局：
  - `M=1`：A/C 的 RowMajor 与 ColumnMajor 物理存储等价，使用 A/B/C alignment `8/8/8`。
  - `M>1`：A/B/C 均使用 ColumnMajor；A/C 根据 M 选择能整除 M 的最大 alignment `8/4/2`，B使用alignment 8。
- 例如 `M=28` 使用 A/B/C alignment `4/8/4`，`M=14` 使用 `2/8/2`，`M=40` 使用 `8/8/8`。
- SM80多级流水线的 `cp.async` 最小支持4字节访问，因此 `M>1` 必须为偶数；当前全部 Qwen2.5 preset 均满足该条件。
- LM Head 和部分 MLP 用例会分配较大的矩阵，需要足够的主机内存和 GPU 显存。
- `run_gemm.sh` 默认运行当前 CUTLASS 根目录下的 `examples/gemm/gemm`。
