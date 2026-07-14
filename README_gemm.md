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

运行其他模型：

```bash
./run_gemm.sh --model 0.5b
./run_gemm.sh --model 1.5b
./run_gemm.sh --model 3b
./run_gemm.sh --model 14b
./run_gemm.sh --model 32b
./run_gemm.sh --model 72b
```

覆盖模型参数：

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

## 注意事项

- A、B、C/D 的 alignment 均为8，因此 Decode shape 需要满足 `K % 8 == 0` 和 `N % 8 == 0`。
- LM Head 和部分 MLP 用例会分配较大的矩阵，需要足够的主机内存和 GPU 显存。
- `run_gemm.sh` 默认运行当前 CUTLASS 根目录下的 `examples/gemm/gemm`。
