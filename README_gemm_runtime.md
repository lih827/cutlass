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
├── estimate_qwen_gemm.py
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
128, 129, 130, 131, 133, 137, 256, 257, 512, 513, 1024, 1025, 2048, 2049
```

只运行 Prefill：

```bash
bash run_gemm.sh --model 7b --stage prefill
```

Prefill 默认序列长度：

```text
128, 256, 512, 1024, 2048, 129, 130, 132, 136
```

默认 Decode 列表同时覆盖常用 L 边界和各默认 Prefill `S` 的首次 Decode `L=S+1`。Qwen2.5-7B 默认包含 32 个唯一 Decode GEMM 和 54 个唯一 Prefill GEMM；跨阶段全局去重后共 85 个。相同 `M/N/K` 的来源算子会合并，只执行一次；`Q/Attention Out`、`K/V`、`MLP Up/MLP Gate` 在整模型估算时仍分别计两次调用。Prefill LM Head 只计算最后位置，固定为 `M=1`，并与 Decode LM Head 共用同一 shape。

本测试以 Batch=1、无融合、普通 GEMM Attention、FP16 A/B/C 与 FP16 累加来近似 BF16 推理的 GEMM-only 下界，不包含 softmax、mask、RMSNorm、RoPE、激活、残差、KV 管理和 kernel 间隙。

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

ColumnMajor 用例中 A/C 按 M、B 按 K 独立选择最大合法 alignment 8/4/2/1，不做 padding。例如 M 只能对齐到4而 K 能对齐到8时使用 `4/8/4`。A 或 B 任一侧为 alignment 1 时使用同步 Stages 2，否则大 M 使用 Stages 3。默认 Prefill S=129/130/132 覆盖 alignment 1/2/4。

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

## 汇总 GEMM-only 模型估算

用目标长度的运行日志累加各 GEMM 最佳 `avg_time`：

```bash
bash run_gemm.sh --model 7b --stage prefill --lengths 128 | tee cutlass_prefill.log
python3 estimate_qwen_gemm.py --log cutlass_prefill.log --model 7b --stage prefill --length 128

bash run_gemm.sh --model 7b --stage decode --lengths 129 | tee cutlass_decode.log
python3 estimate_qwen_gemm.py --log cutlass_decode.log --model 7b --stage decode --length 129
```

Prompt 长度为 `P` 时，Prefill 使用 `S=P`，下一次 Decode forward 使用 `L=P+1`。脚本按 Qwen2.5 模型层数汇总，对 Q/Out、K/V、Up/Gate 各计两次，LM Head 只计一次。整模型延迟应累加 `avg_time × 调用次数`，不能平均各用例 GFLOPS。

### 一个 forward 的估算范围

- Prefill `--length S` 表示一次处理全部 S 个 prompt token 的 forward，并由最后位置 LM Head 产生第一个输出 token。
- Decode `--length L` 表示一次读取长度为 L 的 KV Cache、产生一个新 token 的 forward。

单个 forward 的 GEMM-only 时间为：

```text
num_layers * (
    2*T(Q/Out) + 2*T(K/V) + T(QK^T) + T(PV)
  + 2*T(Up/Gate) + T(Down)
) + T(LM_Head_M1)
```

所有 `T(shape)` 均读取日志中的实测最佳 `avg_time`。当前脚本严格要求日志包含全部所需 shape；缺失时直接报错，不使用理论峰值、FLOPs 比例、插值或外推。

若 `S=128、G=128`，并且 G 包含 Prefill 产生的第一个 token，则完整生成过程为 Prefill `S=128` 加 Decode `L=129...255` 共127次：

```text
T_total = T_prefill(128) + sum(T_decode(L), L=129...255)
```

当前脚本一次只估算一个 forward，尚不自动循环和累加整个 `S/G` 区间。

### 基于全量实测数据估算 S/G

以下示例估算 Qwen2.5-7B 的 `S=128、G=128`。G 包含 Prefill 产生的第一个 token，所以 Decode 共127次，KV长度为 `L=129...255`：

```bash
MODEL=7b
S=128
G=128
FIRST_L=$((S + 1))
LAST_L=$((S + G - 1))
DECODE_LENGTHS=$(seq -s, "$FIRST_L" "$LAST_L")

bash run_gemm.sh --model "$MODEL" --stage prefill --lengths "$S" \
  | tee "prefill_s${S}.log"

bash run_gemm.sh --model "$MODEL" --stage decode --lengths "$DECODE_LENGTHS" \
  | tee "decode_s${S}_g${G}.log"
```

累加每个 forward 的实测最佳时间：

```bash
PREFILL_MS=$(
  python3 estimate_qwen_gemm.py --log "prefill_s${S}.log" \
    --model "$MODEL" --stage prefill --length "$S" \
  | awk '/GEMM-only lower-bound latency:/ {print $4}'
)

DECODE_MS=$(
  for L in $(seq "$FIRST_L" "$LAST_L"); do
    python3 estimate_qwen_gemm.py --log "decode_s${S}_g${G}.log" \
      --model "$MODEL" --stage decode --length "$L"
  done \
  | awk '/GEMM-only lower-bound latency:/ {sum += $4} END {printf "%.6f", sum}'
)

awk -v prefill="$PREFILL_MS" -v decode="$DECODE_MS" \
  'BEGIN {printf "Total GEMM-only: %.6f ms\n", prefill + decode}'
```

该方法要求日志覆盖每一个 L，不进行插值。`G=1` 时只运行 Prefill；若 G 表示 Prefill 之后额外生成的 token 数，则 Decode 范围改为 `L=S+1...S+G`。

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
- LM Head 固定使用 `M=1`，只计算最后位置 logits；部分 MLP 用例仍需要较大的主机内存与 GPU显存。
- 更换 GPU、CUDA版本或 `.inc` 后必须重新执行 build。
