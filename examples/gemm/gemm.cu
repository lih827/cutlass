/***************************************************************************************************
 * TensorOp GEMM configuration comparison example.
 *
 * Computes D = alpha * A * B + beta * C with half precision inputs,
 * half precision accumulation, and half precision output.
 **************************************************************************************************/

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

namespace {

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag = cutlass::arch::Sm80;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr int kAlignmentA = 8;
constexpr int kAlignmentB = 8;
constexpr int kAlignmentC = 8;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, kAlignmentC, ElementAccumulator, ElementCompute>;

template <typename ThreadblockShape, typename WarpShape,
          typename ThreadblockSwizzle, int kStages>
using GemmConfiguration = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    ElementAccumulator, OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp, ThreadblockSwizzle, kStages, kAlignmentA, kAlignmentB>;

// Candidate 1: skinny-M GEMMs with abundant parallelism in N.
using LinearGemm = GemmConfiguration<
    cutlass::gemm::GemmShape<32, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

// Candidate 2: short-K attention QK^T GEMMs.
using AttentionQKGemm = GemmConfiguration<
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

// Candidate 3: long-K attention AV and other low-tile-count GEMMs.
using AttentionAVStreamKGemm = GemmConfiguration<
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4>;

using ReferenceGemm = cutlass::reference::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    ElementAccumulator, ElementCompute>;

template <typename T>
struct ConfigName {
  static char const *value() { return "unknown"; }
};

template <>
struct ConfigName<cutlass::half_t> {
  static char const *value() { return "half"; }
};

template <>
struct ConfigName<cutlass::layout::RowMajor> {
  static char const *value() { return "row-major"; }
};

template <>
struct ConfigName<cutlass::layout::ColumnMajor> {
  static char const *value() { return "column-major"; }
};

template <>
struct ConfigName<cutlass::arch::OpClassTensorOp> {
  static char const *value() { return "OpClassTensorOp"; }
};

template <>
struct ConfigName<cutlass::arch::Sm80> {
  static char const *value() { return "Sm80"; }
};

template <>
struct ConfigName<cutlass::gemm::threadblock::ThreadblockSwizzleStreamK> {
  static char const *value() { return "Stream-K"; }
};

template <int kSwizzleSize>
struct ConfigName<
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<kSwizzleSize>> {
  static char const *value() { return "Identity"; }
};

struct Options {
  bool help = false;
  int m = 1024;
  int n = 1024;
  int k = 1024;
  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 20;

  void parse(int argc, char const **argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  bool valid() const {
    return m > 0 && n > 0 && k > 0 && iterations > 0 &&
           (k % kAlignmentA == 0) &&
           (k % kAlignmentB == 0) &&
           (n % kAlignmentC == 0);
  }

  void print_usage(char const *program) const {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  --m=<int>            GEMM M dimension (default: 1024)\n"
        << "  --n=<int>            GEMM N dimension (default: 1024)\n"
        << "  --k=<int>            GEMM K dimension (default: 1024)\n"
        << "  --alpha=<float>      Epilogue alpha (default: 1.0)\n"
        << "  --beta=<float>       Epilogue beta (default: 0.0)\n"
        << "  --iterations=<int>   Timed iterations (default: 20)\n"
        << "  --help               Show this message\n";
  }
};

struct Tensors {
  cutlass::HostTensor<ElementA, LayoutA> a;
  cutlass::HostTensor<ElementB, LayoutB> b;
  cutlass::HostTensor<ElementC, LayoutC> c;
  cutlass::HostTensor<ElementC, LayoutC> d;
  cutlass::HostTensor<ElementC, LayoutC> reference;
};

struct Result {
  cutlass::Status status = cutlass::Status::kSuccess;
  double avg_time_ms = 0.0;
  double gflops = 0.0;
  bool passed = false;
};

#define CUDA_RETURN_IF_ERROR(expr)                                                       \
  do {                                                                                   \
    cudaError_t error__ = (expr);                                                        \
    if (error__ != cudaSuccess) {                                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__) << "\n";              \
      return false;                                                                      \
    }                                                                                    \
  } while (false)

bool initialize_tensors(Options const &options, Tensors &tensors) {
  tensors.a.resize({options.m, options.k});
  tensors.b.resize({options.k, options.n});
  tensors.c.resize({options.m, options.n});
  tensors.d.resize({options.m, options.n});
  tensors.reference.resize({options.m, options.n});

  cutlass::reference::host::TensorFillRandomUniform(
      tensors.a.host_view(), 2026, ElementA(2), ElementA(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensors.b.host_view(), 2027, ElementB(2), ElementB(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensors.c.host_view(), 2028, ElementC(2), ElementC(-2), 0);
  cutlass::reference::host::TensorFill(tensors.d.host_view(), ElementC(0));
  cutlass::reference::host::TensorFill(tensors.reference.host_view(), ElementC(0));

  tensors.a.sync_device();
  tensors.b.sync_device();
  tensors.c.sync_device();
  tensors.d.sync_device();
  tensors.reference.sync_device();
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return true;
}

bool compute_reference(Options const &options, Tensors &tensors) {
  ReferenceGemm reference_gemm;
  reference_gemm(
      {options.m, options.n, options.k},
      ElementCompute(options.alpha), tensors.a.device_ref(), tensors.b.device_ref(),
      ElementCompute(options.beta), tensors.c.device_ref(), tensors.reference.device_ref());

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  tensors.reference.sync_host();
  return true;
}

template <typename Gemm, typename Swizzle = typename Gemm::ThreadblockSwizzle>
struct ArgumentFactory {
  static typename Gemm::Arguments make(Options const &options, Tensors &tensors) {
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k},
        1,
        {ElementCompute(options.alpha), ElementCompute(options.beta)},
        tensors.a.device_data(), tensors.b.device_data(),
        tensors.c.device_data(), tensors.d.device_data(),
        int64_t(options.m) * options.k,
        int64_t(options.k) * options.n,
        int64_t(options.m) * options.n,
        int64_t(options.m) * options.n,
        tensors.a.layout().stride(0), tensors.b.layout().stride(0),
        tensors.c.layout().stride(0), tensors.d.layout().stride(0));
  }
};

template <typename Gemm>
struct ArgumentFactory<
    Gemm, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK> {
  static typename Gemm::Arguments make(Options const &options, Tensors &tensors) {
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k},
        1,
        {ElementCompute(options.alpha), ElementCompute(options.beta)},
        tensors.a.device_data(), tensors.b.device_data(),
        tensors.c.device_data(), tensors.d.device_data(),
        int64_t(options.m) * options.k,
        int64_t(options.k) * options.n,
        int64_t(options.m) * options.n,
        int64_t(options.m) * options.n,
        tensors.a.layout().stride(0), tensors.b.layout().stride(0),
        tensors.c.layout().stride(0), tensors.d.layout().stride(0),
        -1);
  }
};

template <typename Gemm>
Result run_tensorop_gemm(Options const &options, Tensors &tensors) {
  Result result;
  Gemm gemm;
  typename Gemm::Arguments arguments =
      ArgumentFactory<Gemm>::make(options, tensors);

  result.status = Gemm::can_implement(arguments);
  if (result.status != cutlass::Status::kSuccess) {
    return result;
  }

  cutlass::device_memory::allocation<uint8_t> workspace(
      Gemm::get_workspace_size(arguments));
  result.status = gemm.initialize(arguments, workspace.get());
  if (result.status != cutlass::Status::kSuccess) {
    return result;
  }

  result.status = gemm();  // Warm-up.
  if (result.status != cutlass::Status::kSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    return result;
  }

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  if (cudaEventCreate(&start_event) != cudaSuccess ||
      cudaEventCreate(&stop_event) != cudaSuccess) {
    result.status = cutlass::Status::kErrorInternal;
    return result;
  }

  cudaEventRecord(start_event);
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    result.status = gemm();
    if (result.status != cutlass::Status::kSuccess) {
      break;
    }
  }
  cudaEventRecord(stop_event);
  cudaEventSynchronize(stop_event);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  if (result.status != cutlass::Status::kSuccess) {
    return result;
  }

  result.avg_time_ms = double(elapsed_ms) / options.iterations;
  result.gflops = 2.0 * double(options.m) * double(options.n) * double(options.k) /
                  (result.avg_time_ms * 1.0e6);

  tensors.d.sync_host();
  result.passed = cutlass::reference::host::TensorEquals(
      tensors.d.host_view(), tensors.reference.host_view());
  return result;
}

template <typename Gemm>
void print_configuration(char const *configuration_name, Options const &options) {
  std::cout
      << "\nGEMM configuration: " << configuration_name << "\n"
      << "  Problem: " << options.m << " x " << options.n << " x " << options.k << "\n"
      << "  alpha / beta: " << options.alpha << " / " << options.beta << "\n"
      << "  iterations: " << options.iterations << "\n"
      << "  A: " << ConfigName<ElementA>::value() << ", "
      << ConfigName<LayoutA>::value() << "\n"
      << "  B: " << ConfigName<ElementB>::value() << ", "
      << ConfigName<LayoutB>::value() << "\n"
      << "  C: " << ConfigName<ElementC>::value() << ", "
      << ConfigName<LayoutC>::value() << "\n"
      << "  accumulator: " << ConfigName<ElementAccumulator>::value() << "\n"
      << "  operator class / arch: "
      << ConfigName<typename Gemm::OperatorClass>::value() << " / "
      << ConfigName<typename Gemm::ArchTag>::value() << "\n"
      << "  threadblock: "
      << Gemm::ThreadblockShape::kM << "x"
      << Gemm::ThreadblockShape::kN << "x"
      << Gemm::ThreadblockShape::kK << "\n"
      << "  warp: "
      << Gemm::WarpShape::kM << "x"
      << Gemm::WarpShape::kN << "x"
      << Gemm::WarpShape::kK << "\n"
      << "  instruction: "
      << Gemm::InstructionShape::kM << "x"
      << Gemm::InstructionShape::kN << "x"
      << Gemm::InstructionShape::kK << "\n"
      << "  alignment A/B/C: "
      << Gemm::kAlignmentA << " / "
      << Gemm::kAlignmentB << " / "
      << Gemm::kAlignmentC << "\n"
      << "  stages: " << Gemm::kStages << "\n"
      << "  threadblock swizzle: "
      << ConfigName<typename Gemm::ThreadblockSwizzle>::value() << "\n";
}

void print_result(char const *configuration_name, Result const &result) {
  std::cout << std::fixed << std::setprecision(4)
            << "Results: " << configuration_name << "\n"
            << "  Status: "
            << (result.status == cutlass::Status::kSuccess
                    ? (result.passed ? "Passed" : "Failed verification")
                    : cutlassGetStatusString(result.status))
            << "\n"
            << "  avg_time: " << result.avg_time_ms << " ms\n"
            << "  gflops: " << result.gflops << "\n";
}

}  // namespace

int main(int argc, char const **argv) {
  // Step 1: require CUDA Toolkit 9.0 or newer.
#if !defined(__CUDACC_VER_MAJOR__) || (__CUDACC_VER_MAJOR__ < 9)
  std::cerr << "This example requires CUDA Toolkit 9.0 or newer.\n";
  return EXIT_FAILURE;
#endif

  // Step 2: parse arguments.
  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  if (!options.valid()) {
    std::cerr
        << "Invalid problem: dimensions and iterations must be positive; "
        << "k must satisfy A/B alignment and n must satisfy C/D alignment.\n";
    options.print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  // Step 3: select the device and check it immediately before GEMM setup/execution.
  int device_id = 0;
  cudaDeviceProp device_properties{};
  cudaError_t cuda_status = cudaSetDevice(device_id);
  if (cuda_status != cudaSuccess) {
    std::cerr << "cudaSetDevice(" << device_id << ") failed: "
              << cudaGetErrorString(cuda_status) << "\n";
    return EXIT_FAILURE;
  }
  cuda_status = cudaGetDeviceProperties(&device_properties, device_id);
  if (cuda_status != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed: "
              << cudaGetErrorString(cuda_status) << "\n";
    return EXIT_FAILURE;
  }
  if (device_properties.major * 10 + device_properties.minor < 80) {
    std::cerr << "This example requires SM80 or newer; detected SM"
              << device_properties.major << device_properties.minor << ".\n";
    return EXIT_FAILURE;
  }

  // Step 4: initialize tensors and compute the reference result once.
  Tensors tensors;
  if (!initialize_tensors(options, tensors) || !compute_reference(options, tensors)) {
    return EXIT_FAILURE;
  }

  // Step 5: run all TensorOp candidates against the same data and reference result.
  print_configuration<LinearGemm>("Linear", options);
  Result linear = run_tensorop_gemm<LinearGemm>(options, tensors);
  print_result("Linear", linear);

  print_configuration<AttentionQKGemm>("Attention QK^T", options);
  Result attention_qk = run_tensorop_gemm<AttentionQKGemm>(options, tensors);
  print_result("Attention QK^T", attention_qk);

  print_configuration<AttentionAVStreamKGemm>("Attention AV Stream-K", options);
  Result attention_av = run_tensorop_gemm<AttentionAVStreamKGemm>(options, tensors);
  print_result("Attention AV Stream-K", attention_av);

  // Step 6: report the fastest candidate that also passed verification.
  char const *best_configuration_name = nullptr;
  Result const *best_result = nullptr;
  if (linear.status == cutlass::Status::kSuccess && linear.passed) {
    best_configuration_name = "Linear";
    best_result = &linear;
  }
  if (attention_qk.status == cutlass::Status::kSuccess && attention_qk.passed &&
      (!best_result || attention_qk.avg_time_ms < best_result->avg_time_ms)) {
    best_configuration_name = "Attention QK^T";
    best_result = &attention_qk;
  }
  if (attention_av.status == cutlass::Status::kSuccess && attention_av.passed &&
      (!best_result || attention_av.avg_time_ms < best_result->avg_time_ms)) {
    best_configuration_name = "Attention AV Stream-K";
    best_result = &attention_av;
  }

  if (!best_result) {
    std::cerr << "\nNo candidate completed successfully and passed verification.\n";
    return EXIT_FAILURE;
  }

  std::cout << "\nBest configuration: " << best_configuration_name << "\n"
            << "  avg_time: " << best_result->avg_time_ms << " ms\n"
            << "  gflops: " << best_result->gflops << "\n";
  return EXIT_SUCCESS;
}
