/***************************************************************************************************
 * TensorOp GEMM configuration comparison example.
 *
 * Computes D = alpha * A * B + beta * C with half precision inputs,
 * half precision accumulation, and half precision output.
 **************************************************************************************************/

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
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

#if defined(GEMM_CONCISE_LOG) && GEMM_CONCISE_LOG
constexpr bool kConciseLog = true;
#else
constexpr bool kConciseLog = false;
#endif

#if defined(GEMM_USE_CHRONO) && GEMM_USE_CHRONO
constexpr bool kUseChronoTimer = true;
#else
constexpr bool kUseChronoTimer = false;
#endif

#if defined(GEMM_OPTIMAL_ONLY) && GEMM_OPTIMAL_ONLY
constexpr bool kOptimalOnly = true;
#else
constexpr bool kOptimalOnly = false;
#endif

#if defined(GEMM_SKIP_VERIFICATION) && GEMM_SKIP_VERIFICATION
constexpr bool kVerifyResults = false;
#else
constexpr bool kVerifyResults = true;
#endif

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;

using LayoutAM1 = cutlass::layout::RowMajor;
using LayoutBM1 = cutlass::layout::ColumnMajor;
using LayoutCM1 = cutlass::layout::RowMajor;

using LayoutAAttention = cutlass::layout::ColumnMajor;
using LayoutBAttention = cutlass::layout::ColumnMajor;
using LayoutCAttention = cutlass::layout::ColumnMajor;

using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag = cutlass::arch::Sm80;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr int kAlignmentAM1 = 8;
constexpr int kAlignmentBM1 = 8;
constexpr int kAlignmentCM1 = 8;

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC,
          typename ThreadblockShape, typename WarpShape,
          typename ThreadblockSwizzle, int kStages>
using GemmConfiguration = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    ElementAccumulator, OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, kAlignmentC, ElementAccumulator, ElementCompute>,
    ThreadblockSwizzle, kStages, kAlignmentA, kAlignmentB>;

// Candidate 1: skinny-M GEMMs with abundant parallelism in N.
template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC>
using LinearGemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<32, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

// Candidate 2: short-K attention QK^T GEMMs.
template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC>
using AttentionQKGemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

// Candidate 3: long-K attention PV and other low-tile-count GEMMs.
template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC>
using AttentionPVStreamKGemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4>;

// Prefill candidates: large M/N grids already provide abundant parallelism,
// so use larger CTA tiles and the identity swizzle.
template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
using LargeM128x128Gemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
using LargeM128x256Gemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
using LargeM256x128Gemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
using LargeM64x128Gemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
using LargeM128x64Gemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
using LargeM64x256Gemm = GemmConfiguration<
    LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

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
  int m = 1;
  int n = 1024;
  int k = 1024;
  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 20;
  int split_k_slices = 1;  // Set by generated cuBLASLt-derived dispatch.

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
    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0) {
      return false;
    }
    if (m == 1) {
      return (k % kAlignmentAM1 == 0) &&
             (k % kAlignmentBM1 == 0) &&
             (n % kAlignmentCM1 == 0);
    }
    return true;  // Non-aligned M/K use the Alignment=1 synchronous fallback.
  }

  void print_usage(char const *program) const {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  --m=<int>            GEMM M dimension (default: 1)\n"
        << "  --n=<int>            GEMM N dimension (default: 1024)\n"
        << "  --k=<int>            GEMM K dimension (default: 1024)\n"
        << "  --alpha=<float>      Epilogue alpha (default: 1.0)\n"
        << "  --beta=<float>       Epilogue beta (default: 0.0)\n"
        << "  --iterations=<int>   Timed iterations (default: 20)\n"
        << "  --help               Show this message\n";
  }
};

template <typename LayoutA_, typename LayoutB_, typename LayoutC_>
struct Tensors {
  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;
  using LayoutC = LayoutC_;

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

struct GeneratedCandidateResult {
  bool available = false;
  std::string name;
  Result result;
  int threadblock_m = 0, threadblock_n = 0, threadblock_k = 0;
  int warp_m = 0, warp_n = 0, warp_k = 0;
  int stages = 0, alignment_a = 0, alignment_b = 0, alignment_c = 0;
  int split_k_slices = 1;
};

GeneratedCandidateResult generated_candidate_result;

#define CUDA_RETURN_IF_ERROR(expr)                                                       \
  do {                                                                                   \
    cudaError_t error__ = (expr);                                                        \
    if (error__ != cudaSuccess) {                                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__) << "\n";              \
      return false;                                                                      \
    }                                                                                    \
  } while (false)

template <typename TensorSet>
bool initialize_tensors(Options const &options, TensorSet &tensors) {
  tensors.a.resize({options.m, options.k});
  tensors.b.resize({options.k, options.n});
  tensors.c.resize({options.m, options.n});
  tensors.d.resize({options.m, options.n});
  if constexpr (kVerifyResults) {
    tensors.reference.resize({options.m, options.n});
  }

  if constexpr (!kVerifyResults) {
    CUDA_RETURN_IF_ERROR(cudaMemset(
        tensors.a.device_data(), 0,
        size_t(options.m) * options.k * sizeof(ElementA)));
    CUDA_RETURN_IF_ERROR(cudaMemset(
        tensors.b.device_data(), 0,
        size_t(options.k) * options.n * sizeof(ElementB)));
    CUDA_RETURN_IF_ERROR(cudaMemset(
        tensors.c.device_data(), 0,
        size_t(options.m) * options.n * sizeof(ElementC)));
    return true;
  }

  cutlass::reference::host::TensorFillRandomUniform(
      tensors.a.host_view(), 2026, ElementA(2), ElementA(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensors.b.host_view(), 2027, ElementB(2), ElementB(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensors.c.host_view(), 2028, ElementC(2), ElementC(-2), 0);
  if constexpr (kVerifyResults) {
    cutlass::reference::host::TensorFill(tensors.d.host_view(), ElementC(0));
    cutlass::reference::host::TensorFill(tensors.reference.host_view(), ElementC(0));
  }

  tensors.a.sync_device();
  tensors.b.sync_device();
  tensors.c.sync_device();
  if constexpr (kVerifyResults) {
    tensors.d.sync_device();
    tensors.reference.sync_device();
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return true;
}

template <typename TensorSet>
bool compute_reference(Options const &options, TensorSet &tensors) {
  using ReferenceGemm = cutlass::reference::device::Gemm<
      ElementA, typename TensorSet::LayoutA,
      ElementB, typename TensorSet::LayoutB,
      ElementC, typename TensorSet::LayoutC,
      ElementAccumulator, ElementCompute>;
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
  template <typename TensorSet>
  static typename Gemm::Arguments make(Options const &options, TensorSet &tensors) {
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k},
        options.split_k_slices,
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
  template <typename TensorSet>
  static typename Gemm::Arguments make(Options const &options, TensorSet &tensors) {
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k},
        options.split_k_slices,
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

template <typename Gemm, typename TensorSet>
Result run_tensorop_gemm(Options const &options, TensorSet &tensors) {
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

  float elapsed_ms = 0.0f;
  if constexpr (kUseChronoTimer) {
    if (cudaDeviceSynchronize() != cudaSuccess) {
      result.status = cutlass::Status::kErrorInternal;
      return result;
    }
    auto const start_time = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result.status = gemm();
      if (result.status != cutlass::Status::kSuccess) {
        break;
      }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      result.status = cutlass::Status::kErrorInternal;
      return result;
    }
    auto const stop_time = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration<float, std::milli>(
                     stop_time - start_time).count();
  } else {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (cudaEventCreate(&start_event) != cudaSuccess ||
        cudaEventCreate(&stop_event) != cudaSuccess) {
      if (start_event) cudaEventDestroy(start_event);
      if (stop_event) cudaEventDestroy(stop_event);
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
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  if (result.status != cutlass::Status::kSuccess) {
    return result;
  }

  result.avg_time_ms = double(elapsed_ms) / options.iterations;
  result.gflops = 2.0 * double(options.m) * double(options.n) * double(options.k) /
                  (result.avg_time_ms * 1.0e6);

  if constexpr (kVerifyResults) {
    tensors.d.sync_host();
    result.passed = cutlass::reference::host::TensorEquals(
        tensors.d.host_view(), tensors.reference.host_view());
  } else {
    result.passed = true;
  }
  return result;
}

template <typename Gemm>
void print_configuration(char const *configuration_name, Options const &options) {
  std::cout
      << "\nGEMM configuration: " << configuration_name << "\n"
      << "  Problem: " << options.m << " x " << options.n << " x " << options.k << "\n"
      << "  alpha / beta: " << options.alpha << " / " << options.beta << "\n"
      << "  iterations: " << options.iterations << "\n"
      << "  verification: " << (kVerifyResults ? "enabled" : "disabled") << "\n"
      << "  timer: " << (kUseChronoTimer ? "chrono" : "cuda-event") << "\n"
      << "  split-K slices: " << options.split_k_slices << "\n"
      << "  A: " << ConfigName<ElementA>::value() << ", "
      << ConfigName<typename Gemm::LayoutA>::value() << "\n"
      << "  B: " << ConfigName<ElementB>::value() << ", "
      << ConfigName<typename Gemm::LayoutB>::value() << "\n"
      << "  C: " << ConfigName<ElementC>::value() << ", "
      << ConfigName<typename Gemm::LayoutC>::value() << "\n"
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

template <typename Gemm>
std::string make_configuration_name(char const *family) {
  std::ostringstream name;
  name << family
       << " TB" << Gemm::ThreadblockShape::kM
       << "x" << Gemm::ThreadblockShape::kN
       << "x" << Gemm::ThreadblockShape::kK
       << "_W" << Gemm::WarpShape::kM
       << "x" << Gemm::WarpShape::kN
       << "x" << Gemm::WarpShape::kK
       << "_S" << Gemm::kStages;
  return name.str();
}

void print_result(char const *configuration_name, Result const &result) {
  std::cout << std::fixed << std::setprecision(4)
            << "Results: " << configuration_name << "\n"
            << "  Status: "
            << (result.status == cutlass::Status::kSuccess
                    ? (!kVerifyResults ? "Not verified"
                       : (result.passed ? "Passed" : "Failed verification"))
                    : cutlassGetStatusString(result.status))
            << "\n"
            << "  avg_time: " << result.avg_time_ms << " ms\n"
            << "  gflops: " << result.gflops << "\n";
}

template <typename Gemm>
void print_cutlass_record(char const *record_type, char const *source,
                          char const *configuration_name,
                          Options const &options, Result const &result) {
  std::cout << std::fixed << std::setprecision(6)
            << record_type
            << " m=" << options.m << " n=" << options.n << " k=" << options.k
            << " source=" << source << " name=" << configuration_name
            << " layout_a=" << (options.m == 1 ? "LayoutAM1" : "LayoutAAttention")
            << " layout_b=" << (options.m == 1 ? "LayoutBM1" : "LayoutBAttention")
            << " layout_c=" << (options.m == 1 ? "LayoutCM1" : "LayoutCAttention")
            << " align_a=" << Gemm::kAlignmentA
            << " align_b=" << Gemm::kAlignmentB
            << " align_c=" << Gemm::kAlignmentC
            << " tb_m=" << Gemm::ThreadblockShape::kM
            << " tb_n=" << Gemm::ThreadblockShape::kN
            << " tb_k=" << Gemm::ThreadblockShape::kK
            << " warp_m=" << Gemm::WarpShape::kM
            << " warp_n=" << Gemm::WarpShape::kN
            << " warp_k=" << Gemm::WarpShape::kK
            << " swizzle=" << ConfigName<typename Gemm::ThreadblockSwizzle>::value()
            << " stages=" << Gemm::kStages
            << " split_k=" << options.split_k_slices
            << " valid=" << ((result.status == cutlass::Status::kSuccess && result.passed) ? 1 : 0)
            << " avg_time_ms=" << result.avg_time_ms
            << " gflops=" << result.gflops << "\n";
}

void print_generated_record(char const *record_type, Options const &options,
                            Result const &result) {
  auto const &g = generated_candidate_result;
  std::cout << std::fixed << std::setprecision(6)
            << record_type
            << " m=" << options.m << " n=" << options.n << " k=" << options.k
            << " source=cublaslt-derived name=" << g.name
            << " layout_a=" << (options.m == 1 ? "LayoutAM1" : "LayoutAAttention")
            << " layout_b=" << (options.m == 1 ? "LayoutBM1" : "LayoutBAttention")
            << " layout_c=" << (options.m == 1 ? "LayoutCM1" : "LayoutCAttention")
            << " align_a=" << g.alignment_a << " align_b=" << g.alignment_b
            << " align_c=" << g.alignment_c
            << " tb_m=" << g.threadblock_m << " tb_n=" << g.threadblock_n
            << " tb_k=" << g.threadblock_k
            << " warp_m=" << g.warp_m << " warp_n=" << g.warp_n
            << " warp_k=" << g.warp_k
            << " swizzle=Identity stages=" << g.stages
            << " split_k=" << g.split_k_slices
            << " valid=" << ((result.status == cutlass::Status::kSuccess && result.passed) ? 1 : 0)
            << " avg_time_ms=" << result.avg_time_ms
            << " gflops=" << result.gflops << "\n";
}

void print_generated_configuration(Options const &options) {
  auto const &g = generated_candidate_result;
  std::cout << "\nGEMM configuration: " << g.name << "\n"
            << "  Problem: " << options.m << " x " << options.n << " x " << options.k << "\n"
            << "  alpha / beta: " << options.alpha << " / " << options.beta << "\n"
            << "  iterations: " << options.iterations << "\n"
            << "  verification: " << (kVerifyResults ? "enabled" : "disabled") << "\n"
            << "  timer: " << (kUseChronoTimer ? "chrono" : "cuda-event") << "\n"
            << "  split-K slices: " << g.split_k_slices << "\n"
            << "  A: half, " << (options.m == 1 ? "row-major" : "column-major") << "\n"
            << "  B: half, column-major\n"
            << "  C: half, " << (options.m == 1 ? "row-major" : "column-major") << "\n"
            << "  accumulator: half\n"
            << "  operator class / arch: OpClassTensorOp / Sm80\n"
            << "  threadblock: " << g.threadblock_m << "x" << g.threadblock_n << "x" << g.threadblock_k << "\n"
            << "  warp: " << g.warp_m << "x" << g.warp_n << "x" << g.warp_k << "\n"
            << "  instruction: 16x8x16\n"
            << "  alignment A/B/C: " << g.alignment_a << " / " << g.alignment_b << " / " << g.alignment_c << "\n"
            << "  stages: " << g.stages << "\n"
            << "  threadblock swizzle: Identity\n";
}

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC>
int profile_all_candidates(Options const &options) {
  using TensorSet = Tensors<LayoutA, LayoutB, LayoutC>;
  using Linear = LinearGemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC>;
  using AttentionQK = AttentionQKGemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC>;
  using AttentionPV = AttentionPVStreamKGemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC>;

  std::string const linear_name = make_configuration_name<Linear>("Linear");
  std::string const attention_qk_name =
      make_configuration_name<AttentionQK>("Attention-QK");
  std::string const attention_pv_name =
      make_configuration_name<AttentionPV>("Attention-PV-StreamK");

  TensorSet tensors;
  if (!initialize_tensors(options, tensors) ||
      (kVerifyResults && !compute_reference(options, tensors))) {
    return EXIT_FAILURE;
  }

  if (!kConciseLog) print_configuration<Linear>(linear_name.c_str(), options);
  Result linear = run_tensorop_gemm<Linear>(options, tensors);
  print_cutlass_record<Linear>("CUTLASS_CANDIDATE", "baseline", linear_name.c_str(), options, linear);
  if (!kConciseLog) print_result(linear_name.c_str(), linear);

  if (!kConciseLog) print_configuration<AttentionQK>(attention_qk_name.c_str(), options);
  Result attention_qk = run_tensorop_gemm<AttentionQK>(options, tensors);
  print_cutlass_record<AttentionQK>("CUTLASS_CANDIDATE", "baseline", attention_qk_name.c_str(), options, attention_qk);
  if (!kConciseLog) print_result(attention_qk_name.c_str(), attention_qk);

  if (!kConciseLog) print_configuration<AttentionPV>(attention_pv_name.c_str(), options);
  Result attention_pv = run_tensorop_gemm<AttentionPV>(options, tensors);
  print_cutlass_record<AttentionPV>("CUTLASS_CANDIDATE", "baseline", attention_pv_name.c_str(), options, attention_pv);
  if (!kConciseLog) print_result(attention_pv_name.c_str(), attention_pv);

  char const *best_configuration_name = nullptr;
  Result const *best_result = nullptr;
  bool best_is_generated = false;
  if (generated_candidate_result.available &&
      generated_candidate_result.result.status == cutlass::Status::kSuccess &&
      generated_candidate_result.result.passed) {
    best_configuration_name = generated_candidate_result.name.c_str();
    best_result = &generated_candidate_result.result;
    best_is_generated = true;
  }
  if (linear.status == cutlass::Status::kSuccess && linear.passed) {
    if (!best_result || linear.avg_time_ms < best_result->avg_time_ms) {
      best_configuration_name = linear_name.c_str();
      best_result = &linear;
      best_is_generated = false;
    }
  }
  if (attention_qk.status == cutlass::Status::kSuccess && attention_qk.passed &&
      (!best_result || attention_qk.avg_time_ms < best_result->avg_time_ms)) {
    best_configuration_name = attention_qk_name.c_str();
    best_result = &attention_qk;
    best_is_generated = false;
  }
  if (attention_pv.status == cutlass::Status::kSuccess && attention_pv.passed &&
      (!best_result || attention_pv.avg_time_ms < best_result->avg_time_ms)) {
    best_configuration_name = attention_pv_name.c_str();
    best_result = &attention_pv;
    best_is_generated = false;
  }

  if (!best_result) {
    std::cerr << "\nNo candidate completed successfully and passed verification.\n";
    return EXIT_FAILURE;
  }

  if (kConciseLog) {
    if (best_is_generated) {
      print_generated_configuration(options);
      print_result(best_configuration_name, *best_result);
    } else if (best_result == &linear) {
      print_configuration<Linear>(best_configuration_name, options);
      print_result(best_configuration_name, *best_result);
    } else if (best_result == &attention_qk) {
      print_configuration<AttentionQK>(best_configuration_name, options);
      print_result(best_configuration_name, *best_result);
    } else {
      print_configuration<AttentionPV>(best_configuration_name, options);
      print_result(best_configuration_name, *best_result);
    }
  }

  std::cout << "\nBest configuration: " << best_configuration_name << "\n"
            << "  avg_time: " << best_result->avg_time_ms << " ms\n"
            << "  gflops: " << best_result->gflops << "\n";
  if (best_is_generated) {
    print_generated_record("CUTLASS_BEST", options, *best_result);
  } else if (best_result == &linear) {
    print_cutlass_record<Linear>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  } else if (best_result == &attention_qk) {
    print_cutlass_record<AttentionQK>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  } else {
    print_cutlass_record<AttentionPV>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  }
  return EXIT_SUCCESS;
}

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
int profile_large_m_candidates(Options const &options) {
  using TensorSet = Tensors<LayoutA, LayoutB, LayoutC>;
  using Tile128x128 = LargeM128x128Gemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC, kStages>;
  using Tile128x256 = LargeM128x256Gemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC, kStages>;
  using Tile256x128 = LargeM256x128Gemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC, kStages>;
  using Tile64x128 = LargeM64x128Gemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC, kStages>;
  using Tile128x64 = LargeM128x64Gemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC, kStages>;
  using Tile64x256 = LargeM64x256Gemm<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC, kStages>;

  std::string const tile_128x128_name =
      make_configuration_name<Tile128x128>("Large-M");
  std::string const tile_128x256_name =
      make_configuration_name<Tile128x256>("Large-M");
  std::string const tile_256x128_name =
      make_configuration_name<Tile256x128>("Large-M");
  std::string const tile_64x128_name =
      make_configuration_name<Tile64x128>("Large-M");
  std::string const tile_128x64_name =
      make_configuration_name<Tile128x64>("Large-M");
  std::string const tile_64x256_name =
      make_configuration_name<Tile64x256>("Large-M");

  TensorSet tensors;
  if (!initialize_tensors(options, tensors) ||
      (kVerifyResults && !compute_reference(options, tensors))) {
    return EXIT_FAILURE;
  }

  if (!kConciseLog) print_configuration<Tile128x128>(tile_128x128_name.c_str(), options);
  Result tile_128x128 = run_tensorop_gemm<Tile128x128>(options, tensors);
  print_cutlass_record<Tile128x128>("CUTLASS_CANDIDATE", "baseline", tile_128x128_name.c_str(), options, tile_128x128);
  if (!kConciseLog) print_result(tile_128x128_name.c_str(), tile_128x128);

  if (!kConciseLog) print_configuration<Tile128x256>(tile_128x256_name.c_str(), options);
  Result tile_128x256 = run_tensorop_gemm<Tile128x256>(options, tensors);
  print_cutlass_record<Tile128x256>("CUTLASS_CANDIDATE", "baseline", tile_128x256_name.c_str(), options, tile_128x256);
  if (!kConciseLog) print_result(tile_128x256_name.c_str(), tile_128x256);

  if (!kConciseLog) print_configuration<Tile256x128>(tile_256x128_name.c_str(), options);
  Result tile_256x128 = run_tensorop_gemm<Tile256x128>(options, tensors);
  print_cutlass_record<Tile256x128>("CUTLASS_CANDIDATE", "baseline", tile_256x128_name.c_str(), options, tile_256x128);
  if (!kConciseLog) print_result(tile_256x128_name.c_str(), tile_256x128);

  if (!kConciseLog) print_configuration<Tile64x128>(tile_64x128_name.c_str(), options);
  Result tile_64x128 = run_tensorop_gemm<Tile64x128>(options, tensors);
  print_cutlass_record<Tile64x128>("CUTLASS_CANDIDATE", "baseline", tile_64x128_name.c_str(), options, tile_64x128);
  if (!kConciseLog) print_result(tile_64x128_name.c_str(), tile_64x128);

  if (!kConciseLog) print_configuration<Tile128x64>(tile_128x64_name.c_str(), options);
  Result tile_128x64 = run_tensorop_gemm<Tile128x64>(options, tensors);
  print_cutlass_record<Tile128x64>("CUTLASS_CANDIDATE", "baseline", tile_128x64_name.c_str(), options, tile_128x64);
  if (!kConciseLog) print_result(tile_128x64_name.c_str(), tile_128x64);

  if (!kConciseLog) print_configuration<Tile64x256>(tile_64x256_name.c_str(), options);
  Result tile_64x256 = run_tensorop_gemm<Tile64x256>(options, tensors);
  print_cutlass_record<Tile64x256>("CUTLASS_CANDIDATE", "baseline", tile_64x256_name.c_str(), options, tile_64x256);
  if (!kConciseLog) print_result(tile_64x256_name.c_str(), tile_64x256);

  char const *best_configuration_name = nullptr;
  Result const *best_result = nullptr;
  bool best_is_generated = false;
  auto consider = [&](char const *name, Result const &candidate) {
    if (candidate.status == cutlass::Status::kSuccess && candidate.passed &&
        (!best_result || candidate.avg_time_ms < best_result->avg_time_ms)) {
      best_configuration_name = name;
      best_result = &candidate;
      best_is_generated = false;
    }
  };
  if (generated_candidate_result.available &&
      generated_candidate_result.result.status == cutlass::Status::kSuccess &&
      generated_candidate_result.result.passed) {
    best_configuration_name = generated_candidate_result.name.c_str();
    best_result = &generated_candidate_result.result;
    best_is_generated = true;
  }
  consider(tile_128x128_name.c_str(), tile_128x128);
  consider(tile_128x256_name.c_str(), tile_128x256);
  consider(tile_256x128_name.c_str(), tile_256x128);
  consider(tile_64x128_name.c_str(), tile_64x128);
  consider(tile_128x64_name.c_str(), tile_128x64);
  consider(tile_64x256_name.c_str(), tile_64x256);

  if (!best_result) {
    std::cerr << "\nNo large-M candidate completed successfully and passed verification.\n";
    return EXIT_FAILURE;
  }

  if (kConciseLog) {
    if (best_is_generated) {
      print_generated_configuration(options);
    } else if (best_result == &tile_128x128) {
      print_configuration<Tile128x128>(best_configuration_name, options);
    } else if (best_result == &tile_128x256) {
      print_configuration<Tile128x256>(best_configuration_name, options);
    } else if (best_result == &tile_256x128) {
      print_configuration<Tile256x128>(best_configuration_name, options);
    } else if (best_result == &tile_64x128) {
      print_configuration<Tile64x128>(best_configuration_name, options);
    } else if (best_result == &tile_128x64) {
      print_configuration<Tile128x64>(best_configuration_name, options);
    } else {
      print_configuration<Tile64x256>(best_configuration_name, options);
    }
    print_result(best_configuration_name, *best_result);
  }

  std::cout << "\nBest configuration: " << best_configuration_name << "\n"
            << "  avg_time: " << best_result->avg_time_ms << " ms\n"
            << "  gflops: " << best_result->gflops << "\n";
  if (best_is_generated) {
    print_generated_record("CUTLASS_BEST", options, *best_result);
  } else if (best_result == &tile_128x128) {
    print_cutlass_record<Tile128x128>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  } else if (best_result == &tile_128x256) {
    print_cutlass_record<Tile128x256>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  } else if (best_result == &tile_256x128) {
    print_cutlass_record<Tile256x128>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  } else if (best_result == &tile_64x128) {
    print_cutlass_record<Tile64x128>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  } else if (best_result == &tile_128x64) {
    print_cutlass_record<Tile128x64>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  } else {
    print_cutlass_record<Tile64x256>("CUTLASS_BEST", "baseline", best_configuration_name, options, *best_result);
  }
  return EXIT_SUCCESS;
}

// Executes one preselected template for exact shapes whose best configuration
// has already been established on the target environment.
template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC,
          typename ThreadblockShape, typename WarpShape,
          typename ThreadblockSwizzle, int kStages>
int profile_optimal_template(char const *family, Options const &options,
                             int split_k_slices = 1) {
  using Gemm = GemmConfiguration<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
      ThreadblockShape, WarpShape, ThreadblockSwizzle, kStages>;
  using TensorSet = Tensors<LayoutA, LayoutB, LayoutC>;
  std::string const name = make_configuration_name<Gemm>(family);
  Options tuned_options = options;
  tuned_options.split_k_slices = std::max(1, split_k_slices);
  TensorSet tensors;
  if (!initialize_tensors(tuned_options, tensors) ||
      (kVerifyResults && !compute_reference(tuned_options, tensors))) {
    return EXIT_FAILURE;
  }
  print_configuration<Gemm>(name.c_str(), tuned_options);
  Result result = run_tensorop_gemm<Gemm>(tuned_options, tensors);
  print_result(name.c_str(), result);
  if (result.status != cutlass::Status::kSuccess || !result.passed) {
    return EXIT_FAILURE;
  }
  std::cout << "\nBest configuration: " << name << "\n"
            << "  avg_time: " << result.avg_time_ms << " ms\n"
            << "  gflops: " << result.gflops << "\n";
  return EXIT_SUCCESS;
}

int profile_optimal_only_candidate(Options const &options) {
#define GEMM_OPTIMAL_ENTRY(M, N, K, LAYOUT_A, LAYOUT_B, LAYOUT_C,              \
                           ALIGN_A, ALIGN_B, ALIGN_C,                           \
                           TB_M, TB_N, TB_K, W_M, W_N, W_K, SWIZZLE, STAGES)   \
  if (options.m == (M) && options.n == (N) && options.k == (K)) {              \
    return profile_optimal_template<                                            \
        LAYOUT_A, LAYOUT_B, LAYOUT_C, ALIGN_A, ALIGN_B, ALIGN_C,                \
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                             \
        cutlass::gemm::GemmShape<W_M, W_N, W_K>, SWIZZLE, STAGES>(              \
            "Optimal", options);                                               \
  }
#define GEMM_OPTIMAL_ENTRY_EX(M, N, K, LAYOUT_A, LAYOUT_B, LAYOUT_C,           \
                              ALIGN_A, ALIGN_B, ALIGN_C,                        \
                              TB_M, TB_N, TB_K, W_M, W_N, W_K, SWIZZLE,        \
                              STAGES, SPLIT_K)                                  \
  if (options.m == (M) && options.n == (N) && options.k == (K)) {              \
    return profile_optimal_template<                                            \
        LAYOUT_A, LAYOUT_B, LAYOUT_C, ALIGN_A, ALIGN_B, ALIGN_C,                \
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                             \
        cutlass::gemm::GemmShape<W_M, W_N, W_K>, SWIZZLE, STAGES>(              \
            "Optimal", options, SPLIT_K);                                      \
  }
#if __has_include("optimal_configurations.inc")
#include "optimal_configurations.inc"
#endif
#undef GEMM_OPTIMAL_ENTRY_EX
#undef GEMM_OPTIMAL_ENTRY
  return -1;
}

template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC, int kStages>
int profile_unmapped_single_candidate(Options const &options) {
  using Identity = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using TB32x256 = cutlass::gemm::GemmShape<32, 256, 32>;
  using TB32x128 = cutlass::gemm::GemmShape<32, 128, 32>;
  using TB64x128 = cutlass::gemm::GemmShape<64, 128, 32>;
  using W32x64 = cutlass::gemm::GemmShape<32, 64, 32>;
  using W16x64 = cutlass::gemm::GemmShape<16, 64, 32>;
  unsigned long long hash =
      static_cast<unsigned long long>(options.m) * 73856093ull ^
      static_cast<unsigned long long>(options.n) * 19349663ull ^
      static_cast<unsigned long long>(options.k) * 83492791ull;
  switch (hash % 3) {
    case 0:
      return profile_optimal_template<LayoutA, LayoutB, LayoutC,
          kAlignmentA, kAlignmentB, kAlignmentC,
          TB32x256, W32x64, Identity, kStages>("Fallback-random", options);
    case 1:
      return profile_optimal_template<LayoutA, LayoutB, LayoutC,
          kAlignmentA, kAlignmentB, kAlignmentC,
          TB32x128, W16x64, Identity, kStages>("Fallback-random", options);
    default:
      return profile_optimal_template<LayoutA, LayoutB, LayoutC,
          kAlignmentA, kAlignmentB, kAlignmentC,
          TB64x128, W32x64, Identity, kStages>("Fallback-random", options);
  }
}

// Runs one template generated from the locally measured cuBLASLt winner.
// The generated header only instantiates configurations referenced by the
// current Decode/Prefill shape set, keeping the CUTLASS compile search small.
template <typename LayoutA, typename LayoutB, typename LayoutC,
          int kAlignmentA, int kAlignmentB, int kAlignmentC,
          typename ThreadblockShape, typename WarpShape, int kStages>
int profile_cublaslt_template(char const *name, int split_k_slices,
                              Options const &options) {
  using Gemm = GemmConfiguration<
      LayoutA, LayoutB, LayoutC, kAlignmentA, kAlignmentB, kAlignmentC,
      ThreadblockShape, WarpShape,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;
  using TensorSet = Tensors<LayoutA, LayoutB, LayoutC>;
  Options tuned_options = options;
  tuned_options.split_k_slices = std::max(1, split_k_slices);
  TensorSet tensors;
  if (!initialize_tensors(tuned_options, tensors) ||
      (kVerifyResults && !compute_reference(tuned_options, tensors))) {
    return -1;
  }
  if (!kConciseLog) print_configuration<Gemm>(name, tuned_options);
  Result result = run_tensorop_gemm<Gemm>(tuned_options, tensors);
  if (!kConciseLog) print_result(name, result);
  if (result.status != cutlass::Status::kSuccess || !result.passed) {
    std::cerr << "Generated cuBLASLt-derived template was not runnable; "
                 "falling back to the baseline candidates.\n";
    return -1;
  }
  generated_candidate_result.available = true;
  generated_candidate_result.name = name;
  generated_candidate_result.result = result;
  generated_candidate_result.threadblock_m = ThreadblockShape::kM;
  generated_candidate_result.threadblock_n = ThreadblockShape::kN;
  generated_candidate_result.threadblock_k = ThreadblockShape::kK;
  generated_candidate_result.warp_m = WarpShape::kM;
  generated_candidate_result.warp_n = WarpShape::kN;
  generated_candidate_result.warp_k = WarpShape::kK;
  generated_candidate_result.stages = kStages;
  generated_candidate_result.alignment_a = kAlignmentA;
  generated_candidate_result.alignment_b = kAlignmentB;
  generated_candidate_result.alignment_c = kAlignmentC;
  generated_candidate_result.split_k_slices = tuned_options.split_k_slices;
  print_generated_record("CUTLASS_CANDIDATE", tuned_options, result);
  return -1;  // Always continue into the CUTLASS baseline candidate comparison.
}

#if __has_include("cublaslt_generated_candidates.inc")
#include "cublaslt_generated_candidates.inc"
#else
int profile_cublaslt_generated_candidate(Options const &) {
  return -1;
}
#endif

int maximum_fp16_alignment(int extent) {
  if (extent % 8 == 0) return 8;
  if (extent % 4 == 0) return 4;
  if (extent % 2 == 0) return 2;
  return 1;
}

template <int kAlignmentA>
int profile_large_m_by_alignment_b(Options const &options, int alignment_b) {
#define DISPATCH_LARGE_M(ALIGN_B)                                             \
  return profile_large_m_candidates<                                         \
      LayoutAAttention, LayoutBAttention, LayoutCAttention,                   \
      kAlignmentA, ALIGN_B, kAlignmentA,                                     \
      (kAlignmentA == 1 || ALIGN_B == 1) ? 2 : 3>(options)
  switch (alignment_b) {
    case 8: DISPATCH_LARGE_M(8);
    case 4: DISPATCH_LARGE_M(4);
    case 2: DISPATCH_LARGE_M(2);
    default: DISPATCH_LARGE_M(1);
  }
#undef DISPATCH_LARGE_M
}

int profile_large_m_by_alignment(Options const &options) {
  int alignment_a = maximum_fp16_alignment(options.m);
  int alignment_b = maximum_fp16_alignment(options.k);
  switch (alignment_a) {
    case 8: return profile_large_m_by_alignment_b<8>(options, alignment_b);
    case 4: return profile_large_m_by_alignment_b<4>(options, alignment_b);
    case 2: return profile_large_m_by_alignment_b<2>(options, alignment_b);
    default: return profile_large_m_by_alignment_b<1>(options, alignment_b);
  }
}

template <int kAlignmentA>
int profile_small_m_by_alignment_b(Options const &options, int alignment_b) {
#define DISPATCH_SMALL_M(ALIGN_B)                                             \
  return profile_all_candidates<                                             \
      LayoutAAttention, LayoutBAttention, LayoutCAttention,                   \
      kAlignmentA, ALIGN_B, kAlignmentA>(options)
  // Alignment 1 cannot use the cp.async candidates in profile_all_candidates.
  if constexpr (kAlignmentA == 1) {
    return profile_large_m_by_alignment_b<kAlignmentA>(options, alignment_b);
  } else {
    if (alignment_b == 1) {
      return profile_large_m_by_alignment_b<kAlignmentA>(options, alignment_b);
    }
    switch (alignment_b) {
      case 8: DISPATCH_SMALL_M(8);
      case 4: DISPATCH_SMALL_M(4);
      default: DISPATCH_SMALL_M(2);
    }
  }
#undef DISPATCH_SMALL_M
}

int profile_small_m_by_alignment(Options const &options) {
  int alignment_a = maximum_fp16_alignment(options.m);
  int alignment_b = maximum_fp16_alignment(options.k);
  switch (alignment_a) {
    case 8: return profile_small_m_by_alignment_b<8>(options, alignment_b);
    case 4: return profile_small_m_by_alignment_b<4>(options, alignment_b);
    case 2: return profile_small_m_by_alignment_b<2>(options, alignment_b);
    default: return profile_small_m_by_alignment_b<1>(options, alignment_b);
  }
}

template <int kAlignmentA, bool kLargeM>
int profile_unmapped_by_alignment_b(Options const &options, int alignment_b) {
#define DISPATCH_UNMAPPED(ALIGN_B)                                            \
  return profile_unmapped_single_candidate<                                  \
      LayoutAAttention, LayoutBAttention, LayoutCAttention,                   \
      kAlignmentA, ALIGN_B, kAlignmentA,                                     \
      (kAlignmentA == 1 || ALIGN_B == 1) ? 2 : (kLargeM ? 3 : 2)>(options)
  switch (alignment_b) {
    case 8: DISPATCH_UNMAPPED(8);
    case 4: DISPATCH_UNMAPPED(4);
    case 2: DISPATCH_UNMAPPED(2);
    default: DISPATCH_UNMAPPED(1);
  }
#undef DISPATCH_UNMAPPED
}

template <bool kLargeM>
int profile_unmapped_by_alignment(Options const &options) {
  int alignment_a = maximum_fp16_alignment(options.m);
  int alignment_b = maximum_fp16_alignment(options.k);
  switch (alignment_a) {
    case 8: return profile_unmapped_by_alignment_b<8, kLargeM>(options, alignment_b);
    case 4: return profile_unmapped_by_alignment_b<4, kLargeM>(options, alignment_b);
    case 2: return profile_unmapped_by_alignment_b<2, kLargeM>(options, alignment_b);
    default: return profile_unmapped_by_alignment_b<1, kLargeM>(options, alignment_b);
  }
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
        << "Invalid problem: dimensions and iterations must be positive. "
        << "For M=1, K and N must be aligned to 8.\n";
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

  if (kOptimalOnly) {
    int optimal_status = profile_optimal_only_candidate(options);
    if (optimal_status != -1) {
      return optimal_status;
    }
    std::cout << "Optimal-only: no exact M/N/K mapping; selecting one "
                 "deterministic pseudo-random fallback template.\n";
    if (options.m == 1) {
      return profile_unmapped_single_candidate<
          LayoutAM1, LayoutBM1, LayoutCM1, 8, 8, 8, 4>(options);
    }
    // The optimal-only fallback shares the same independent A/C-versus-B
    // alignment dispatch as the full candidate path below.
    return options.m >= 128 ? profile_unmapped_by_alignment<true>(options)
                            : profile_unmapped_by_alignment<false>(options);
  }

  // A generated exact-shape template takes precedence. A return value of -1
  // means no local cuBLASLt mapping exists and preserves the baseline search.
  profile_cublaslt_generated_candidate(options);

  // Step 4: select the physical layout used by the cuBLAS comparison.
  if (options.m == 1) {
    return profile_all_candidates<
        LayoutAM1, LayoutBM1, LayoutCM1,
        kAlignmentAM1, kAlignmentBM1, kAlignmentCM1>(options);
  }
  return options.m >= 128 ? profile_large_m_by_alignment(options)
                          : profile_small_m_by_alignment(options);
}
