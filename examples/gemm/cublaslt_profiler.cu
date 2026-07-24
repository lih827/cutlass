#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { auto s_ = (x); if (s_ != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(s_) << "\n"; return 1; } } while (0)
#define LT_CHECK(x) do { auto s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  std::cerr << "cuBLASLt error: " << int(s_) << "\n"; return 1; } } while (0)

#if defined(GEMM_CUBLASLT_FP32_ACCUMULATOR) && GEMM_CUBLASLT_FP32_ACCUMULATOR
using ScaleType = float;
constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
constexpr cudaDataType_t kScaleType = CUDA_R_32F;
constexpr char const *kAccumulatorName = "float";
#else
using ScaleType = __half;
constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_16F;
constexpr cudaDataType_t kScaleType = CUDA_R_16F;
constexpr char const *kAccumulatorName = "half";
#endif

template <typename T>
int get_config(cublasLtMatmulAlgo_t const &algo,
               cublasLtMatmulAlgoConfigAttributes_t attribute, T &value) {
  return cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), nullptr) == CUBLAS_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
  int m = 0, n = 0, k = 0, iterations = 20, requested = 32;
  int batch_count = 1;
  std::string trans_string = "NN";
  size_t workspace_bytes = 64ull << 20;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto integer = [&](char const *prefix, int &out) {
      std::string p(prefix); if (a.rfind(p, 0) == 0) { out = std::stoi(a.substr(p.size())); return true; }
      return false;
    };
    if (integer("--m=", m) || integer("--n=", n) || integer("--k=", k) ||
        integer("--iterations=", iterations) || integer("--candidates=", requested) ||
        integer("--batch-count=", batch_count)) continue;
    if (a.rfind("--workspace-mb=", 0) == 0)
      workspace_bytes = size_t(std::stoull(a.substr(15))) << 20;
    else if (a.rfind("--trans=", 0) == 0)
      trans_string = a.substr(8);
    else if (a == "--help") {
      std::cout << "Usage: cublaslt_profiler --m=M --n=N --k=K [--iterations=20] "
                   "[--trans=NN|NT|TN|TT] [--batch-count=1] "
                   "[--candidates=32] [--workspace-mb=64]\n";
      return 0;
    } else { std::cerr << "Unknown argument: " << a << "\n"; return 2; }
  }
  std::transform(trans_string.begin(), trans_string.end(), trans_string.begin(), ::toupper);
  if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0 || requested <= 0 ||
      batch_count <= 0 ||
      (trans_string != "NN" && trans_string != "NT" &&
       trans_string != "TN" && trans_string != "TT")) return 2;

  size_t a_elements = size_t(m) * k;
  size_t b_elements = size_t(k) * n;
  size_t c_elements = size_t(m) * n;
  size_t a_bytes = a_elements * batch_count * sizeof(__half);
  size_t b_bytes = b_elements * batch_count * sizeof(__half);
  size_t c_bytes = c_elements * batch_count * sizeof(__half);
  __half *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
  void *workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&A, a_bytes)); CUDA_CHECK(cudaMalloc(&B, b_bytes));
  CUDA_CHECK(cudaMalloc(&C, c_bytes)); CUDA_CHECK(cudaMalloc(&D, c_bytes));
  CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
  CUDA_CHECK(cudaMemset(A, 0, a_bytes)); CUDA_CHECK(cudaMemset(B, 0, b_bytes));
  CUDA_CHECK(cudaMemset(C, 0, c_bytes)); CUDA_CHECK(cudaMemset(D, 0, c_bytes));

  cublasLtHandle_t lt{}; cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t ad{}, bd{}, cd{}, dd{}; cublasLtMatmulPreference_t pref{};
  LT_CHECK(cublasLtCreate(&lt));
  LT_CHECK(cublasLtMatmulDescCreate(&op, kComputeType, kScaleType));
  cublasOperation_t transa = trans_string[0] == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = trans_string[1] == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  int const a_rows = transa == CUBLAS_OP_N ? m : k;
  int const a_cols = transa == CUBLAS_OP_N ? k : m;
  int const b_rows = transb == CUBLAS_OP_N ? k : n;
  int const b_cols = transb == CUBLAS_OP_N ? n : k;
  LT_CHECK(cublasLtMatrixLayoutCreate(&ad, CUDA_R_16F, a_rows, a_cols, a_rows));
  LT_CHECK(cublasLtMatrixLayoutCreate(&bd, CUDA_R_16F, b_rows, b_cols, b_rows));
  LT_CHECK(cublasLtMatrixLayoutCreate(&cd, CUDA_R_16F, m, n, m));
  LT_CHECK(cublasLtMatrixLayoutCreate(&dd, CUDA_R_16F, m, n, m));
  if (batch_count > 1) {
    int64_t stride_a = int64_t(a_elements);
    int64_t stride_b = int64_t(b_elements);
    int64_t stride_c = int64_t(c_elements);
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        ad, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        bd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        cd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        dd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        ad, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        bd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        cd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        dd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
  }
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));

  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(requested);
  int returned = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op, ad, bd, cd, dd, pref, requested, heuristics.data(), &returned));
  if (!returned) { std::cerr << "No cuBLASLt heuristic candidate\n"; return 1; }

  ScaleType alpha = ScaleType(1.0f), beta = ScaleType(0.0f);
  cudaEvent_t start{}, stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  float best_ms = std::numeric_limits<float>::max(); int best = -1;
  for (int i = 0; i < returned; ++i) {
    auto &h = heuristics[i]; if (h.state != CUBLAS_STATUS_SUCCESS || h.workspaceSize > workspace_bytes) continue;
    if (cublasLtMatmul(lt, op, &alpha, A, ad, B, bd, &beta, C, cd, D, dd,
                       &h.algo, workspace, workspace_bytes, 0) != CUBLAS_STATUS_SUCCESS) continue;
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(start));
    bool good = true;
    for (int j = 0; j < iterations; ++j)
      good &= cublasLtMatmul(lt, op, &alpha, A, ad, B, bd, &beta, C, cd, D, dd,
                             &h.algo, workspace, workspace_bytes, 0) == CUBLAS_STATUS_SUCCESS;
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0; CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    if (good && elapsed / iterations < best_ms) { best_ms = elapsed / iterations; best = i; }
  }
  if (best < 0) { std::cerr << "No runnable cuBLASLt candidate\n"; return 1; }

  auto const &h = heuristics[best];
  int algo_id = -1, tile_id = 0, stages_id = 0, split_k = 0, reduction = 0, swizzle = 0, custom = 0;
  get_config(h.algo, CUBLASLT_ALGO_CONFIG_ID, algo_id);
  get_config(h.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile_id);
  get_config(h.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, stages_id);
  get_config(h.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, split_k);
  get_config(h.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, reduction);
  get_config(h.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle);
  get_config(h.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom);
  double gflops = 2.0 * double(m) * n * k * batch_count /
                  (double(best_ms) * 1.0e6);
  std::cout << std::fixed << std::setprecision(4)
            << "CUBLASLT_BEST m=" << m << " n=" << n << " k=" << k
            << " trans=" << trans_string
            << " batch_count=" << batch_count
            << " accumulator=" << kAccumulatorName
            << " algo_id=" << algo_id << " tile_id=" << tile_id
            << " stages_id=" << stages_id << " split_k=" << split_k
            << " reduction=" << reduction << " swizzle=" << swizzle
            << " custom=" << custom << " workspace=" << h.workspaceSize
            << " avg_time_ms=" << best_ms << " gflops=" << gflops << "\n";

  cudaEventDestroy(start); cudaEventDestroy(stop); cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(ad); cublasLtMatrixLayoutDestroy(bd);
  cublasLtMatrixLayoutDestroy(cd); cublasLtMatrixLayoutDestroy(dd);
  cublasLtMatmulDescDestroy(op); cublasLtDestroy(lt);
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D); cudaFree(workspace);
  return 0;
}
