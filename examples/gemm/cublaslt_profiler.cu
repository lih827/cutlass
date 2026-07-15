#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

template <typename T>
int get_config(cublasLtMatmulAlgo_t const &algo,
               cublasLtMatmulAlgoConfigAttributes_t attribute, T &value) {
  return cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), nullptr) == CUBLAS_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
  int m = 0, n = 0, k = 0, iterations = 20, requested = 32;
  size_t workspace_bytes = 64ull << 20;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto integer = [&](char const *prefix, int &out) {
      std::string p(prefix); if (a.rfind(p, 0) == 0) { out = std::stoi(a.substr(p.size())); return true; }
      return false;
    };
    if (integer("--m=", m) || integer("--n=", n) || integer("--k=", k) ||
        integer("--iterations=", iterations) || integer("--candidates=", requested)) continue;
    if (a.rfind("--workspace-mb=", 0) == 0)
      workspace_bytes = size_t(std::stoull(a.substr(15))) << 20;
    else if (a == "--help") {
      std::cout << "Usage: cublaslt_profiler --m=M --n=N --k=K [--iterations=20] "
                   "[--candidates=32] [--workspace-mb=64]\n";
      return 0;
    } else { std::cerr << "Unknown argument: " << a << "\n"; return 2; }
  }
  if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0 || requested <= 0) return 2;

  size_t a_bytes = size_t(m) * k * sizeof(__half);
  size_t b_bytes = size_t(k) * n * sizeof(__half);
  size_t c_bytes = size_t(m) * n * sizeof(__half);
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
  LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_16F, CUDA_R_16F));
  cublasOperation_t trans = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
  // Column-major NN: A[M,K], B[K,N], C/D[M,N].
  LT_CHECK(cublasLtMatrixLayoutCreate(&ad, CUDA_R_16F, m, k, m));
  LT_CHECK(cublasLtMatrixLayoutCreate(&bd, CUDA_R_16F, k, n, k));
  LT_CHECK(cublasLtMatrixLayoutCreate(&cd, CUDA_R_16F, m, n, m));
  LT_CHECK(cublasLtMatrixLayoutCreate(&dd, CUDA_R_16F, m, n, m));
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));

  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(requested);
  int returned = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op, ad, bd, cd, dd, pref, requested, heuristics.data(), &returned));
  if (!returned) { std::cerr << "No cuBLASLt heuristic candidate\n"; return 1; }

  __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
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
  double gflops = 2.0 * double(m) * n * k / (double(best_ms) * 1.0e6);
  std::cout << std::fixed << std::setprecision(4)
            << "CUBLASLT_BEST m=" << m << " n=" << n << " k=" << k
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
