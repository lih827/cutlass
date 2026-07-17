// Exact-shape templates used by builds with GEMM_OPTIMAL_ONLY=1.
// Add or replace one GEMM_OPTIMAL_ENTRY per measured M/N/K winner.
// Parameters:
// M, N, K, LayoutA/B/C, AlignmentA/B/C,
// Threadblock M/N/K, Warp M/N/K, Swizzle, Stages.

GEMM_OPTIMAL_ENTRY(1, 3584, 3584,
    LayoutAM1, LayoutBM1, LayoutCM1, 8, 8, 8,
    32, 128, 32, 16, 64, 32,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4)

GEMM_OPTIMAL_ENTRY(1, 18944, 3584,
    LayoutAM1, LayoutBM1, LayoutCM1, 8, 8, 8,
    32, 256, 32, 32, 64, 32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4)

GEMM_OPTIMAL_ENTRY(1, 3584, 18944,
    LayoutAM1, LayoutBM1, LayoutCM1, 8, 8, 8,
    32, 128, 32, 16, 64, 32,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4)

GEMM_OPTIMAL_ENTRY(1, 152064, 3584,
    LayoutAM1, LayoutBM1, LayoutCM1, 8, 8, 8,
    32, 256, 32, 32, 64, 32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4)

GEMM_OPTIMAL_ENTRY(128, 3584, 3584,
    LayoutAAttention, LayoutBAttention, LayoutCAttention, 8, 8, 8,
    128, 64, 32, 64, 32, 32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3)

GEMM_OPTIMAL_ENTRY(128, 18944, 3584,
    LayoutAAttention, LayoutBAttention, LayoutCAttention, 8, 8, 8,
    128, 128, 32, 64, 64, 32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3)

GEMM_OPTIMAL_ENTRY(128, 3584, 18944,
    LayoutAAttention, LayoutBAttention, LayoutCAttention, 8, 8, 8,
    64, 64, 32, 32, 64, 32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4)
