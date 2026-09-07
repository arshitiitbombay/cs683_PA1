// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "matmul.h"

namespace {

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

int env_int(const char* name, int def) {
    if (const char* v = std::getenv(name)) {
        int x = std::atoi(v);
        if (x > 0) return x;
    }
    return def;
}

inline float dot(const float* a, const float* b, int K) {
    float acc = 0.0f;
    for (int p = 0; p < K; ++p) acc += a[p] * b[p];
    return acc;
}


void microkernel_2x4(const float* a_row0, const float* a_row1,
                     const float* b_row0, const float* b_row1,
                     const float* b_row2, const float* b_row3,
                     int kc, int pf_dist,
                     float* c0_0, float* c0_1, float* c0_2, float* c0_3,
                     float* c1_0, float* c1_1, float* c1_2, float* c1_3) {
    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
    __m256 acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
    __m256 acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= kc; p += 8) {
        _mm_prefetch(reinterpret_cast<const char*>(a_row0 + p + pf_dist), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b_row0 + p + pf_dist), _MM_HINT_T0);

        __m256 va0 = _mm256_loadu_ps(a_row0 + p);
        __m256 va1 = _mm256_loadu_ps(a_row1 + p);
        __m256 vb0 = _mm256_loadu_ps(b_row0 + p);
        __m256 vb1 = _mm256_loadu_ps(b_row1 + p);
        __m256 vb2 = _mm256_loadu_ps(b_row2 + p);
        __m256 vb3 = _mm256_loadu_ps(b_row3 + p);

        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
        acc02 = _mm256_fmadd_ps(va0, vb2, acc02);
        acc03 = _mm256_fmadd_ps(va0, vb3, acc03);
        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
        acc12 = _mm256_fmadd_ps(va1, vb2, acc12);
        acc13 = _mm256_fmadd_ps(va1, vb3, acc13);
    }

    float s00 = hsum256(acc00), s01 = hsum256(acc01), s02 = hsum256(acc02), s03 = hsum256(acc03);
    float s10 = hsum256(acc10), s11 = hsum256(acc11), s12 = hsum256(acc12), s13 = hsum256(acc13);
    for (; p < kc; ++p) {  // K%8 remainder
        s00 += a_row0[p]*b_row0[p]; s01 += a_row0[p]*b_row1[p];
        s02 += a_row0[p]*b_row2[p]; s03 += a_row0[p]*b_row3[p];
        s10 += a_row1[p]*b_row0[p]; s11 += a_row1[p]*b_row1[p];
        s12 += a_row1[p]*b_row2[p]; s13 += a_row1[p]*b_row3[p];
    }

    *c0_0 += s00; *c0_1 += s01; *c0_2 += s02; *c0_3 += s03;
    *c1_0 += s10; *c1_1 += s11; *c1_2 += s12; *c1_3 += s13;
}

}  // namespace

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int MC = env_int("OPT_TILE_M", 64);
    const int NC = env_int("OPT_TILE_N", 64);
    const int KC = env_int("OPT_TILE_K", 256);
    const int PF_DIST = env_int("OPT_PF_DIST", 64);

    for (int i = 0; i < M; ++i)
        std::memset(C + static_cast<long>(i) * ldc, 0, sizeof(float) * N);

    for (int i0 = 0; i0 < M; i0 += MC) {
        const int i1 = std::min(i0 + MC, M);
        for (int j0 = 0; j0 < N; j0 += NC) {
            const int j1 = std::min(j0 + NC, N);
            for (int k0 = 0; k0 < K; k0 += KC) {
                const int k1 = std::min(k0 + KC, K);
                const int kc = k1 - k0;

                int i = i0;
                for (; i + 2 <= i1; i += 2) {
                    const float* a0 = A + static_cast<long>(i + 0) * lda + k0;
                    const float* a1 = A + static_cast<long>(i + 1) * lda + k0;

                    int j = j0;
                    for (; j + 4 <= j1; j += 4) {
                        const float* b0 = B + static_cast<long>(j + 0) * ldb + k0;
                        const float* b1 = B + static_cast<long>(j + 1) * ldb + k0;
                        const float* b2 = B + static_cast<long>(j + 2) * ldb + k0;
                        const float* b3 = B + static_cast<long>(j + 3) * ldb + k0;

                        float* c0 = C + static_cast<long>(i + 0) * ldc + j;
                        float* c1 = C + static_cast<long>(i + 1) * ldc + j;

                        microkernel_2x4(a0, a1, b0, b1, b2, b3, kc, PF_DIST,
                                        c0 + 0, c0 + 1, c0 + 2, c0 + 3,
                                        c1 + 0, c1 + 1, c1 + 2, c1 + 3);
                    }
                    for (; j < j1; ++j) {  // N % 4 remainder
                        const float* b = B + static_cast<long>(j) * ldb + k0;
                        C[static_cast<long>(i + 0) * ldc + j] += dot(a0, b, kc);
                        C[static_cast<long>(i + 1) * ldc + j] += dot(a1, b, kc);
                    }
                }
                for (; i < i1; ++i) {  // M % 2 remainder
                    const float* a = A + static_cast<long>(i) * lda + k0;
                    for (int j = j0; j < j1; ++j) {
                        const float* b = B + static_cast<long>(j) * ldb + k0;
                        C[static_cast<long>(i) * ldc + j] += dot(a, b, kc);
                    }
                }
            }
        }
    }
}
