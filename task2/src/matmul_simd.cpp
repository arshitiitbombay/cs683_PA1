// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include "matmul.h"
#include <immintrin.h>

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc)
{
    for (int i = 0; i < M; ++i) {
        const float* a = A + (long)i * lda;

        for (int j = 0; j < N; ++j) {
            const float* b = B + (long)j * ldb;

            __m256 vacc = _mm256_setzero_ps();

            int p = 0;
            for (; p + 7 < K; p += 8) {
                __m256 va = _mm256_loadu_ps(a + p);
                __m256 vb = _mm256_loadu_ps(b + p);

                vacc = _mm256_fmadd_ps(va, vb, vacc);
            }

            __m128 low  = _mm256_castps256_ps128(vacc);
            __m128 high = _mm256_extractf128_ps(vacc, 1);

            low = _mm_add_ps(low, high);

            __m128 temp = _mm_movehl_ps(low, low);
            low = _mm_add_ps(low, temp);

            temp = _mm_shuffle_ps(low, low, 1);
            low = _mm_add_ss(low, temp);

            float acc = _mm_cvtss_f32(low);

            for (; p < K; ++p)
                acc += a[p] * b[p];

            C[(long)i * ldc + j] = acc;
        }
    }
}
}
