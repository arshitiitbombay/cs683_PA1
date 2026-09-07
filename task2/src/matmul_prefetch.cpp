#include "matmul.h"
#include <immintrin.h>
#include <cstdlib>

const int TILE_SIZE = 32;

void matmul_prefetch(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
    int tile_size = 32;
    const char* env = std::getenv("TILE_SIZE");
    if (env) tile_size = std::atoi(env);

    for (int i = 0; i < M; ++i) {
        const float* a = A + (long)i * lda;
        float* c = C + (long)i * ldc;

        for (int j = 0; j < N; ++j) {
            const float* b = B + (long)j * ldb;
            float acc = 0.0f;

            for (int p0 = 0; p0 < K; p0 += tile_size) {
                int p_end = (p0 + tile_size < K) ? p0 + tile_size : K;

                int next = p0 + tile_size;

                if (next < K) {
                    _mm_prefetch(
                        reinterpret_cast<const char*>(a + next),
                        _MM_HINT_T0
                    );

                    _mm_prefetch(
                        reinterpret_cast<const char*>(b + next),
                        _MM_HINT_T0
                    );
                }

                for (int p = p0; p < p_end; ++p)
                    acc += a[p] * b[p];
            }

            c[j] = acc;
        }
    }
}
