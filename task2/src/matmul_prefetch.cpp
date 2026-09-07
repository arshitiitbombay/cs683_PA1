#include "matmul.h"
#include <immintrin.h>
#include <cstdlib>

void matmul_prefetch(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
    int tile_size = 16;
    const char* env = std::getenv("TILE_SIZE");
    if (env) tile_size = std::atoi(env);

    for (int i = 0; i < M; ++i) {
        const float* a = A + (long)i * lda;
        float* c = C + (long)i * ldc;

        for (int j0 = 0; j0 < N; j0 += tile_size) {
            int j_end = (j0 + tile_size < N) ? j0 + tile_size : N;

            float acc[32] = {0.0f};

            for (int k0 = 0; k0 < K; k0 += tile_size) {
                int k_end = (k0 + tile_size < K) ? k0 + tile_size : K;

                int next = k0 + tile_size;

                if (next < K) {
                    _mm_prefetch(
                        reinterpret_cast<const char*>(a + next),
                        _MM_HINT_T0
                    );

                    for (int j = j0; j < j_end; ++j) {
                        const float* b = B + (long)j * ldb;

                        _mm_prefetch(
                            reinterpret_cast<const char*>(b + next),
                            _MM_HINT_T0
                        );
                    }
                }

                for (int p = k0; p < k_end; ++p) {
                    float av = a[p];

                    for (int j = j0; j < j_end; ++j) {
                        const float* b = B + (long)j * ldb;
                        acc[j - j0] += av * b[p];
                    }
                }
            }

            for (int j = j0; j < j_end; ++j)
                c[j] = acc[j - j0];
        }
    }
}
