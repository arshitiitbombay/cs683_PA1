// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include "convolution.h"
#include <immintrin.h>

void conv_optimized(const float* in, float* out, const float* ker,
               int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    if (K == 3) {

        const __m256 k00 = _mm256_set1_ps(ker[0]);
        const __m256 k01 = _mm256_set1_ps(ker[1]);
        const __m256 k02 = _mm256_set1_ps(ker[2]);

        const __m256 k10 = _mm256_set1_ps(ker[3]);
        const __m256 k11 = _mm256_set1_ps(ker[4]);
        const __m256 k12 = _mm256_set1_ps(ker[5]);

        const __m256 k20 = _mm256_set1_ps(ker[6]);
        const __m256 k21 = _mm256_set1_ps(ker[7]);
        const __m256 k22 = _mm256_set1_ps(ker[8]);

        for (int oy = 0; oy < H; ++oy) {

            const float* row0 =
                in + oy * in_stride;

            const float* row1 =
                row0 + in_stride;

            const float* row2 =
                row1 + in_stride;

            float* out_row =
                out + oy * W;

            int ox = 0;

            for (; ox + 7 < W; ox += 8) {

                __m256 acc = _mm256_setzero_ps();

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row0 + ox),
                    k00,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row0 + ox + 1),
                    k01,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row0 + ox + 2),
                    k02,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row1 + ox),
                    k10,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row1 + ox + 1),
                    k11,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row1 + ox + 2),
                    k12,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row2 + ox),
                    k20,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row2 + ox + 1),
                    k21,
                    acc);

                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row2 + ox + 2),
                    k22,
                    acc);

                _mm256_storeu_ps(out_row + ox, acc);
            }

            for (; ox < W; ++ox) {

                float acc = 0.0f;

                acc += row0[ox]     * ker[0];
                acc += row0[ox + 1] * ker[1];
                acc += row0[ox + 2] * ker[2];

                acc += row1[ox]     * ker[3];
                acc += row1[ox + 1] * ker[4];
                acc += row1[ox + 2] * ker[5];

                acc += row2[ox]     * ker[6];
                acc += row2[ox + 1] * ker[7];
                acc += row2[ox + 2] * ker[8];

                out_row[ox] = acc;
            }
        }

        return;
    }


    // General K: AVX2 SIMD + loop unrolling
for (int oy = 0; oy < H; ++oy) {

    float* out_row = out + oy * W;

    int ox = 0;

    for (; ox + 7 < W; ox += 8) {

        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();

        for (int ky = 0; ky < K; ++ky) {

            const float* in_row =
                in + (oy + ky) * in_stride;

            int kx = 0;

            for (; kx + 3 < K; kx += 4) {

                __m256 k0 =
                    _mm256_broadcast_ss(
                        &ker[ky * K + kx]
                    );

                __m256 k1 =
                    _mm256_broadcast_ss(
                        &ker[ky * K + kx + 1]
                    );

                __m256 k2 =
                    _mm256_broadcast_ss(
                        &ker[ky * K + kx + 2]
                    );

                __m256 k3 =
                    _mm256_broadcast_ss(
                        &ker[ky * K + kx + 3]
                    );

                __m256 in0 =
                    _mm256_loadu_ps(
                        in_row + ox + kx
                    );

                __m256 in1 =
                    _mm256_loadu_ps(
                        in_row + ox + kx + 1
                    );

                __m256 in2 =
                    _mm256_loadu_ps(
                        in_row + ox + kx + 2
                    );

                __m256 in3 =
                    _mm256_loadu_ps(
                        in_row + ox + kx + 3
                    );

                acc1 =
                    _mm256_fmadd_ps(
                        in0, k0, acc1
                    );

                acc2 =
                    _mm256_fmadd_ps(
                        in1, k1, acc2
                    );

                acc3 =
                    _mm256_fmadd_ps(
                        in2, k2, acc3
                    );

                acc4 =
                    _mm256_fmadd_ps(
                        in3, k3, acc4
                    );
            }

            for (; kx < K; ++kx) {

                __m256 vk =
                    _mm256_broadcast_ss(
                        &ker[ky * K + kx]
                    );

                __m256 vin =
                    _mm256_loadu_ps(
                        in_row + ox + kx
                    );

                acc1 =
                    _mm256_fmadd_ps(
                        vin, vk, acc1
                    );
            }
        }

        __m256 acc =
            _mm256_add_ps(acc1, acc2);

        acc =
            _mm256_add_ps(acc, acc3);

        acc =
            _mm256_add_ps(acc, acc4);

        _mm256_storeu_ps(
            out_row + ox,
            acc
        );
    }

    for (; ox < W; ++ox) {

        float acc = 0.0f;

        for (int ky = 0; ky < K; ++ky) {

            for (int kx = 0; kx < K; ++kx) {

                acc +=
                    in[(oy + ky) * in_stride + ox + kx]
                    * ker[ky * K + kx];
            }
        }

        out_row[ox] = acc;
    }
}
}

