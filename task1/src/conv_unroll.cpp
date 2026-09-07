#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    if (K == 3) {

        const float k00 = ker[0];
        const float k01 = ker[1];
        const float k02 = ker[2];
        const float k10 = ker[3];
        const float k11 = ker[4];
        const float k12 = ker[5];
        const float k20 = ker[6];
        const float k21 = ker[7];
        const float k22 = ker[8];

        for (int oy = 0; oy < H; ++oy) {

            const float* row0 = in + (oy + 0) * in_stride;
            const float* row1 = in + (oy + 1) * in_stride;
            const float* row2 = in + (oy + 2) * in_stride;

            float* out_row = out + oy * W;

            for (int ox = 0; ox < W; ++ox) {

                out_row[ox] =
                    row0[ox]     * k00 +
                    row0[ox + 1] * k01 +
                    row0[ox + 2] * k02 +

                    row1[ox]     * k10 +
                    row1[ox + 1] * k11 +
                    row1[ox + 2] * k12 +

                    row2[ox]     * k20 +
                    row2[ox + 1] * k21 +
                    row2[ox + 2] * k22;

            }
        }

        return;
    }

    constexpr int N = 5;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {

            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            // float acc6 = 0.0f;
            // float acc7 = 0.0f;
            // float acc8 = 0.0f;
            // float acc9 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {

                const float* in_row =
                    in + (oy + ky) * in_stride + ox;

                const float* ker_row =
                    ker + ky * K;

                const int limit = K - (K % N);

                for (int kx = 0; kx < limit; kx += N) {

                    acc1 += in_row[kx]     * ker_row[kx];
                    acc2 += in_row[kx + 1] * ker_row[kx + 1];
                    acc3 += in_row[kx + 2] * ker_row[kx + 2];
                    acc4 += in_row[kx + 3] * ker_row[kx + 3];
                    acc5 += in_row[kx + 4] * ker_row[kx + 4];
                    // acc6 += in_row[kx + 5] * ker_row[kx + 5];
                    // acc7 += in_row[kx + 6] * ker_row[kx + 6];
                    // acc8 += in_row[kx + 7] * ker_row[kx + 7];
                    // acc9 += in_row[kx + 8] * ker_row[kx + 8];
                }

                // Remaining elements
                for (int kx = limit; kx < K; ++kx) {
                    acc1 += in_row[kx] * ker_row[kx];
                }
            }

            float acc = acc1 + acc2 + acc3 + acc4 + acc5;
                    //   + acc6 + acc7 + acc8 + acc9;

            out[oy * W + ox] = acc;
        }
    }
}
