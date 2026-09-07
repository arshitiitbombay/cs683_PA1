#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    constexpr int TILE_H = 64;
    constexpr int TILE_W = 64;
    // (Arshit) This will work but it exploits other factors also
    if (K == 0) {

        const float k00 = ker[0];
        const float k01 = ker[1];
        const float k02 = ker[2];

        const float k10 = ker[3];
        const float k11 = ker[4];
        const float k12 = ker[5];

        const float k20 = ker[6];
        const float k21 = ker[7];
        const float k22 = ker[8];

        for (int ty = 0; ty < H; ty += TILE_H) {
            for (int tx = 0; tx < W; tx += TILE_W) {

                const int y_end =
                    (ty + TILE_H < H) ? ty + TILE_H : H;

                const int x_end =
                    (tx + TILE_W < W) ? tx + TILE_W : W;

                for (int oy = ty; oy < y_end; ++oy) {

                    const float* row0 =
                        in + (oy + 0) * in_stride;

                    const float* row1 =
                        in + (oy + 1) * in_stride;

                    const float* row2 =
                        in + (oy + 2) * in_stride;

                    float* out_row =
                        out + oy * W;

                    for (int ox = tx; ox < x_end; ++ox) {

                        float acc = 0.0f;

                        acc += row0[ox]     * k00;
                        acc += row0[ox + 1] * k01;
                        acc += row0[ox + 2] * k02;

                        acc += row1[ox]     * k10;
                        acc += row1[ox + 1] * k11;
                        acc += row1[ox + 2] * k12;

                        acc += row2[ox]     * k20;
                        acc += row2[ox + 1] * k21;
                        acc += row2[ox + 2] * k22;

                        out_row[ox] = acc;
                    }
                }
            }
        }

        return;
    }

    // General K 
    for (int ty = 0; ty < H; ty += TILE_H) {
        for (int tx = 0; tx < W; tx += TILE_W) {

            int y_end = std::min(ty + TILE_H, H);
            int x_end = std::min(tx + TILE_W, W);

            for (int oy = ty; oy < y_end; ++oy)
                for (int ox = tx; ox < x_end; ++ox)
                    out[oy * W + ox] = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    float k = ker[ky * K + kx];

                    for (int oy = ty; oy < y_end; ++oy) {

                        const float* in_row =
                            in + (oy + ky) * in_stride;

                        for (int ox = tx; ox < x_end; ++ox) {

                            out[oy * W + ox] +=
                                in_row[ox + kx] * k;
                        }
                    }
                }
            }
        }
    }
}
