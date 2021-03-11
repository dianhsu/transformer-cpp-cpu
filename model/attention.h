//
// Created by dianhsu on 2021/03/10.
//

#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H


#include <cmath>
#include <array>

#include "linear.h"
#include "dropout.h"
#include "softmax.h"

namespace transformer {
    template<typename T, int DIM, int HEAD_SIZE>
    struct MultiHeadAttentionParameter {
        LinearParameter <T, DIM, DIM> linear_q_p[HEAD_SIZE], linear_k_p[HEAD_SIZE], linear_v_p[HEAD_SIZE];
        LinearParameter<T, DIM * HEAD_SIZE, DIM> linear_p;
        T dr;

        MultiHeadAttentionParameter() {
            dr = 0.1;
        }

        long long count() {
            return linear_k_p[0].count() * HEAD_SIZE * 3 + linear_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int HEAD_SIZE>
    class MultiHeadAttention {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &q_in,
                            std::array<std::array<T, DIM>, DEP> &k_in,
                            std::array<std::array<T, DIM>, DEP> &v_in,
                            std::array<std::array<T, DIM>, DEP> &output,
                            MultiHeadAttentionParameter<T, DIM, HEAD_SIZE> &p) {
            T scale = 1.0 / sqrt((T) DIM * 1.0 / HEAD_SIZE);

            auto q_tmp = std::array<std::array<std::array<T, DIM>, DEP>, HEAD_SIZE>{};
            auto k_tmp = std::array<std::array<std::array<T, DIM>, DEP>, HEAD_SIZE>{};
            auto v_tmp = std::array<std::array<std::array<T, DIM>, DEP>, HEAD_SIZE>{};
            auto q_tmp_2 = std::array<std::array<std::array<T, DIM>, DEP>, HEAD_SIZE>{};


            for (int i = 0; i < HEAD_SIZE; ++i) {
                MultiLinear<T, DIM, DIM, DEP>::forward(q_in, q_tmp[i], p.linear_q_p[i]);
                MultiLinear<T, DIM, DIM, DEP>::forward(k_in, k_tmp[i], p.linear_k_p[i]);
                MultiLinear<T, DIM, DIM, DEP>::forward(v_in, v_tmp[i], p.linear_v_p[i]);

                for (int j = 0; j < DEP; ++j) {
                    Dropout<T, DIM>::forward(q_tmp[i][j], q_tmp_2[i][j], p.dr);
                    for (int k = 0; k < DIM; ++k) {
                        q_tmp_2[i][j][k] *= scale;
                    }
                }
            }

            // Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V && Concat
            auto nex_tmp = std::array<std::array<std::array<std::array<T, DEP>, DEP>, HEAD_SIZE>, 2>{};
            for (int h = 0; h < HEAD_SIZE; ++h) {
                for (int i = 0; i < DEP; ++i) {
                    for (int j = 0; j < DEP; ++j) {
                        nex_tmp[0][h][i][j] = 0;
                        for (int k = 0; k < DIM; ++k) {
                            nex_tmp[0][h][i][j] += q_tmp_2[h][i][k] * k_tmp[h][j][k];
                        }
                    }
                }
                Softmax<T, DEP, DEP>::forward(nex_tmp[0][h], nex_tmp[1][h]);
            }
            auto f_nex_tmp = std::array<std::array<T, DIM * HEAD_SIZE>, DEP>{};
            for (int h = 0; h < HEAD_SIZE; ++h) {
                for (int i = 0; i < DEP; ++i) {
                    for (int j = 0; j < DIM; ++j) {
                        f_nex_tmp[i][h * HEAD_SIZE + j] = 0;
                        for (int k = 0; k < DEP; ++k) {
                            f_nex_tmp[i][h * HEAD_SIZE + j] += nex_tmp[1][h][i][k] * v_tmp[h][k][j];
                        }
                    }
                }
            }

            MultiLinear<T, DIM * HEAD_SIZE, DIM, DEP>::forward(f_nex_tmp, output, p.linear_p);
        }
    };
}
#endif //TRANSFORMER_ATTENTION_H
