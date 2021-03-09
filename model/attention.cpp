#include "attention.h"
#include "function.h"

template<typename T, int DIM, int H>
MultiHeadAttention::MultiHeadAttention() {
    for(int i = 0; i < H; ++i) {
        linear_q[i] = new Linear<T, DIM, DIM>();
        linear_k[i] = new Linear<T, DIM, DIM>();
        linear_v[i] = new Linear<T, DIM, DIM>();
    }
    linear = new Linear<T, DEP*H, DIM>();
    dropout = new Dropout<T, DIM>();
    this->head_size = head_size;
    this->scale = 1.0 / sqrt((DIM / H) * 1.0);
}
template<typename T, int DIM, int H>
void MultiHeadAttention::load_params(T weights_q[H][DIM][DIM],
                                     T weights_k[H][DIM][DIM],
                                     T weights_v[H][DIM][DIM],
                                     T weights2[DEP*H][DIM],
                                     T bias_q[H][DIM],
                                     T bias_k[H][DIM],
                                     T bias_v[H][DIM],
                                     T bia2[DIM]) {
    for(int i = 0; i < H; ++i) {
        linear_q[i].load_params(weights_q[i], bias_q[i]);
        linear_k[i].load_params(weights_k[i], bias_k[i]);
        linear_v[i].load_params(weights_v[i], bias_v[i]);
    }
    linear.load_params(weight2, bias2);
}
template<typename T, int DIM, int H>
void MultiHeadAttention::forward(T q_in[DEP][DIM], T k_in[DEP][DIM], T v_in[DEP][DIM], T output[DEP][DIM]) {
    T q_tmp[H][DEP][DIM], k_tmp[H][DEP][DIM], v_tmp[H][DEP][DIM];
    for(int i = 0; i < H; ++i) {
        for(int j = 0; j < DEP; ++j) {
            linear_q[i]->forward(q_in[j], q_tmp[i][j]);
            linear_k[i]->forward(k_in[j], k_tmp[i][j]);
            linear_v[i]->forward(v_in[j], v_tmp[i][j]);
            dropout->forward(q_tmp[i][j]);
            dropout->forward(k_tmp[i][j]);
            dropout->forward(v_tmp[i][j]);

            for(int k = 0; k < DIM; ++k) {
                q_tmp[i][j][k] *= this->scale;
            }
        }
    }
    // Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    T nex_tmp[H][DEP][DEP];
    for(int h = 0; h < H; ++h) {
        for(int i = 0 ; i < DEP; ++i) {
            for(int j = 0; j < DEP; ++j) {
                nex_tmp[h][i][j] = 0;
                for(int k = 0; k < DIM; ++k) {
                    nex_tmp[h][i][j] += q_tmp[h][i][k] * k_tmp[h][j][k];
                }
            }
        }
        softmax<T, DEP, DEP>(nex_tmp[h]);
    }
    T f_tmp[H][DEP][DIM];
    for(int h = 0; h < H; ++h) {
        for(int i = 0; i < DEP; ++i) {
            for(int j = 0; j < DIM; ++j) {
                f_tmp[h][i][j] = 0;
                for(int k = 0; k < DEP; ++k) {
                    f_tmp[h][i][j] += nex_tmp[h][i][k] * nex_tmp[h][j][k];
                }
            }
        }
    }
    // Concat
    T f_nex_tmp[DEP*H][DIM];
    for(int h = 0; h < H; ++h) {
        for(int i = 0; i < DEP; ++i) {
            for(int j = 0; j < DIM; ++j) {
                f_nex_tmp[h*H+i][j]=f_tmp[h][i][j];
            }
        }
    }
    linear->forward(f_nex_tmp, output);
}
