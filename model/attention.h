#ifndef __MODEL_ATTENTION_H__
#define __MODEL_ATTENTION_H__

#include <cmath>

#include "linear.h"
#include "dropout.h"
#include "function.h"

template<typename T, int DIM, int H>
struct MultiHeadAttentionParam {
    LinearParam<T, DIM, DIM> *linear_q_p[H], *linear_k_p[H], *linear_v_p[H];
    LinearParam<T, DIM * H, DIM> *linear_p;
    T *dropout_rate;

    MultiHeadAttentionParam() {
        for (int i = 0; i < H; ++i) {
            linear_q_p[i] = new LinearParam<T, DIM, DIM>();
            linear_k_p[i] = new LinearParam<T, DIM, DIM>();
            linear_v_p[i] = new LinearParam<T, DIM, DIM>();
        }
        linear_p = new LinearParam<T, DIM * H, DIM>();
    }

    ~MultiHeadAttentionParam() {
        for (int i = 0; i < H; ++i) {
            delete linear_q_p[i];
            delete linear_k_p[i];
            delete linear_v_p[i];
        }
        delete linear_p;
    }
    long long count(){
        return linear_k_p[0]->count() * H * 3 + linear_p->count();
    }
};


template<typename T, int DIM, int DEP, int H>
class MultiHeadAttention {
public:
    MultiHeadAttention() {
        for (int i = 0; i < H; ++i) {
            linear_q[i] = new Linear<T, DIM, DIM>();
            linear_k[i] = new Linear<T, DIM, DIM>();
            linear_v[i] = new Linear<T, DIM, DIM>();
        }
        linear = new Linear<T, DIM * H, DIM>();
        dropout = new Dropout<T, DIM>();
        this->scale = 1.0 / sqrt((DIM / H) * 1.0);
    }

    void load_params(MultiHeadAttentionParam<T, DIM, H> *p) {
        for (int i = 0; i < H; ++i) {
            if (p->linear_q_p[i] != nullptr) {
                linear_q[i]->load_params(p->linear_q_p[i]);

            }
            if (p->linear_k_p[i] != nullptr) {
                linear_k[i]->load_params(p->linear_k_p[i]);
            }
            if (p->linear_v_p[i] != nullptr) {
                linear_v[i]->load_params(p->linear_v_p[i]);
            }
        }
        if (p->linear_p != nullptr) {
            linear->load_params(p->linear_p);
        }
        if (p->dropout_rate != nullptr) {
            dropout->load_params(*(p->dropout_rate));
        }
    }

    void forward(T q_in[DEP][DIM], T k_in[DEP][DIM], T v_in[DEP][DIM], T output[DEP][DIM]) {
        T q_tmp[H][DEP][DIM], k_tmp[H][DEP][DIM], v_tmp[H][DEP][DIM];
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < DEP; ++j) {
                linear_q[i]->forward(q_in[j], q_tmp[i][j]);
                linear_k[i]->forward(k_in[j], k_tmp[i][j]);
                linear_v[i]->forward(v_in[j], v_tmp[i][j]);
                dropout->forward(q_tmp[i][j]);
                dropout->forward(k_tmp[i][j]);
                dropout->forward(v_tmp[i][j]);
                for (int k = 0; k < DIM; ++k) {
                    q_tmp[i][j][k] *= this->scale;
                }
            }
        }
        // Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        T nex_tmp[H][DEP][DEP];
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DEP; ++j) {
                    nex_tmp[h][i][j] = 0;
                    for (int k = 0; k < DIM; ++k) {
                        nex_tmp[h][i][j] += q_tmp[h][i][k] * k_tmp[h][j][k];
                    }
                }
            }
            softmax<T, DEP, DEP>(nex_tmp[h]);
        }
        T f_tmp[H][DEP][DIM];
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    f_tmp[h][i][j] = 0;
                    for (int k = 0; k < DEP; ++k) {
                        f_tmp[h][i][j] += nex_tmp[h][i][k] * nex_tmp[h][j][k];
                    }
                }
            }
        }
        // Concat
        T f_nex_tmp[DEP][DIM * H];
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    f_nex_tmp[i][h * H + j] = f_tmp[h][i][j];
                }
            }
        }
        for (int i = 0; i < DEP; ++i) {
            linear->forward(f_nex_tmp, output);
        }
    }

private:
    Linear<T, DIM, DIM> *linear_q[H], *linear_k[H], *linear_v[H];
    Linear<T, DIM * H, DIM> *linear;
    Dropout<T, DIM> *dropout;
    double scale{};
};

#endif