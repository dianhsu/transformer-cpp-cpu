#ifndef __MODEL_ATTENTION_H__
#define __MODEL_ATTENTION_H__

#include <cmath>

#include "linear.h"
#include "dropout.h"
#include "function.h"

template<typename T, int DIM, int H>
struct MultiHeadAttentionParam {
    LinearParam<T, DIM, DIM> linear_q_p[H], linear_k_p[H], linear_v_p[H];
    LinearParam<T, DIM * H, DIM> linear_p;
    T dropout_rate;

    MultiHeadAttentionParam() {
        dropout_rate = 0.1;
    }

    long long count() {
        return linear_k_p[0].count() * H * 3 + linear_p.count();
    }
};


template<typename T, int DIM, int DEP, int H>
class MultiHeadAttention {
public:
    explicit MultiHeadAttention(MultiHeadAttentionParam<T, DIM, H> &p) {
        for (int i = 0; i < H; ++i) {
            linear_q[i] = new Linear<T, DIM, DIM>(p.linear_q_p[i]);
            linear_k[i] = new Linear<T, DIM, DIM>(p.linear_k_p[i]);
            linear_v[i] = new Linear<T, DIM, DIM>(p.linear_v_p[i]);
        }
        linear = new Linear<T, DIM * H, DIM>(p.linear_p);
        dropout = new Dropout<T, DIM>(p.dropout_rate);
        this->scale = 1.0 / sqrt((DIM / H) * 1.0);
    }
    ~MultiHeadAttention(){
        for(int i = 0; i < H; ++i){
            delete linear_q[i];
            delete linear_k[i];
            delete linear_v[i];
        }
        delete linear;
        delete dropout;
    }
    void forward(const array<array<T, DIM>, DEP> &q_in,
                 const array<array<T, DIM>, DEP> &k_in,
                 const array<array<T, DIM>, DEP> &v_in,
                 array<array<T, DIM>, DEP> &output) {
        auto *q_tmp = new array<array<array<T, DIM>, DEP>, H>();
        auto *k_tmp = new array<array<array<T, DIM>, DEP>, H>();
        auto *v_tmp = new array<array<array<T, DIM>, DEP>, H>();
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < DEP; ++j) {
                linear_q[i]->forward(q_in[j], (*q_tmp)[i][j]);
                linear_k[i]->forward(k_in[j], (*k_tmp)[i][j]);
                linear_v[i]->forward(v_in[j], (*v_tmp)[i][j]);
                dropout->forward((*q_tmp)[i][j]);
                dropout->forward((*k_tmp)[i][j]);
                dropout->forward((*v_tmp)[i][j]);
                for (int k = 0; k < DIM; ++k) {
                    (*q_tmp)[i][j][k] *= this->scale;
                }
            }
        }
        // Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        auto *nex_tmp = new array<array<array<T, DEP>, DEP>, H>();
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DEP; ++j) {
                    (*nex_tmp)[h][i][j] = 0;
                    for (int k = 0; k < DIM; ++k) {
                        (*nex_tmp)[h][i][j] += (*q_tmp)[h][i][k] * (*k_tmp)[h][j][k];
                    }
                }
            }
            softmax<T, DEP, DEP>((*nex_tmp)[h]);
        }
        auto *f_tmp = new array<array<array<T, DIM>, DEP>, H>();
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    (*f_tmp)[h][i][j] = 0;
                    for (int k = 0; k < DEP; ++k) {
                        (*f_tmp)[h][i][j] += (*nex_tmp)[h][i][k] * (*v_tmp)[h][j][k];
                    }
                }
            }
        }
        // Concat
        auto *f_nex_tmp = new array<array<T, DIM * H>, DEP>();
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    (*f_nex_tmp)[i][h * H + j] = (*f_tmp)[h][i][j];
                }
            }
        }
        for (int i = 0; i < DEP; ++i) {
            linear->forward((*f_nex_tmp)[i], output[i]);
        }
        delete q_tmp;
        delete k_tmp;
        delete v_tmp;
        delete f_tmp;
        delete f_nex_tmp;
        delete nex_tmp;
    }

private:
    Linear<T, DIM, DIM> *linear_q[H], *linear_k[H], *linear_v[H];
    Linear<T, DIM * H, DIM> *linear;
    Dropout<T, DIM> *dropout;
    double scale{};
};

#endif