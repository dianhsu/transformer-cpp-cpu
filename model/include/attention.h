#ifndef __MODEL_ATTENTION_H__
#define __MODEL_ATTENTION_H__

#include "linear.h"
#include "dropout.h"


template<typename T, int DIM, int H>
struct MultiHeadAttentionParam {
    LinearParam<T, DIM, DIM> *linear_q_p[H], *linear_k_p[H], *linear_v_p[H];
    LinearParam<T, DIM*H, DIM> *linear_p;
    T *dropout_rate;
    MultiHeadAttentionParam() {
        for(int i = 0; i < H; ++i) {
            linear_q_p = new LinearParam<T, DIM, DIM>();
            linear_k_p = new LinearParam<T, DIM, DIM>();
            linear_v_p = new LinearParam<T, DIM, DIM>();
        }
        linear_p = new LinearParam<T, DIM*H, DIM>();
    }
    ~MultiHeadAttentionParam() {
        delete[] linear_q_p;
        delete[] linear_k_p;
        delete[] linear_v_p;
        delete linear_p;
    }
};


template<typename T, int DIM, int DEP, int H>
class MultiHeadAttention {
    MultiHeadAttention();
    void load_params(MultiHeadAttentionParam<T, DIM, H> *p);
    void forward(T q_in[DEP][DIM], T k_in[DEP][DIM], T v_in[DEP][DIM], T output[DEP][DIM]);
private:
    Linear<T, DIM, DIM> *linear_q[H], *linear_k[H], *linear_v[H];
    Linear<T, DIM*H, DIM> *linear;
    Dropout<T, DIM> *dropout;
    double scale;
};
#endif