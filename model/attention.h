#ifndef __MODEL_ATTENTION_H__
#define __MODEL_ATTENTION_H__

#include "linear.h"
#include "dropout.h"

template<typename T, int DIM, int DEP, int H>
struct MultiHeadAttentionParam{
    LinearParam<T, DIM, DIM> *linear_q_p[H], *linear_k_p[H], *linear_v_p[H];
    LinearParam<T, DEP*H, DIM> *linear_p;
    T *dropout_rate;
};


template<typename T, int DIM, int DEP, int H>
class MultiHeadAttention {
    MultiHeadAttention();
    void load_params(MultiHeadAttentionParam<T, DIM, DEP, H> *p);
    void forward(T q_in[DIM], T k_in[DIM], T v_in[DIM], T output[DIM]);
private:
    Linear<T, DIM, DIM> *linear_q[H], *linear_k[H], *linear_v[H];
    Linear<T, DEP*H, DIM> *linear;
    Dropout<T, DIM> *dropout;
    double scale;
};
#endif