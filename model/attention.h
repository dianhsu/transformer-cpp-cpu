#ifndef __MODEL_ATTENTION_H__
#define __MODEL_ATTENTION_H__

#include "linear.h"
#include "dropout.h"

template<typename T, int DIM, int H>
class MultiHeadAttention {
    MultiHeadAttention();
    void load_params(T weights1[DIM*3][DIM], T weights2[DIM][DIM]);
    void forward(T q_in[DIM], T k_in[DIM], T v_in[DIM], T output[DIM]);
private:
    T q_w[DIM][DIM], k_w[DIM][DIM], v_w[DIM][DIM];
    T weights[DIM][DIM];
    Linear<T, DIM, DIM> *linear_q, *linear_k, *linear_v, *linear;
    Dropout<T, DIM> *dropout;
    double scale;
};
#endif