#ifndef __MODEL_ATTENTION_H__
#define __MODEL_ATTENTION_H__

#include "linear.h"
#include "dropout.h"

template<typename T, int DIM>
class MultiHeadAttention {
    MultiHeadAttention(int head_size);
    void load_params(T weights[DIM*3][DIM], T weights[DIM][DIM]);
    void forward(T q_in[DIM], T k_in[DIM], T v_in[DIM], T output[DIM]);
private:
    T q_w[DIM][DIM], k_w[DIM][DIM], v_w[DIM][DIM];
    T weights[DIM][DIM];
    Linear *linear_q, *linear_k, *linear_v;
    Linear *linear;
    Dropout *dropout;
    int head_size;
    double scale;
};
#endif