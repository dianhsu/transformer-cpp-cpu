#ifndef __MODEL_ATTENTION_H__
#define __MODEL_ATTENTION_H__

#include "linear.h"
#include "dropout.h"

template<typename T, int DIM, int H>
class MultiHeadAttention {
    MultiHeadAttention();
    void load_params(T weights_q[H][DIM][DIM],
                     T weights_k[H][DIM][DIM],
                     T weights_v[H][DIM][DIM],
                     T weights2[DEP*H][DIM],
                     T bias_q[H][DIM],
                     T bias_k[H][DIM],
                     T bias_v[H][DIM],
                     T bia2[DIM]);
    void forward(T q_in[DIM], T k_in[DIM], T v_in[DIM], T output[DIM]);
private:
    Linear<T, DIM, DIM> *linear_q[H], *linear_k[H], *linear_v[H];
    Linear<T, DEP*H, DIM> *linear;
    Dropout<T, DIM> *dropout;
    double scale;
};
#endif