#ifndef __MODEL_ENCODER_LAYER_H__
#define __MODEL_ENCODER_LAYER_H__

#include "linear.h"
#include "feedforward.h"
#include "dropout.h"
#include "norm.h"
#include "attention.h"

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class EncoderLayer {
public:
    EncoderLayer();
    void load_params(
        // norm1 params
        T weights_norm1[DIM],
        T bias_norm1[DIM],
        // attention params
        T weights_q[HEAD_SIZE][DIM][DIM],
        T weights_k[HEAD_SIZE][DIM][DIM],
        T weights_v[HEAD_SIZE][DIM][DIM],
        T weights2[DEP*HEAD_SIZE][DIM],
        T bias_q[HEAD_SIZE][DIM],
        T bias_k[HEAD_SIZE][DIM],
        T bias_v[HEAD_SIZE][DIM],
        T bia2[DIM]
        // dropout1 params
        T dropout_rate1,
        // ff params
        T weights1_ff[DIM][D_H],
        T bias1_ff[D_H],
        T weights2_ff[D_H][DIM],
        T bias2_ff[DIM],
        T dropout_rate_ff,
        T weights_norm2[DIM],
        T bias_norm2[DIM],
        // dropout2 params
        T dropout_rate2);
    void forward(T input[DEP][DIM], T output[DEP][DIM]);
private:
    LayerNorm<T, DIM> *norm1;
    MultiHeadAttention<T, DIM, HEAD_SIZE> *attention;
    dropout<T, DIM> *dropout1;
    LayerNorm<T, DIM> *norm2;
    FeedForwardNetwork<T, DIM, DIM, D_H> *ff;
    dropout<T, DIM> *dropout2;
};

#endif