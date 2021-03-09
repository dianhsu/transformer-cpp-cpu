#ifndef __MODEL_ENCODER_LAYER_H__
#define __MODEL_ENCODER_LAYER_H__

#include "linear.h"
#include "feedforward.h"
#include "dropout.h"
#include "norm.h"
#include "attention.h"

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
struct EncoderLayerParam {
    LayerNormParam<T, DIM> *norm1_p;
    MultiHeadAttentionParam<T, DIM, DEP, HEAD_SIZE> *attn_p;
    T *dropout_rate1;
    LayerNormParam<T, DIM> *norm2_p;
    FeedForwardNetworkParam<T, DIM, DIM, D_H> *ff_p;
    T *dropout_rate2;
};


template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class EncoderLayer {
public:
    EncoderLayer();
    void load_params(EncoderLayerParam<T, DIM, DEP, D_H, HEAD_SIZE> *p);
    void forward(T input[DEP][DIM], T output[DEP][DIM]);
private:
    LayerNorm<T, DIM> *norm1;
    MultiHeadAttention<T, DIM, DEP, HEAD_SIZE> *attention;
    Dropout<T, DIM> *dropout1;
    LayerNorm<T, DIM> *norm2;
    FeedForwardNetwork<T, DIM, DIM, D_H> *ff;
    Dropout<T, DIM> *dropout2;
};

#endif