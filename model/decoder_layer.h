#ifndef __MODEL_DECODER_LAYER_H__
#define __MODEL_DECODER_LAYER_H__

#include "norm.h"
#include "attention.h"
#include "dropout.h"
#include "feedforward.h"

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
struct DecoderLayerParam{
    LayerNormParam<T, DIM> *norm1_p;
    MultiHeadAttentionParam<T, DIM, DEP, HEAD_SIZE> *attention1_p;
    T *dropout_rate1;
    LayerNormParam<T, DIM> *norm2_p;
    MultiHeadAttentionParam<T, DIM, DEP, HEAD_SIZE> *attention2_p;
    LayerNormParam<T, DIM> *norm3_p;
    FeedForwardNetwork<T, DIM, DIM, D_H> *ff_p;
    T *dropout_rate2;
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class DecoderLayer {
public:
    DecoderLayer();
    ~DecoderLayer();
    void load_params(DecoderLayerParam<T, DIM, DEP, D_H, HEAD_SIZE> *p);
    void forward();
private:
    LayerNorm<T, DIM> *norm1;
    MultiHeadAttention<T, DIM, DEP, HEAD_SIZE> *attention1;
    Dropout<T, DIM> *dropout1;
    LayerNorm<T, DIM> *norm2;
    MultiHeadAttention<T, DIM, DEP, HEAD_SIZE> *attention2;
    LayerNorm<T, DIM> *norm3;
    FeedForwardNetwork<T, DIM, DIM, D_H> *ff;
    Dropout<T, DIM> *dropout2;
};

#endif