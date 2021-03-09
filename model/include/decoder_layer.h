#ifndef __MODEL_DECODER_LAYER_H__
#define __MODEL_DECODER_LAYER_H__

#include "norm.h"
#include "attention.h"
#include "dropout.h"
#include "feedforward.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE>
struct DecoderLayerParam {
    LayerNormParam<T, DIM> *norm1_p;
    MultiHeadAttentionParam<T, DIM, HEAD_SIZE> *attention1_p;
    T *dropout_rate1;
    LayerNormParam<T, DIM> *norm2_p;
    MultiHeadAttentionParam<T, DIM, HEAD_SIZE> *attention2_p;
    LayerNormParam<T, DIM> *norm3_p;
    FeedForwardNetworkParam<T, DIM, DIM, D_H> *ff_p;
    T *dropout_rate2;
    DecoderLayerParam() {
        norm1_p = new LayerNormParam<T, DIM>();
        attention1_p = new MultiHeadAttentionParam<T, DIM, HEAD_SIZE>();
        norm2_p = new LayerNormParam<T, DIM>();
        attention2_p = new MultiHeadAttentionParam<T, DIM, HEAD_SIZE>();
        norm3_p = new LayerNormParam<T, DIM>();
        ff_p = new FeedForwardNetworkParam<T, DIM, DIM, D_H>();
    }
    ~DecoderLayerParam() {
        delete norm1_p;
        delete attention1_p;
        delete norm2_p;
        delete attention2_p;
        delete norm3_p;
        delete ff_p;
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class DecoderLayer {
public:
    DecoderLayer();
    ~DecoderLayer();
    void load_params(DecoderLayerParam<T, DIM, D_H, HEAD_SIZE> *p);
    void forward(T input[DEP][DIM], T enc_output[DEP][DIM], T output[DEP][DIM]);
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