#ifndef __MODEL_ENCODER_H__
#define __MODEL_ENCODER_H__

#include "encoder_layer.h"
#include "norm.h"
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
struct EncoderParam {
    EncoderLayerParam<T, DIM, DEP, D_H, HEAD_SIZE> *layers_p[LAYER_CNT];
    LayerNormParam<T, DIM> *norm_p;
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
class Encoder {
public:
    Encoder();
    void load_params(EncoderParam<T, DIM, DEP, D_H, HEAD_SIZE, LAYER_CNT> *p);
    void forward(T input[DEP][DIM], T output[DEP][DIM]);
private:
    EncoderLayer<T, DIM, DEP, D_H, HEAD_SIZE> *layers[LAYER_CNT];
    LayerNorm<T, DIM> *norm;
};

#endif