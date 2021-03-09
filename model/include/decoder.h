#ifndef __MODEL_DECODER_H__
#define __MODEL_DECODER_H__

#include "decoder_layer.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE, int LAYER_CNT>
struct DecoderParam{
    DecoderLayerParam<T, DIM, D_H, HEAD_SIZE> *layers_p[LAYER_CNT];
    LayerNormParam<T, DIM> *norm_p;
    DecoderParam(){
        for(int i = 0; i < LAYER_CNT; ++i){
            layers_p[i] = new DecoderLayerParam<T, DIM, D_H, HEAD_SIZE>();
        }
        norm_p = new LayerNormParam<T, DIM>();
    }
    ~DecoderParam(){
        for(int i = 0; i < LAYER_CNT; ++i){
            delete layers_p[i];
        }
        delete norm_p;
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
class Decoder {
public:
    Decoder();
    ~Decoder();
    void load_params(DecoderParam<T, DIM, D_H, HEAD_SIZE, LAYER_CNT> *p);
    void forward(T input[DEP][DIM], T output[DEP][DIM]);
private:
    DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE> *layers[LAYER_CNT];
    LayerNorm<T, DIM> *norm;
};

#endif