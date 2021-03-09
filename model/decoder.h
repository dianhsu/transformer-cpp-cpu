#ifndef __MODEL_DECODER_H__
#define __MODEL_DECODER_H__

#include "decoder_layer.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE, int LAYER_CNT>
struct DecoderParam {
    DecoderLayerParam<T, DIM, D_H, HEAD_SIZE> *layers_p[LAYER_CNT];
    LayerNormParam<T, DIM> *norm_p;

    DecoderParam() {
        for (int i = 0; i < LAYER_CNT; ++i) {
            layers_p[i] = new DecoderLayerParam<T, DIM, D_H, HEAD_SIZE>();
        }
        norm_p = new LayerNormParam<T, DIM>();
    }

    ~DecoderParam() {
        for (int i = 0; i < LAYER_CNT; ++i) {
            delete layers_p[i];
        }
        delete norm_p;
    }
    long long count(){
        return layers_p[0]->count() * LAYER_CNT + norm_p->count();
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
class Decoder {
public:
    Decoder() {
        for (int i = 0; i < LAYER_CNT; ++i) {
            layers[i] = new DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>();
        }
        norm = new LayerNorm<T, DIM>();
    }

    ~Decoder() {
        for (int i = 0; i < LAYER_CNT; ++i) {
            delete layers[i];
        }
        delete norm;
    }

    void load_params(DecoderParam<T, DIM, D_H, HEAD_SIZE, LAYER_CNT> *p) {
        if (p != nullptr) {
            for (int i = 0; i < LAYER_CNT; ++i) {
                layers[i]->load_params(p->layers_p[i]);
            }
            norm->load_params(p->norm_p);
        }
    }

    void forward(T input[DEP][DIM], T output[DEP][DIM]) {
        T tmp[LAYER_CNT][DEP][DIM];
        for (int i = 0; i < LAYER_CNT; ++i) {
            if (i == 0) {
                layers[0]->forward(input, tmp[0]);
            } else {
                layers[i]->forward(tmp[i - 1], tmp[i]);
            }
        }
        for (int i = 0; i < DEP; ++i) {
            norm->forward(tmp[LAYER_CNT - 1][i], output[i]);
        }

    }

private:
    DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE> *layers[LAYER_CNT];
    LayerNorm<T, DIM> *norm;
};

#endif