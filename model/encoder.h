#ifndef __MODEL_ENCODER_H__
#define __MODEL_ENCODER_H__

#include "encoder_layer.h"
#include "norm.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE, int LAYER_CNT>
struct EncoderParam {
    array<EncoderLayerParam<T, DIM, D_H, HEAD_SIZE>, LAYER_CNT> layers_p;
    LayerNormParam<T, DIM> norm_p;


    long long count() {
        return layers_p[0].count() * LAYER_CNT + norm_p.count();
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
class Encoder {
public:
    explicit Encoder(EncoderParam<T, DIM, D_H, HEAD_SIZE, LAYER_CNT> &p) {
        for (int i = 0; i < LAYER_CNT; ++i) {
            layers[i] = new EncoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>(p.layers_p[i]);
        }
        norm = new LayerNorm<T, DIM>(p.norm_p);
    }

    ~Encoder() {
        for (int i = 0; i < LAYER_CNT; ++i) {
            delete layers[i];
        }
        delete norm;
    }

    void forward(const array<array<T, DIM>, DEP> input, array<array<T, DIM>, DEP> &output) {
        auto *tmp = new array<array<array<T, DIM>, DEP>, LAYER_CNT>{};
        for (int i = 0; i < LAYER_CNT; ++i) {
            if (i == 0) {
                layers[0]->forward(input, (*tmp)[0]);
            } else {
                layers[i]->forward((*tmp)[i - 1], (*tmp)[i]);
            }
        }
        for (int i = 0; i < DEP; ++i) {
            norm->forward((*tmp)[i - 1][i], output[i]);
        }
        delete tmp;
    }

private:
    EncoderLayer<T, DIM, DEP, D_H, HEAD_SIZE> *layers[LAYER_CNT];
    LayerNorm<T, DIM> *norm;
};

#endif