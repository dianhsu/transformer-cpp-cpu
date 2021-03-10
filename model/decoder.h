#ifndef __MODEL_DECODER_H__
#define __MODEL_DECODER_H__

#include "decoder_layer.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE, int LAYER_CNT>
struct DecoderParam {
    DecoderLayerParam<T, DIM, D_H, HEAD_SIZE> layers_p[LAYER_CNT];
    LayerNormParam<T, DIM> norm_p;


    long long count() {
        return layers_p[0].count() * LAYER_CNT + norm_p.count();
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
class Decoder {
public:
    explicit Decoder(DecoderParam<T, DIM, D_H, HEAD_SIZE, LAYER_CNT> &p) {
        for (int i = 0; i < LAYER_CNT; ++i) {
            layers[i] = new DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>(p.layers_p[i]);
        }
        norm = new LayerNorm<T, DIM>(p.norm_p);
    }

    ~Decoder() {
        for (int i = 0; i < LAYER_CNT; ++i) {
            delete layers[i];
        }
        delete norm;
    }


    void forward(const array<array<T, DIM>, DEP> input,
                 const array<array<T, DIM>, DEP> enc_output,
                 array<array<T, DIM>, DEP> &output) {
        auto tmp = array<array<array<T, DIM>, DEP>, LAYER_CNT>{};
        for (int i = 0; i < LAYER_CNT; ++i) {
            if (i == 0) {
                layers[0]->forward(input, enc_output, tmp[0]);
            } else {
                layers[i]->forward(tmp[i - 1], enc_output, tmp[i]);
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