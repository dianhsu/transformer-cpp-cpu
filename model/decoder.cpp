#include "decoder.h"

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
Decoder::Decoder() {
    for(int i = 0; i < LAYER_CNT; ++i) {
        layers[i] = new DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>();
    }
    norm = new LayerNorm<T, DIM>();
}
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
~Decoder::Decoder() {
    for(int i = 0; i < LAYER_CNT; ++i) {
        delete layers[i];
    }
    delete norm;
}
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
void Decoder::load_params(DecoderParam<T, DIM, DEP, D_H, HEAD_SIZE, LAYER_CNT> *p) {
    if(p != nullptr) {
        for(int i = 0; i < LAYER_CNT; ++i) {
            layers[i]->load_params(
                p->layers_p[i]
            );
        }
        norm->load_params(p->norm_p);
    }
}

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int LAYER_CNT>
void Decoder::forward(T input[DEP][DIM], T output[DEP][DIM]) {
    T tmp[LAYER_CNT][DEP][DIM];
    for(int i = 0; i < LAYER_CNT; ++i) {
        if(i == 0) {
            layers[0]->forward(input, tmp[0]);
        } else {
            layers[i]->forward(tmp[i-1], tmp[i]);
        }
    }
    for(int i = 0; i < DEP; ++i) {
        norm->forward(tmp[LAYER_CNT-1][i], output[i]);
    }
}