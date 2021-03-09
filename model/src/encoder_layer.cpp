#include "encoder_layer.h"

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
EncoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>::EncoderLayer() {
    norm1 = new LayerNorm<T, DIM>();
    attention = new MultiHeadAttention<T, DIM, HEAD_SIZE>();
    dropout1 = new Dropout<T, DIM>();
    norm2 = new LayerNorm<T, DIM>();
    ff = new FeedForwardNetwork<T, DIM, DIM, D_H>();
    dropout2 = new Dropout<T, DIM>();
}
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
void EncoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>::load_params(EncoderLayerParam<T, DIM, D_H, HEAD_SIZE> *p) {
    if(p != nullptr) {
        norm1->load_params(p->norm1_p);
        attention->load_params(p->attn_p);
        if(p->dropout_rate1 != nullptr) {
            dropout1->load_params(*(p->dropout_rate1));
        }
        norm2->load_params(p->norm2_p);
        ff->load_params(p->ff_p);
        if(p->dropout_rate2 != nullptr) {
            dropout2->load_params(*(p->dropout_rate2));
        }
    }
}

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
void EncoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>::forward(T input[DEP][DIM], T output[DEP][DIM]) {
    T i_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i) {
        norm1->forward(input[i], i_tmp[i]);
    }
    T nex_tmp[DEP][DIM];
    attention->forward(i_tmp, i_tmp, i_tmp, nex_tmp);
    for(int i = 0; i < DEP; ++i) {
        dropout1->forward(nex_tmp[i]);
    }
    for(int i = 0; i < DEP; ++i) {
        for(int j = 0; j < DIM; ++j) {
            nex_tmp[i][j] += input[i][j];
        }
    }
    T nex2_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i) {
        norm2->forward(nex_tmp[i], nex2_tmp[i]);
    }
    T nex3_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i) {
        ff->forward(nex2_tmp, nex3_tmp);
    }
    for(int i = 0; i < DEP; ++i) {
        dropout2->forward(nex3_tmp[i]);
    }
    for(int i = 0; i < DEP; ++i) {
        for(int j = 0; j < DIM; ++j) {
            output[i][j] = input[i][j] + nex3_tmp[i][j];
        }
    }
}