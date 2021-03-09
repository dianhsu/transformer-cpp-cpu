#ifndef __MODEL_ENCODER_LAYER_H__
#define __MODEL_ENCODER_LAYER_H__

#include "linear.h"
#include "feedforward.h"
#include "dropout.h"
#include "norm.h"
#include "attention.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE>
struct EncoderLayerParam {
    LayerNormParam<T, DIM> *norm1_p;
    MultiHeadAttentionParam<T, DIM, HEAD_SIZE> *attn_p;
    T *dropout_rate1;
    LayerNormParam<T, DIM> *norm2_p;
    FeedForwardNetworkParam<T, DIM, DIM, D_H> *ff_p;
    T *dropout_rate2;

    EncoderLayerParam() {
        norm1_p = new LayerNormParam<T, DIM>();
        attn_p = new MultiHeadAttentionParam<T, DIM, HEAD_SIZE>();
        norm2_p = new LayerNormParam<T, DIM>();
        ff_p = new FeedForwardNetworkParam<T, DIM, DIM, D_H>();
    }

    ~EncoderLayerParam() {
        delete norm1_p;
        delete attn_p;
        delete norm2_p;
        delete ff_p;
    }
    long long count(){
        return norm1_p->count() + attn_p->count() + norm2_p->count() + ff_p->count();
    }
};


template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class EncoderLayer {
public:
    EncoderLayer() {
        norm1 = new LayerNorm<T, DIM>();
        attention = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>();
        dropout1 = new Dropout<T, DIM>();
        norm2 = new LayerNorm<T, DIM>();
        ff = new FeedForwardNetwork<T, DIM, DIM, D_H>();
        dropout2 = new Dropout<T, DIM>();
    }

    void load_params(EncoderLayerParam<T, DIM, D_H, HEAD_SIZE> *p) {
        if (p != nullptr) {
            norm1->load_params(p->norm1_p);
            attention->load_params(p->attn_p);
            if (p->dropout_rate1 != nullptr) {
                dropout1->load_params(*(p->dropout_rate1));
            }
            norm2->load_params(p->norm2_p);
            ff->load_params(p->ff_p);
            if (p->dropout_rate2 != nullptr) {
                dropout2->load_params(*(p->dropout_rate2));
            }
        }
    }

    void forward(T input[DEP][DIM], T output[DEP][DIM]) {
        T i_tmp[DEP][DIM];
        for (int i = 0; i < DEP; ++i) {
            norm1->forward(input[i], i_tmp[i]);
        }
        T nex_tmp[DEP][DIM];
        attention->forward(i_tmp, i_tmp, i_tmp, nex_tmp);
        for (int i = 0; i < DEP; ++i) {
            dropout1->forward(nex_tmp[i]);
        }
        for (int i = 0; i < DEP; ++i) {
            for (int j = 0; j < DIM; ++j) {
                nex_tmp[i][j] += input[i][j];
            }
        }
        T nex2_tmp[DEP][DIM];
        for (int i = 0; i < DEP; ++i) {
            norm2->forward(nex_tmp[i], nex2_tmp[i]);
        }
        T nex3_tmp[DEP][DIM];
        for (int i = 0; i < DEP; ++i) {
            ff->forward(nex2_tmp, nex3_tmp);
        }
        for (int i = 0; i < DEP; ++i) {
            dropout2->forward(nex3_tmp[i]);
        }
        for (int i = 0; i < DEP; ++i) {
            for (int j = 0; j < DIM; ++j) {
                output[i][j] = input[i][j] + nex3_tmp[i][j];
            }
        }
    }

private:
    LayerNorm<T, DIM> *norm1;
    MultiHeadAttention<T, DIM, DEP, HEAD_SIZE> *attention;
    Dropout<T, DIM> *dropout1;
    LayerNorm<T, DIM> *norm2;
    FeedForwardNetwork<T, DIM, DIM, D_H> *ff;
    Dropout<T, DIM> *dropout2;
};

#endif