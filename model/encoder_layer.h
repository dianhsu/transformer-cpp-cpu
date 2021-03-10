#ifndef __MODEL_ENCODER_LAYER_H__
#define __MODEL_ENCODER_LAYER_H__

#include "linear.h"
#include "feedforward.h"
#include "dropout.h"
#include "norm.h"
#include "attention.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE>
struct EncoderLayerParam {
    LayerNormParam<T, DIM> norm1_p;
    MultiHeadAttentionParam<T, DIM, HEAD_SIZE> attn_p;
    T dropout_rate1;
    LayerNormParam<T, DIM> norm2_p;
    FeedForwardNetworkParam<T, DIM, DIM, D_H> ff_p;
    T dropout_rate2;

    EncoderLayerParam() {
        dropout_rate1 = 0.1;
        dropout_rate2 = 0.1;
    }

    long long count() {
        return norm1_p.count() + attn_p.count() + norm2_p.count() + ff_p.count();
    }
};


template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class EncoderLayer {
public:
    explicit EncoderLayer(EncoderLayerParam<T, DIM, D_H, HEAD_SIZE> &p) {
        norm1 = new LayerNorm<T, DIM>(p.norm1_p);
        attention = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>(p.attn_p);
        dropout1 = new Dropout<T, DIM>(p.dropout_rate1);
        norm2 = new LayerNorm<T, DIM>(p.norm2_p);
        ff = new FeedForwardNetwork<T, DIM, DIM, D_H>(p.ff_p);
        dropout2 = new Dropout<T, DIM>(p.dropout_rate2);
    }
    ~EncoderLayer(){
        delete norm1;
        delete attention;
        delete dropout1;
        delete norm2;
        delete ff;
        delete dropout2;
    }
    void forward(array<array<T, DIM>, DEP> &input, array<array<T, DIM>, DEP> &output) {
        array<array<T, DIM>, DEP> i_tmp;
        for (int i = 0; i < DEP; ++i) {
            norm1->forward(input[i], i_tmp[i]);
        }
        array<array<T, DIM>, DEP> nex_tmp;
        attention->forward(i_tmp, i_tmp, i_tmp, nex_tmp);
        for (int i = 0; i < DEP; ++i) {
            dropout1->forward(nex_tmp[i]);
        }
        for (int i = 0; i < DEP; ++i) {
            for (int j = 0; j < DIM; ++j) {
                nex_tmp[i][j] += input[i][j];
            }
        }
        array<array<T, DIM>, DEP> nex2_tmp;
        for (int i = 0; i < DEP; ++i) {
            norm2->forward(nex_tmp[i], nex2_tmp[i]);
        }
        array<array<T, DIM>, DEP> nex3_tmp;
        for (int i = 0; i < DEP; ++i) {
            ff->forward(nex2_tmp[i], nex3_tmp[i]);
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