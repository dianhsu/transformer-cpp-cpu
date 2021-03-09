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
    long long count(){
        return norm1_p->count() + attention1_p->count() + norm2_p->count()+ attention2_p->count() + norm3_p->count() + ff_p->count();
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class DecoderLayer {
public:
    DecoderLayer() {
        norm1 = new LayerNorm<T, DIM>();
        attention1 = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>();
        dropout1 = new Dropout<T, DIM>();
        norm2 = new LayerNorm<T, DIM>();
        attention2 = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>();
        norm3 = new LayerNorm<T, DIM>();
        ff = new FeedForwardNetwork<T, DIM, DIM, D_H>();
        dropout2 = new Dropout<T, DIM>();
    }

    ~DecoderLayer() {
        delete norm1;
        delete attention1;
        delete dropout1;
        delete norm2;
        delete attention2;
        delete norm3;
        delete ff;
        delete dropout2;
    }

    void load_params(DecoderLayerParam<T, DIM, D_H, HEAD_SIZE> *p) {
        if (p != nullptr) {
            norm1->load_params(p->norm1_p);
            attention1->load_params(p->attention1_p);
            if (p->dropout_rate1 != nullptr) {
                dropout1->load_params(*(p->dropout_rate1));
            }
            norm2->load_params(p->norm2_p);
            attention2->load_params(p->attention2_p);
            norm3->load_params(p->norm3_p);
            ff->load_params(p->ff_p);
            if (p->dropout_rate2 != nullptr) {
                dropout2->load_params(*(p->dropout_rate2));
            }
        }

    }

    void forward(T input[DEP][DIM], T enc_output[DEP][DIM], T output[DEP][DIM]) {
        T tmp[7][DEP][DIM];
        norm1->forward(input, tmp[0]);
        attention1->forward(tmp[0], tmp[0], tmp[0], tmp[1]);
        dropout1->forward(tmp[1], tmp[2]);
        norm2->forward(tmp[2], tmp[3]);
        attention2->forward(tmp[3], enc_output, enc_output, tmp[4]);
        norm3->forward(tmp[4], tmp[5]);
        ff->forward(tmp[5], tmp[6]);
        dropout2->forward(tmp[6], output);
    }

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