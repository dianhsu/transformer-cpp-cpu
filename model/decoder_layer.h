#ifndef __MODEL_DECODER_LAYER_H__
#define __MODEL_DECODER_LAYER_H__

#include "norm.h"
#include "attention.h"
#include "dropout.h"
#include "feedforward.h"

template<typename T, int DIM, int D_H, int HEAD_SIZE>
struct DecoderLayerParam {
    LayerNormParam<T, DIM> norm1_p;
    MultiHeadAttentionParam<T, DIM, HEAD_SIZE> attention1_p;
    T dropout_rate1;
    LayerNormParam<T, DIM> norm2_p;
    MultiHeadAttentionParam<T, DIM, HEAD_SIZE> attention2_p;
    LayerNormParam<T, DIM> norm3_p;
    FeedForwardNetworkParam<T, DIM, DIM, D_H> ff_p;
    T dropout_rate2;

    DecoderLayerParam() {
        dropout_rate1 = 0.1;
        dropout_rate2 = 0.1;
    }

    long long count() {
        return norm1_p.count() + attention1_p.count() + norm2_p.count() + attention2_p.count() + norm3_p.count() +
               ff_p.count();
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
class DecoderLayer {
public:
    explicit DecoderLayer(DecoderLayerParam<T, DIM, D_H, HEAD_SIZE> &p) {
        norm1 = new LayerNorm<T, DIM>(p.norm1_p);
        attention1 = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>(p.attention1_p);
        dropout1 = new Dropout<T, DIM>(p.dropout_rate1);
        norm2 = new LayerNorm<T, DIM>(p.norm2_p);
        attention2 = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>(p.attention2_p);
        norm3 = new LayerNorm<T, DIM>(p.norm3_p);
        ff = new FeedForwardNetwork<T, DIM, DIM, D_H>(p.ff_p);
        dropout2 = new Dropout<T, DIM>(p.dropout_rate2);
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


    void forward(const array<array<T, DIM>, DEP> input,
                 const array<array<T, DIM>, DEP> enc_output,
                 array<array<T, DIM>, DEP> &output) {
        auto tmp = array<array<array<T, DIM>, DEP>, 7>{};
        for (int i = 0; i < DEP; ++i) {
            norm1->forward(input.at(i), tmp.at(0).at(i));
        }
        attention1->forward(tmp.at(0), tmp.at(0), tmp.at(0), tmp.at(0));
        for (int i = 0; i < DEP; ++i) {
            dropout1->forward(tmp.at(1).at(i), tmp.at(2).at(i));
        }
        for (int i = 0; i < DEP; ++i) {
            norm2->forward(tmp.at(2).at(i), tmp.at(3).at(i));
        }
        attention2->forward(tmp.at(3), enc_output, enc_output, tmp.at(4));
        for (int i = 0; i < DEP; ++i) {
            norm3->forward(tmp.at(4).at(i), tmp.at(5).at(i));
        }
        for (int i = 0; i < DEP; ++i) {
            ff->forward(tmp.at(5).at(i), tmp.at(6).at(i));
        }
        for (int i = 0; i < DEP; ++i) {
            dropout2->forward(tmp.at(6).at(i), output.at(i));
        }
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