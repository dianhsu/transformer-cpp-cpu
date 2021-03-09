#include "decoder_layer.h"

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>::DecoderLayer() {
    norm1 = new LayerNorm<T, DIM>();
    attention1 = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>();
    dropout1 = new Dropout<T, DIM>();
    norm2 = new LayerNorm<T, DIM>();
    attention2 = new MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>();
    norm3 = new LayerNorm<T, DIM>();
    ff = new FeedForwardNetwork<T, DIM, DIM, D_H>();
    dropout2 = new Dropout<T, DIM>();
}

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>::~DecoderLayer() {
    delete norm1;
    delete attention1;
    delete dropout1;
    delete norm2;
    delete attention2;
    delete norm3;
    delete ff;
    delete dropout2;
}
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
void DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>::load_params(DecoderLayerParam<T, DIM, D_H, HEAD_SIZE> *p) {
    if(p != nullptr) {
        norm1->load_params(p->norm1_p);
        attention1->load_params(p->attention1_p);
        if(p->dropout_rate1 != nullptr) {
            dropout1->load_params(*(p->dropout_rate1));
        }
        norm2->load_params(p->norm2_p);
        attention2->load_params(p->attention2_p);
        norm3->load_params(p->norm3_p);
        ff->load_params(p->ff_p);
        if(p->dropout_rate2 != nullptr) {
            dropout2->load_params(*(p->dropout_rate2));
        }
    }

}

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
void DecoderLayer<T, DIM, DEP, D_H, HEAD_SIZE>::forward(T intput[DEP][DIM], T enc_output[DEP][DIM], T output[DEP][DIM]) {
    T tmp[7][DEP][DIM];
    norm1->forward(input, tmp[0]);
    attention1->forward(tmp[0], tmp[0], tmp[0], tmp[1]);
    dropout1->forward(tmp[1],tmp[2]);
    norm2->forward(tmp[2], tmp[3]);
    attention2->forward(tmp[3], enc_output, enc_output, tmp[4]);
    norm->forward(tmp[4], tmp[5]);
    ff->forward(tmp[5], tmp[6]);
    dropout2->forward(tmp[6], output);
}