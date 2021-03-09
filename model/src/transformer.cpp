#include "transformer.h"


template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
Transformer<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>::Transformer() {
    encoder = new Encoder<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT>();
    decoder = new Decoder<T, DIM, DEP, D_H, HEAD_SIZE, DEC_LAYER_CNT>();
}

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
Transformer<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>::~Transformer() {
    delete encoder;
    delete decoder;
}
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
void Transformer<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>::load_params(TransformerParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT> *p) {
    if(p != nullptr) {
        encoder->load_params(p->encoder_p);
        decoder->load_params(p->decoder_p);
    }
}
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
void Transformer<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>::forward(T input[DEP][DIM], T output[DEP][DIM]) {
    T tmp[DEP][DIM];
    encoder->forward(input, tmp);
    decoder->forward(tmp, output);
}