#ifndef __MODEL_TRANSFORMER_H__
#define __MODEL_TRANSFORMER_H__

#include "encoder.h"
#include "decoder.h"


template<typename T, int DIM, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
struct TransformerParam {
    EncoderParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT> encoder_p;
    DecoderParam<T, DIM, D_H, HEAD_SIZE, DEC_LAYER_CNT> decoder_p;

    long long count() {
        return encoder_p.count() + decoder_p.count();
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
class Transformer {
public:
    explicit Transformer(TransformerParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT> &p) {
        encoder = new Encoder<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT>(p.encoder_p);
        decoder = new Decoder<T, DIM, DEP, D_H, HEAD_SIZE, DEC_LAYER_CNT>(p.decoder_p);
    }

    ~Transformer() {
        delete encoder;
        delete decoder;
    }


    void forward(const array<array<T, DIM>, DEP> &input, array<array<T, DIM>, DEP> &output) {
        array<array<T, DIM>, DEP> tmp;
        encoder->forward(input, tmp);
        decoder->forward(tmp, tmp, output);
    }

private:
    Encoder<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT> *encoder;
    Decoder<T, DIM, DEP, D_H, HEAD_SIZE, DEC_LAYER_CNT> *decoder;
};

#endif