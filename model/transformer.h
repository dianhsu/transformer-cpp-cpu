#ifndef __MODEL_TRANSFORMER_H__
#define __MODEL_TRANSFORMER_H__

#include "encoder.h"
#include "decoder.h"


template<typename T, int DIM, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
struct TransformerParam {
    EncoderParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT> *encoder_p;
    DecoderParam<T, DIM, D_H, HEAD_SIZE, DEC_LAYER_CNT> *decoder_p;

    TransformerParam() {
        encoder_p = new EncoderParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT>();
        decoder_p = new DecoderParam<T, DIM, D_H, HEAD_SIZE, DEC_LAYER_CNT>();
    }

    ~TransformerParam() {
        delete encoder_p;
        delete decoder_p;
    }
    long long count(){
        return encoder_p->count() + decoder_p->count();
    }
};

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
class Transformer {
public:
    Transformer() {
        encoder = new Encoder<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT>();
        decoder = new Decoder<T, DIM, DEP, D_H, HEAD_SIZE, DEC_LAYER_CNT>();
    }

    ~Transformer() {
        delete encoder;
        delete decoder;
    }

    void load_params(TransformerParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT> *p) {
        if (p != nullptr) {
            encoder->load_params(p->encoder_p);
            decoder->load_params(p->decoder_p);
        }
    }

    void forward(T input[DEP][DIM], T output[DEP][DIM]) {
        T tmp[DEP][DIM];
        encoder->forward(input, tmp);
        decoder->forward(tmp, output);
    }

private:
    Encoder<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT> *encoder;
    Decoder<T, DIM, DEP, D_H, HEAD_SIZE, DEC_LAYER_CNT> *decoder;
};

#endif