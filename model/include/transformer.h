#ifndef __MODEL_TRANSFORMER_H__
#define __MODEL_TRANSFORMER_H__

#include "encoder.h"
#include "decoder.h"


template<typename T, int DIM, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
struct TransformerParam{
    EncoderParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT> *encoder_p;
    DecoderParam<T, DIM, D_H, HEAD_SIZE, DEC_LAYER_CNT> *decoder_p;
    TransformerParam(){
        encoder_p = new EncoderParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT>();
        decoder_p = new DecoderParam<T, DIM, D_H, HEAD_SIZE, DEC_LAYER_CNT>();
    }
    ~TransformerParam(){
        delete encoder_p;
        delete decoder_p;
    }
};
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
class Transformer{
public:
    Transformer();
    ~Transformer();
    void load_params(TransformerParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT> *p);
    void forward(T input[DEP][DIM], T output[DEP][DIM]);
private:
    Encoder<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT> *encoder;
    Decoder<T, DIM, DEP, D_H, HEAD_SIZE, DEC_LAYER_CNT> *decoder;
};
#endif