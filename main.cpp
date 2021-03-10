
#include "transformer.h"
#include <iostream>
#include <ctime>

typedef float T;
const int DEP = 20;
const int DIM = 512;
const int D_H = 2048;
const int HEAD_SIZE = 8;
const int ENC_LAYER_CNT = 6;
const int DEC_LAYER_CNT = 6;

int main() {
//    auto *param = new TransformerParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>();
//
//    auto *transformer = new Transformer<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>(*param);
    auto *param = new EncoderParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT>();
    auto *encoder = new Encoder<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT>(*param);
    auto *input = new array<array<T, DIM>, DEP>();
    auto *output = new array<array<T, DIM>, DEP>();
//    transformer->forward(*input, *output);
    encoder->forward(*input, *output);
    return 0;
}