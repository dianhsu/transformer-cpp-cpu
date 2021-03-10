
#include "transformer.h"
#include <iostream>

typedef float T;
const int DEP = 5;
const int DIM = 32;
const int D_H = 32;
const int HEAD_SIZE = 2;
const int ENC_LAYER_CNT = 1;
const int DEC_LAYER_CNT = 1;

int main() {

    auto *p_tran = new TransformerParam<T, DIM, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>();
    auto *transformer = new Transformer<T, DIM, DEP, D_H, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>(*p_tran);
    auto *input = new array<array<T, DIM>, DEP>();
    auto *output = new array<array<T, DIM>, DEP>();
    transformer->forward(*input, *output);
    delete output;
    delete input;
    delete transformer;
    delete p_tran;
    return 0;
}