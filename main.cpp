#include <bits/stdc++.h>
#include "transformer.h"

typedef float T;
#define DIM 16
#define DEP 20
#define DIM_HID 32
#define HEAD_SIZE 8
#define ENC_LAYER_CNT 6
#define DEC_LAYER_CNT 6


int main() {
    auto *param = new transformer::TransformerParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>();
    std::cout << "parameters count: " << param->count() << std::endl;
    auto *input = new std::array<std::array<T, DIM>, DEP>{};
    auto *output = new std::array<std::array<T, DIM>, DEP>{};
    transformer::Transformer<T, DIM, DEP, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT>::forward(*input, *output,
                                                                                                     *param);
    for (int i = 0; i < DEP; ++i) {
        for (int j = 0; j < DIM; ++j) {
            std::cout << (*output)[i][j] << " ";
        }
        std::cout << std::endl;

    }
    return 0;
}