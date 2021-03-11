#include <bits/stdc++.h>
#include "transformer.h"

typedef float T;
// Embedding的维度
#define DIM 16  
// 句子长度
#define DEP 20
// FeedForwardNetwork中隐藏层宽度
#define DIM_HID 32
// MultiHeadAttention中Head的数量
#define HEAD_SIZE 8
// Encoder 的层数
#define ENC_LAYER_CNT 6
// Decoder 的层数
#define DEC_LAYER_CNT 6

/**
参考其他模型中的参数 DIM: 512, DEP: 20, DIM_HID: 2048, HEAD_SIZE: 8, ENC_LAYER_CNT: 6, DEC_LAYER_CNT: 6
最终统计：参数量: 176454656
CPU_TIME: 127080 ms
REAL_TIME: 127243 ms
Memory: 711688192 bytes
**/

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
