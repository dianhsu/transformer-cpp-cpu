//
// Created by dianhsu on 2021/03/10.
//

#ifndef TRANSFORMER_RELU_H
#define TRANSFORMER_RELU_H

#include <array>

namespace transformer {
    template<typename T, int DIM>
    class Relu {
    public:
        static void forward(std::array<T, DIM> &input, std::array<T, DIM> &output) {
            for (int i = 0; i < DIM; ++i) {
                if (input[i] < 0) {
                    output[i] = 0;
                } else {
                    output[i] = input[i];
                }
            }
        }
    };
}
#endif //TRANSFORMER_RELU_H
