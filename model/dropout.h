//
// Created by dianhsu on 2021/03/10.
//

#ifndef TRANSFORMER_DROPOUT_H
#define TRANSFORMER_DROPOUT_H

#include <array>
namespace transformer {
    template<typename T, int DIM>
    class Dropout {
    public:
        static void forward(std::array<T, DIM> &input, std::array<T, DIM> &output, T dropout_rate) {
            for (int i = 0; i < DIM; ++i) {
                if (input[i] < dropout_rate) {
                    output[i] = 0;
                } else {
                    output[i] = input[i];
                }
            }
        }
    };

}
#endif //TRANSFORMER_DROPOUT_H
