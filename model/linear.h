//
// Created by dianhsu on 2021/03/10.
//

#ifndef TRANSFORMER_LINEAR_H
#define TRANSFORMER_LINEAR_H

#include <array>


namespace transformer {
    template<typename T, int DIM_IN, int DIM_OUT>
    struct LinearParameter {
        std::array<std::array<T, DIM_OUT>, DIM_IN> weights;
        std::array<T, DIM_OUT> bias;

        long long count() {
            long long ret = 0;
            ret += DIM_OUT * DIM_IN + DIM_OUT;
            return ret;
        }
    };

    template<typename T, int DIM_IN, int DIM_OUT>
    class Linear {
    public:
        static void forward(std::array<T, DIM_IN> &input,
                            std::array<T, DIM_OUT> &output,
                            LinearParameter<T, DIM_IN, DIM_OUT> &param) {
            for (int i = 0; i < DIM_OUT; ++i) {
                output[i] = param.bias[i];
                for (int j = 0; j < DIM_IN; ++j) {
                    output[i] += input[j] * param.weights[j][i];
                }
            }
        }

    };

    template<typename T, int DIM_IN, int DIM_OUT, int DEP>
    class MultiLinear {
    public:
        static void forward(std::array<std::array<T, DIM_IN>, DEP> &input,
                            std::array<std::array<T, DIM_OUT>, DEP> &output,
                            LinearParameter<T, DIM_IN, DIM_OUT> &param) {
            for (int k = 0; k < DEP; ++k) {
                for (int i = 0; i < DIM_OUT; ++i) {
                    output[k][i] = param.bias[i];
                    for (int j = 0; j < DIM_IN; ++j) {
                        output[k][i] += input[k][j] * param.weights[j][i];
                    }
                }
            }
        }
    };
};

#endif //TRANSFORMER_LINEAR_H
