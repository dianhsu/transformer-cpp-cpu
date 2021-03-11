//
// Created by dianhsu on 2021/03/10.
//

#ifndef TRANSFORMER_NORM_H
#define TRANSFORMER_NORM_H

#include <array>

template<typename T, int DIM>
struct LayerNormParameter {
    std::array<T, DIM> weights;
    std::array<T, DIM> bias;

    long long count() {
        return DIM * 2;
    }
};

template<typename T, int DIM>
class LayerNorm {
public:
    static void forward(std::array<T, DIM> &input, std::array<T, DIM> &output, LayerNormParameter<T, DIM> &p) {
        T sum = 0;
        T sum2 = 0;
        for (int i = 0; i < DIM; ++i) {
            sum += input[i];
            sum2 += input[i] * input[i];
        }
        T avg = sum / DIM;
        T avg2 = sum2 / DIM;
        T var = avg2 - avg * avg;
        T sq_var = sqrt(var + 1e-5);
        for (int i = 0; i < DIM; ++i) {
            output[i] = (input[i] - avg) / sq_var * p.weights[i] + p.bias[i];
        }
    }
};

#endif //TRANSFORMER_NORM_H
