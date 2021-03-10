#ifndef __MODEL_LAYERNORM_H__
#define __MODEL_LAYERNORM_H__

#include <cstring>
#include <cmath>
#include <array>

using std::array;

template<typename T, int DIM>
struct LayerNormParam {
    array<T, DIM> weights;
    array<T, DIM> bias;

    long long count() {
        return DIM * 2;
    }
};

template<typename T, int DIM>
class LayerNorm {
public:
    explicit LayerNorm(LayerNormParam<T, DIM> &p) {
        this->params = &p;
    }

    void forward(array<T, DIM> &input, array<T, DIM> &output) {
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
            output[i] = (input[i] - avg) / sq_var * params->weights[i] + params->bias[i];
        }
    }

private:
    LayerNormParam<T, DIM> *params;

};


#endif