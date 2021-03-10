#ifndef __MODEL_LINEAR_H__
#define __MODEL_LINEAR_H__

#include <array>

using std::array;

template<typename T, int D_I, int D_O>
struct LinearParam {
    array<array<T, D_O>, D_I> weights;
    array<T, D_O> bias;

    long long count() {
        long long ret = 0;
        ret += D_I * D_O + D_O;
        return ret;
    }
};

template<typename T, int D_I, int D_O>
class Linear {
public:
    explicit Linear(LinearParam<T, D_I, D_O> &p) {
        this->params = &p;
    }

    void forward(array<T, D_I> &input, array<T, D_O> &output) {
        for (int j = 0; j < D_O; ++j) {
            output[j] = this->params->bias[j];
            for (int i = 0; i < D_I; ++i) {
                output[j] += input[i] * this->params->weights[i][j];
            }
        }
    }

private:
    LinearParam<T, D_I, D_O> *params;
};

#endif