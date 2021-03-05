#include "linear.h"
#include <cstring>

template<typename T, int D_I, int D_O>
void load_params(T weight[D_I][D_O],T bias[D_O]) {
    memcpy(this->weight, weight, sizeof(weight));
    memcpy(this->bias, bias, sizeof(bias));
}


template<typename T, int D_I, int D_O>
void forward(T input[D_I], T output[D_O]) {
    memcpy(output, this->bias, sizeof(this->bias));
    for(int j = 0; j < D_O; ++j) {
        for(int i = 0; i < D_I; ++i) {
            output[j] += input[i] * weight[i][j];
        }
    }
}

template<typename T, int D_I, int D_O>
void Linear::load_params(T weight[D_I][D_O]) {
    memcpy(this->weight, weight, sizeof(weight));
}
template<typename T, int D_I, int D_O>
void Linear::forward(T input[D_I], T output[D_O]) {
    for(int j = 0; j < D_O; ++j) {
        output[j] = 0;
        for(int i = 0; i < D_I; ++i) {
            output[j] += input[i] * weight[i][j];
        }
    }
}

