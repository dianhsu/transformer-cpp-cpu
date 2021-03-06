#include "linear.h"
#include <cstring>

template<typename T, int D_I, int D_O>
void Linear::load_params(T weight[D_I][D_O]) {
    memcpy(this->weight, weight, sizeof(T)*D_I*D_O);
    memcpy(this->bias, bias, sizeof(T)*D_O);
}
template<typename T, int D_I, int D_O>
void Linear::forward(T input[D_I], T output[D_O]) {
    memcpy(output, this->bias, sizeof(this->bias));
    for(int j = 0; j < D_O; ++j) {
        for(int i = 0; i < D_I; ++i) {
            output[j] += input[i] * weight[i][j];
        }
    }
}

