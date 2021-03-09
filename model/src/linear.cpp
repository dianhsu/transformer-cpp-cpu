#include "linear.h"
#include <cstring>

template<typename T, int D_I, int D_O>
Linear::Linear() {
}

template<typename T, int D_I, int D_O>
Linear::~Linear() {
}

template<typename T, int D_I, int D_O>
void Linear::load_params(LinearParam<T, D_I, D_O> *p) {
    if(p != nullptr) {
        this->params = p;
    }
}
template<typename T, int D_I, int D_O>
void Linear::forward(T input[D_I], T output[D_O]) {
    memcpy(output, this->params->bias, sizeof(T)*D_O);
    for(int j = 0; j < D_O; ++j) {
        for(int i = 0; i < D_I; ++i) {
            output[j] += input[i] * this->params->weights[i][j];
        }
    }
}

