#include "linear.h"
#include <cstring>

template<typename T, int D_I, int D_O>
Linear::Linear() {
    params = new LinearParams<T, D_I, D_O>();
}

template<typename T, int D_I, int D_O>
~Linear::Linear() {
    delete params;
}

template<typename T, int D_I, int D_O>
void Linear::load_params(LinearParam<T, D_I, D_O> *p) {
    if(p != nullptr) {
        memcpy(this->params->weights, p->weights, sizeof(T)*D_I*D_O);
        memcpy(this->params->bias, p->bias, sizeof(T)*D_O);
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

