#include "norm.h"

template<typename T, int DIM>
LayerNorm<T, DIM>::LayerNorm() {
}

template<typename T, int DIM>
LayerNorm<T, DIM>::~LayerNorm() {
}

template<typename T, int DIM>
void LayerNorm<T, DIM>::load_params(LayerNormParam<T, DIM> *p) {
    if(p != nullptr) {
        this->params = p;
    }
}
template<typename T, int DIM>
void LayerNorm<T, DIM>::forward(T input[DIM], T output[DIM]) {
    T sum = 0;
    T sum2 = 0;
    for(int i = 0; i < DIM; ++i) {
        sum += input[i];
        sum2 += input[i] * input[i];
    }
    T avg = sum / DIM;
    T avg2 = sum2 / DIM;
    T var = avg2 - avg * avg;
    T sq_var = sqrt(var + 1e-5);
    for(int i = 0; i < DIM; ++i) {
        output[i] = (input[i] - avg)/sq_var * params->weights[i] + params->bias[i];
    }
}