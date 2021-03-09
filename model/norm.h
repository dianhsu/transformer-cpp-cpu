#ifndef __MODEL_LAYERNORM_H__
#define __MODEL_LAYERNORM_H__

#include <cstring>

template<typename T, int DIM>
class LayerNorm {
public:
    void load_params(T weight[DIM], T bias[DIM]) {
        memcpy(this->weight, weight, sizeof(weight));
        memcpy(this->bias, bias, sizeof(bias));
    }
    void forward(T input[DIM], T output[DIM]) {
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
            output[i] = (input[i] - avg)/sq_var * weight[i] + bias[i];
        }
    }
private:
    T weight[DIM];
    T bias[DIM];

};

#endif