#ifndef __MODEL_LAYERNORM_H__
#define __MODEL_LAYERNORM_H__

#include <cstring>

template<typename T, int DIM>
struct LayerNormParams {
    T weights[DIM];
    T bias[DIM];
};

template<typename T, int DIM>
class LayerNorm {
public:
    LayerNorm();
    ~LayerNorm();
    void load_params(LayerNormParam<T, DIM> *p);
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
            output[i] = (input[i] - avg)/sq_var * (params->weight[i]) + (params->bias[i]);
        }
    }
private:
    LayerNormParams<T, DIM> *params;

};


#endif