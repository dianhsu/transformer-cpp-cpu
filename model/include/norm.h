#ifndef __MODEL_LAYERNORM_H__
#define __MODEL_LAYERNORM_H__

#include <cstring>

template<typename T, int DIM>
struct LayerNormParam {
    T weights[DIM];
    T bias[DIM];
};

template<typename T, int DIM>
class LayerNorm {
public:
    LayerNorm();
    ~LayerNorm();
    void load_params(LayerNormParam<T, DIM> *p);
    void forward(T input[DIM], T output[DIM]);
private:
    LayerNormParam<T, DIM> *params;

};


#endif