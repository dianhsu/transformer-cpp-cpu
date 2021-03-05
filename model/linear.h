#ifndef __MODEL_LINEAR_H__
#define __MODEL_LINEAR_H__



template<typename T, int D_I, int D_O>
class LinearWithBias {
public:
    void load_params(T weight[D_I][D_O],T bias[D_O]);
    void forward(T input[D_I], T output[D_O]);
private:
    T weight[D_I][D_O];
    T bias[D_O];
};

template<typename T, int D_I, int D_O>
class Linear {
public:
    void load_params(T weight[D_I][D_O]);
    void forward(T input[D_I], T output[D_O]);
private:
    T weight[D_I][D_O];
};

#endif