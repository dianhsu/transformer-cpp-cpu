#ifndef __MODEL_LINEAR_H__
#define __MODEL_LINEAR_H__

template<typename T, int D_I, int D_O>
struct LinearParam{
    T weights[D_I][D_O];
    T bias[D_O];
};

template<typename T, int D_I, int D_O>
class Linear {
public:
    Linear();
    ~Linear();
    void load_params(LinearParam<T, D_I, D_O> *p);
    void forward(T input[D_I], T output[D_O]);
private:
    LinearParam<T, D_I, D_O> *params;
};

#endif