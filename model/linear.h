#ifndef __MODEL_LINEAR_H__
#define __MODEL_LINEAR_H__

template<typename T, int D_I, int D_O>
struct LinearParam {
    T weights[D_I][D_O];
    T bias[D_O];
    long long count(){
        long long ret = 0;
        ret += D_I * D_O + D_O;
        return ret;
    }
};

template<typename T, int D_I, int D_O>
class Linear {
public:
    Linear() {

    }

    ~Linear() {

    }

    void load_params(LinearParam<T, D_I, D_O> *p) {
        if (p != nullptr) {
            this->params = p;
        }
    }

    void forward(T input[D_I], T output[D_O]) {
        memcpy(output, this->params->bias, sizeof(T) * D_O);
        for (int j = 0; j < D_O; ++j) {
            for (int i = 0; i < D_I; ++i) {
                output[j] += input[i] * this->params->weights[i][j];
            }
        }
    }

private:
    LinearParam<T, D_I, D_O> *params;
};

#endif