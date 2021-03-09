#ifndef __MODEL_FEEDFORWARD_H__
#define __MODEL_FEEDFORWARD_H__

#include <cstring>

#include "dropout.h"
#include "linear.h"
#include "relu.h"

template<typename T, int D_I, int D_O, int D_H>
struct FeedForwardNetworkParam {
    LinearParam<T, D_I, D_H> *linear_p1;
    LinearParam<T, D_H, D_O> *linear_p2;
    T *dropout_rate;

    FeedForwardNetworkParam() {
        linear_p1 = new LinearParam<T, D_I, D_H>();
        linear_p2 = new LinearParam<T, D_H, D_O>();
    }

    ~FeedForwardNetworkParam() {
        delete linear_p1;
        delete linear_p2;
    }
    long long count(){
        return linear_p1->count() + linear_p1->count();
    }
};

template<typename T, int D_I, int D_O, int D_H>
class FeedForwardNetwork {
public:
    FeedForwardNetwork() {
        linear1 = new Linear<T, D_I, D_H>();
        relu = new Relu<T, D_H>();
        dropout = new Dropout<T, D_H>();
        linear2 = new Linear<T, D_H, D_I>();
    }

    ~FeedForwardNetwork() {
        delete linear1;
        delete relu;
        delete dropout;
        delete linear2;
    }

    void load_params(FeedForwardNetworkParam<T, D_I, D_O, D_H> *p) {
        if (p != nullptr) {
            linear1->load_params(p->linear_p1);
            linear2->load_params(p->linear_p2);
            if (p->dropout_rate != nullptr) {
                dropout->load_params(*(p->dropout_rate));
            }
        }
    }

    void forward(T input[D_I], T output[D_O]) {
        T tmp[3][D_H];
        linear1->forward(input, tmp[0]);
        relu->forward(tmp[0], tmp[1]);
        dropout->forward(tmp[1], tmp[2]);
        linear2->forward(tmp[2], output);
    }

private:
    Linear<T, D_I, D_H> *linear1;
    Relu<T, D_H> *relu;
    Dropout<T, D_H> *dropout;
    Linear<T, D_H, D_O> *linear2;
};


#endif