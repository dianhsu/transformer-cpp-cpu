#ifndef __MODEL_FEEDFORWARD_H__
#define __MODEL_FEEDFORWARD_H__

#include <cstring>

#include "dropout.h"
#include "linear.h"
#include "relu.h"

template<typename T, int D_I, int D_O, int D_H>
struct FeedForwardNetworkParam {
    LinearParam<T, D_I, D_H> linear_p1;
    LinearParam<T, D_H, D_O> linear_p2;
    T dropout_rate;

    FeedForwardNetworkParam() {
        dropout_rate = 0.1;
    }

    long long count() {
        return linear_p1.count() + linear_p2.count();
    }
};

template<typename T, int D_I, int D_O, int D_H>
class FeedForwardNetwork {
public:
    explicit FeedForwardNetwork(FeedForwardNetworkParam<T, D_I, D_O, D_H> &param) {
        linear1 = new Linear<T, D_I, D_H>(param.linear_p1);
        relu = new Relu<T, D_H>();
        dropout = new Dropout<T, D_H>(param.dropout_rate);
        linear2 = new Linear<T, D_H, D_I>(param.linear_p2);
    }

    ~FeedForwardNetwork() {
        delete linear1;
        delete relu;
        delete dropout;
        delete linear2;
    }

    void forward(const array<T, D_I> input, array<T, D_O> &output) {
        auto tmp = array<array<T, D_H>, 3>{};
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