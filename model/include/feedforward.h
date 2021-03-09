#ifndef __MODEL_FEEDFORWARD_H__
#define __MODEL_FEEDFORWARD_H__

#include <cstring>

#include "dropout.h"
#include "linear.h"
#include "relu.h"
template<typename T, int D_I, int D_O, int D_H>
struct FeedForwardNetworkParam{
    LinearParam<T, D_I, D_H> *linear_p1;
    LinearParam<T, D_H, D_O> *linear_p2;
    T *dropout_rate;
    FeedForwardNetworkParam(){
        linear_p1 = new LinearParam<T, D_I, D_H>();
        linear_p2 = new LinearParam<T, D_H, D_O>();
    }
    ~FeedForwardNetworkParam(){
        delete linear_p1;
        delete linear_p2;
    }
};

template<typename T, int D_I, int D_O, int D_H>
class FeedForwardNetwork {
public:
    FeedForwardNetwork();
    ~FeedForwardNetwork();
    void load_params(FeedForwardNetworkParam<T, D_I, D_O, D_H> *p);
    void forward(T input[D_I], T output[D_O]);
private:
    Linear<T, D_I, D_H> *linear1;
    Relu<T, D_H> *relu;
    Dropout<T, D_H> *dropout;
    Linear<T, D_H, D_O> *linear2;
};


#endif