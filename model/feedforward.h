#ifndef __MODEL_FEEDFORWARD_H__
#define __MODEL_FEEDFORWARD_H__

#include <cstring>

#include "dropout.h"
#include "linear.h"
#include "relu.h"

template<typename T, int D_I, int D_O, int D_H>
class FeedForwardNetwork {
public:
    FeedForwardNetwork();
    ~FeedForwardNetwork();
    void load_params(T weight1[D_I][D_H], T bias1[D_H], T weight2[D_H][D_O], T bias2[D_O], T dropout_rate);
    void forward(T input[D_I], T output[D_O]);
private:
    T weight[D_I][D_O];
    LinearWithBias<T, D_I, D_H> *linear1;
    Relu<T, D_H> *relu;
    Dropout<T, D_H> *dropout;
    LinearWithBias<T, D_H, D_O> *linear2;
};


#endif