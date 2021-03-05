#include "feedforward.h"

template<typename T, int D_I, int D_O, int D_H>

FeedForwardNetwork::FeedForwardNetwork() {
    linear1 = new LinearWithBias<T, D_I, D_H>();
    relu = new Relu<T, D_H>();
    dropout = new Dropout<T, D_H>();
    linear2 = new LinearWithBias<T, D_H, D_I>();
}
template<typename T, int D_I, int D_O, int D_H>
~FeedForwardNetwork::FeedForwardNetwork() {
    delete linear1;
    delete relu;
    delete dropout;
    delete linear2;
}
template<typename T, int D_I, int D_O, int D_H>
void FeedForwardNetwork::load_params(T weight1[D_I][D_H], T bias1[D_H], T weight2[D_H][D_O], T bias2[D_O], T dropout_rate) {
    linear1->load_params(weight1, bias1);
    linear2->load_params(weight2, bias2);
    dropout->load_params(dropout_rate);
}
template<typename T, int D_I, int D_O, int D_H>
void FeedForwardNetwork::forward(T input[D_I], T output[D_O]) {
    T tmp1[D_H], tmp2[D_H], tmp3[D_H];
    linear1->forward(input, tmp1);
    relu->forward(tmp1, tmp2);
    dropout->forward(tmp2, tmp3);
    linear2->forward(tmp3, output);
}
