#include "feedforward.h"

template<typename T, int D_I, int D_O, int D_H>

FeedForwardNetwork::FeedForwardNetwork() {
    linear1 = new Linear<T, D_I, D_H>();
    relu = new Relu<T, D_H>();
    dropout = new Dropout<T, D_H>();
    linear2 = new Linear<T, D_H, D_I>();
}
template<typename T, int D_I, int D_O, int D_H>
FeedForwardNetwork::~FeedForwardNetwork() {
    delete linear1;
    delete relu;
    delete dropout;
    delete linear2;
}
template<typename T, int D_I, int D_O, int D_H>
void FeedForwardNetwork::load_params(FeedForwardNetworkParam<T, D_I, D_O, D_H> *p) {
    if(p != nullptr) {
        linear1->load_params(p->linear_p1);
        linear2->load_params(p->linear_p2);
        if(p->dropout_rate != nullptr) {
            dropout->load_params(*(p->dropout_rate));
        }
    }
}
template<typename T, int D_I, int D_O, int D_H>
void FeedForwardNetwork::forward(T input[D_I], T output[D_O]) {
    T tmp[3][D_H];
    linear1->forward(input, tmp[0]);
    relu->forward(tmp[0], tmp[1]);
    dropout->forward(tmp[1], tmp[2]);
    linear2->forward(tmp[2], output);
}
