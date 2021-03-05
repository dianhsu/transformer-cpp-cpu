#include "dropout.h"

template<typename T, int DIM>
void Dropout::load_params(T dropout_rate) {
    this->dropout_rate = dropout_rate;
}
template<typename T, int DIM>
void Dropout::forward(T input[DIM], T output[DIM]) {
    for(int i = 0; i < DIM; ++i) {
        if(input[i] < this->dropout_rate) {
            output[i] = 0;
        } else {
            output[i] = input[i];
        }
    }
}
template<typename T, int DIM>
void Dropout::forward(T input[DIM]) {
    for(int i = 0; i < DIM; ++i) {
        if(input[i] < this->dropout_rate) {
            input[i] = 0;
        }
    }
}
