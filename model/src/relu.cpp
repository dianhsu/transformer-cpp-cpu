#include "relu.h"

template<typename T, int DIM>
void Relu::forward(T input[DIM], T output[DIM]) {
    for(int i = 0; i < DIM; ++i) {
        if(input[i] < 0) {
            output[i] = 0;
        } else {
            output[i] = input[i];
        }
    }
}
template<typename T, int DIM>
void Relu::forward(T input[DIM]) {
    for(int i = 0; i < DIM; ++i) {
        if(input[i] < 0) {
            input[i] = 0;
        }
    }
}
