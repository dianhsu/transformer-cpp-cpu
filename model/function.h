#ifndef __MODEL_FUNCTION_H__
#define __MODEL_FUNCTION_H__

template<typename T, int DIM, int DEP>
void softmax(const array<array<T, DIM>, DEP> &input, array<array<T, DIM>, DEP> &output) {
    for (int j = 0; j < DIM; ++j) {
        T tmp = 0;
        for (int i = 0; i < DEP; ++i) {
            tmp += input[i][j];
        }
        for (int i = 0; i < DEP; ++i) {
            output[i][j] = input[i][j] / tmp;
        }
    }
}

template<typename T, int DIM, int DEP>
void softmax(array<array<T, DIM>, DEP> &input) {
    for (int j = 0; j < DIM; ++j) {
        T tmp = 0;
        for (int i = 0; i < DEP; ++i) {
            tmp += input[i][j];
        }
        for (int i = 0; i < DEP; ++i) {
            input[i][j] = input[i][j] / tmp;
        }
    }
}

#endif