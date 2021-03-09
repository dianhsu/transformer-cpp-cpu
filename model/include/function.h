#ifndef __MODEL_FUNCTION_H__
#define __MODEL_FUNCTION_H__

template<typename T, int DIM, int DEP>
void softmax(T input[DEP][DIM], T output[DEP][DIM]){
    for(int j = 0; j < DIM; ++j){
        T tmp = 0;
        for(int i = 0; i < DEP; ++i){
            tmp += input[i][j];
        }
        for(int i = 0; i < DEP; ++i){
            output[i][j] = input[i][j] / tmp;
        }
    }
}

template<typename T, int DIM, int DEP>
void softmax(T input[DEP][DIM]){
    for(int j = 0; j < DIM; ++j){
        T tmp = 0;
        for(int i = 0; i < DEP; ++i){
            tmp += input[i][j];
        }
        for(int i = 0; i < DEP; ++i){
            input[i][j] = input[i][j] / tmp;
        }
    }
}
#endif