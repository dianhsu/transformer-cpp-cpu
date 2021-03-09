#ifndef __MODEL_RELU_H__
#define __MODEL_RELU_H__

template<typename T, int DIM>
class Relu {
public:
    void forward(T input[DIM], T output[DIM]) {
        for (int i = 0; i < DIM; ++i) {
            if (input[i] < 0) {
                output[i] = 0;
            } else {
                output[i] = input[i];
            }
        }
    }

    void forward(T input[DIM]) {
        for (int i = 0; i < DIM; ++i) {
            if (input[i] < 0) {
                input[i] = 0;
            }
        }
    }

};

#endif