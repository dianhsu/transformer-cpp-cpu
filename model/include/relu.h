#ifndef __MODEL_RELU_H__
#define __MODEL_RELU_H__

template<typename T, int DIM>
class Relu {
public:
    void forward(T input[DIM], T output[DIM]);
    void forward(T input[DIM]);
};

#endif