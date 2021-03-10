#ifndef __MODEL_DROPOUT_H__
#define __MODEL_DROPOUT_H__

#include <array>

using std::array;

template<typename T, int DIM>
class Dropout {
public:
    Dropout() {
        this->dropout_rate = 0.1;
    }

    explicit Dropout(T dr) {
        this->dropout_rate = dr;
    }


    void forward(array<T, DIM> &input, array<T, DIM> &output) {
        for (int i = 0; i < DIM; ++i) {
            if (input[i] < this->dropout_rate) {
                output[i] = 0;
            } else {
                output[i] = input[i];
            }
        }
    }

    void forward(array<T, DIM> &input) {
        for (int i = 0; i < DIM; ++i) {
            if (input[i] < this->dropout_rate) {
                input[i] = 0;
            }
        }
    }

private:
    T dropout_rate;
};

#endif