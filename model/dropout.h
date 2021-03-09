#ifndef __MODEL_DROPOUT_H__
#define __MODEL_DROPOUT_H__

template<typename T, int DIM>
class Dropout {
public:
    Dropout() {
        this->dropout_rate = 0.1;
    }

    void load_params(T dr) {
        this->dropout_rate = dr;
    }

    void forward(T input[DIM], T output[DIM]) {
        for (int i = 0; i < DIM; ++i) {
            if (input[i] < this->dropout_rate) {
                output[i] = 0;
            } else {
                output[i] = input[i];
            }
        }
    }

    void forward(T input[DIM]) {
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