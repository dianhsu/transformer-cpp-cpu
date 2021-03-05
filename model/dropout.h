#ifndef __MODEL_DROPOUT_H__
#define __MODEL_DROPOUT_H__

template<typename T, int DIM>
class Dropout {
    void load_params(T dropout_rate);
    void forward(T input[DIM], T output[DIM]);
    void forward(T input[DIM]);
private:
    T dropout_rate;
};

#endif