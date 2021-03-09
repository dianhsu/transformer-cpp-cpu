#include "linear.h"
#include "feedforward.h"
#include "dropout.h"
#include "norm.h"
#include "attention.h"

template<typename T, int DIM, int DEP>
class EncoderLayer {
    void load_params(T weights_q[H][DIM][DIM],
                     T weights_k[H][DIM][DIM],
                     T weights_v[H][DIM][DIM],
                     T weights2[DEP*H][DIM],
                     T bias_q[H][DIM],
                     T bias_k[H][DIM],
                     T bias_v[H][DIM],
                     T bia2[DIM]);
    void forward(T input[DEP][DIM], T output[DEP][DIM]);
};