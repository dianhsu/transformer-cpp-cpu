#include "encoder_layer.h"

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
EncoderLayer::EncoderLayer() {
    norm1 = new LayerNorm<T, DIM>();
    attention = new MultiHeadAttention<T, DIM, HEAD_SIZE>();
    dropout1 = new Dropout<T, DIM>();
    norm2 = new LayerNorm<T, DIM>();
    ff = new FeedForwardNetwork<T, DIM, DIM, D_H>();
    dropout2 = new Dropout<T, DIM>();
}
template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
void EncoderLayer::load_params(
    T weights_norm1[DIM],
    T bias_norm1[DIM],
    T weights_q[HEAD_SIZE][DIM][DIM],
    T weights_k[HEAD_SIZE][DIM][DIM],
    T weights_v[HEAD_SIZE][DIM][DIM],
    T weights2[DEP*HEAD_SIZE][DIM],
    T bias_q[HEAD_SIZE][DIM],
    T bias_k[HEAD_SIZE][DIM],
    T bias_v[HEAD_SIZE][DIM],
    T bia2[DIM]
    T dropout_rate1,
    T weights1_ff[DIM][D_H],
    T bias1_ff[D_H],
    T weights2_ff[D_H][DIM],
    T bias2_ff[DIM],
    T dropout_rate_ff,
    T weights_norm2[DIM],
    T bias_norm2[DIM],
    T dropout_rate2
) {
    norm1->load_params(weights_norm1, bias_norm1);
    attention->load_params(weights_q, weights_k, weights_v, weights2, bias_q, bias_k, bias_v, bias2);
    dropout1->load_params(dropout_rate1);
    norm2->load_params(weights_norm2, bias_norm2);
    ff->load_params(weights1_ff, bias1_ff, weights2_ff, bias2_ff, dropout_rate_ff);
    dropout2->load_params(dropout_rate2);
}

template<typename T, int DIM, int DEP, int D_H, int HEAD_SIZE>
void EncoderLayer::forward(T input[DEP][DIM], T output[DEP][DIM]) {
    T i_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i) {
        norm1->forward(input[i], i_tmp[i]);
    }
    T nex_tmp[DEP][DIM];
    attention->forward(i_tmp, i_tmp, i_tmp, nex_tmp);
    for(int i = 0; i < DEP; ++i){
        dropout1->forward(nex_tmp[i]);
    }
    for(int i = 0; i < DEP; ++i){
        for(int j = 0; j < DIM; ++j){
            nex_tmp[i][j] += input[i][j];
        }
    }
    T nex2_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i){
        norm2->forward(nex_tmp[i], nex2_tmp[i]);
    }
    T nex3_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i){
        ff->forward(nex2_tmp, nex3_tmp);
    }
    for(int i = 0; i < DEP; ++i){
        dropout2->forward(nex3_tmp[i]);
    }
    for(int i = 0; i < DEP; ++i){
        for(int j = 0; j < DIM; ++j){
            output[i][j] = input[i][j] + nex3_tmp[i][j];
        }
    }
}