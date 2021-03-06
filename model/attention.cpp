#include "attention.h"
#include "function.h"

template<typename T, int DIM, int H>
MultiHeadAttention::MultiHeadAttention() {
    linear_q = new Linear<T, DIM, DIM>();
    linear_k = new Linear<T, DIM, DIM>();
    linear_v = new Linear<T, DIM, DIM>();
    linear = new Linear<T, DIM, DIM>();
    dropout = new Dropout<T, DIM>();
    this->head_size = head_size;
    this->scale = 1.0 / sqrt((DIM / H) * 1.0);
}
template<typename T, int DIM, int H>
void MultiHeadAttention::load_params(T weights1[DIM*3][DIM], T weights2[DIM][DIM], T bias1[DIM*3], T bia2[DIM]) {
    linear_q.load_params(weight1, bias1);
    linear_k.load_params(weight1+DIM*DIM, bias1+DIM);
    linear_v.load_params(weight1+DIM*2*DIM, bias1+DIM*2);
    linear.load_params(weight2, bias2);
}
template<typename T, int DIM, int H>
void MultiHeadAttention::forward(T q_in[H][DEP][DIM], T k_in[DEP][DIM], T v_in[DEP][DIM], T output[DEP][DIM]) {
    T q_tmp[DEP][DIM], k_tmp[DEP][DIM], v_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i) {
        linear_q->forward(q_in[i], q_tmp[i]);
        linear_k->forward(k_in[i], k_tmp[i]);
        linear_v->forward(v_in[i], v_tmp[i]);
        dropout->forward(q_tmp[i]);
        dropout->forward(k_tmp[i]);
        dropout->forward(v_tmp[i]);
        for(int j = 0; j < DIM; ++j) {
            q_tmp[i][j] *= this->scale;
        }
    }
    // Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    T nex_tmp[DEP][DEP];
    for(int i = 0 ;i < DEP; ++i){
        for(int j = 0; j < DEP; ++j){
            nex_tmp[i][j] = 0;
            for(int k = 0; k < DIM; ++k){
                nex_tmp[i][j] += q_tmp[i][k] * k_tmp[j][k];
            }
        }
    }
    softmax<T, DEP, DEP>(nex_tmp);
    T f_tmp[DEP][DIM];
    for(int i = 0; i < DEP; ++i){
        for(int j = 0; j < DIM; ++j){
            f_tmp[i][j] = 0;
            for(int k = 0; k < DEP; ++k){
                f_tmp[i][j] += nex_tmp[i][k] * nex_tmp[j][k];
            }
        }
    }
}
