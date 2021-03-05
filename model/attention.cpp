#include "attention.h"

template<typename T, int DIM>
MultiHeadAttention::MultiHeadAttention(int head_size){
    linear_q = new Linear<T, DIM, DIM>();
    linear_k = new Linear<T, DIM, DIM>();
    linear_v = new Linear<T, DIM, DIM>();
    linear = new Linear<T, DIM*3, DIM>();
    dropout = new Dropout<T, DIM>();
    this->head_size = head_size;
    this->scale = 1.0 / sqrt((DIM / head_size) * 1.0);
}
template<typename T, int DIM>
void MultiHeadAttention::load_params(T weights1[DIM*3][DIM], T weights2[DIM][DIM]) {
    memcpy(this->q_w, weights1, sizeof(T)*DIM*DIM);
    memcpy(this->k_w, weights1+DIM*DIM, sizeof(T)*DIM*DIM);
    memcpy(this->v_w, weights1+DIM*DIM*2, sizeof(T)*DIM*DIM);
    memcpy(this->weights, weights2, sizeof(T)*DIM*DIM);
}
template<typename T, int DIM>
void MultiHeadAttention::forward(T q_in[DIM], T k_in[DIM], T v_in[DIM], T output[DIM]) {
    T q_tmp[DIM], k_tmp[DIM], v_tmp[DIM];
    linear_q->forward(q_in, q_tmp);
    linear_k->forward(k_in, k_tmp);
    linear_v->forward(v_in, v_tmp);
    dropout->forward(q_tmp);
    dropout->forward(k_tmp);
    dropout->forward(v_tmp);

    for(int i = 0; i < DIM; ++i){
        q_tmp[i] *= this->scale;
    }
}
