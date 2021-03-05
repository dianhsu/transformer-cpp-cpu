

template<typename T, int DIM>
class MultiHeadAttention {
    void load_params(T weights[DIM*3][DIM], T weights[DIM][DIM]) {
        memcpy(this->q_w, weights, sizeof(T)*DIM*DIM);
        memcpy(this->k_w, weights+DIM*DIM, sizeof(T)*DIM*DIM);
        memcpy(this->v_w, weights+DIM*DIM*2, sizeof(T)*DIM*DIM);

    }
    void forward(T q_in[DIM][DIM], T k_in[DIM][DIM], T v_in[DIM][DIM], T output[DIM]) {

    }
private:
    T q_w[DIM][DIM], k_w[DIM][DIM], v_w[DIM][DIM];
    T weight[DIM][DIM];
    LinearLayer *linear;
    DropoutLayer *dropout;
};
class Encoder {

};
int main() {

    return 0;
}