//
// Created by dianhsu on 2021/03/11.
//

#ifndef TRANSFORMER_DECODER_LAYER_H
#define TRANSFORMER_DECODER_LAYER_H

#include "norm.h"
#include "attention.h"
#include "dropout.h"
#include "feedforward.h"

namespace transformer {
    template<typename T, int DIM, int DIM_HID, int HEAD_SIZE>
    struct DecoderLayerParameter {
        LayerNormParameter<T, DIM> norm1_p;
        MultiHeadAttentionParameter<T, DIM, HEAD_SIZE> attention1_p;
        T dr1;
        LayerNormParameter<T, DIM> norm2_p;
        MultiHeadAttentionParameter<T, DIM, HEAD_SIZE> attention2_p;
        LayerNormParameter<T, DIM> norm3_p;
        FeedForwardNetworkParameter<T, DIM, DIM, DIM_HID> ff_p;
        T dr2;

        DecoderLayerParameter() {
            dr1 = 0.1;
            dr2 = 0.1;
        }

        long long count() {
            return norm1_p.count() + attention1_p.count() + norm2_p.count() + attention2_p.count() + norm3_p.count() +
                   ff_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int DIM_HID, int HEAD_SIZE>
    class DecoderLayer {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &input,
                            std::array<std::array<T, DIM>, DEP> &enc_output,
                            std::array<std::array<T, DIM>, DEP> &output,
                            DecoderLayerParameter<T, DIM, DIM_HID, HEAD_SIZE> &p) {
            auto tmp = std::array<std::array<std::array<T, DIM>, DEP>, 7>{};
            for (int i = 0; i < DEP; ++i) {
                LayerNorm<T, DIM>::forward(input[i], tmp[0][i], p.norm1_p);
            }
            MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>::forward(tmp[0], tmp[0], tmp[0], tmp[1], p.attention1_p);
            for (int i = 0; i < DEP; ++i) {
                Dropout<T, DIM>::forward(tmp[1][i], tmp[2][i], p.dr1);
            }
            for (int i = 0; i < DEP; ++i) {
                LayerNorm<T, DIM>::forward(tmp[2][i], tmp[3][i], p.norm2_p);
            }
            MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>::forward(tmp[3], enc_output, enc_output, tmp[4], p.attention2_p);
            for (int i = 0; i < DEP; ++i) {
                LayerNorm<T, DIM>::forward(tmp[4][i], tmp[5][i], p.norm3_p);
            }
            for (int i = 0; i < DEP; ++i) {
                FeedForwardNetwork<T, DIM, DIM, DIM_HID>::forward(tmp[5][i], tmp[6][i], p.ff_p);
            }
            for (int i = 0; i < DEP; ++i) {
                Dropout<T, DIM>::forward(tmp[6][i], output[i], p.dr2);
            }
        }
    };
}
#endif //TRANSFORMER_DECODER_LAYER_H
