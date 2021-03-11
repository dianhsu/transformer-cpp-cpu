//
// Created by dianhsu on 2021/03/10.
//

#ifndef TRANSFORMER_ENCODER_H
#define TRANSFORMER_ENCODER_H

#include <array>
#include "encoder_layer.h"

namespace transformer {
    template<typename T, int DIM, int DIM_HID, int HEAD_SIZE, int LAYER_CNT>
    struct EncoderParameter {
        std::array<EncoderLayerParameter<T, DIM, DIM_HID, HEAD_SIZE>, LAYER_CNT> layers_p;
        LayerNormParameter<T, DIM> norm_p;

        long long count() {
            return layers_p[0].count() * LAYER_CNT + norm_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int DIM_HID, int HEAD_SIZE, int LAYER_CNT>
    class Encoder {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &input,
                            std::array<std::array<T, DIM>, DEP> &output,
                            EncoderParameter<T, DIM, DIM_HID, HEAD_SIZE, LAYER_CNT> &p) {
            auto tmp = std::array<std::array<std::array<T, DIM>, DEP>, LAYER_CNT>{};
            for (int i = 0; i < LAYER_CNT; ++i) {
                if (i == 0) {
                    EncoderLayer<T, DIM, DEP, DIM_HID, HEAD_SIZE>::forward(input, tmp[0], p.layers_p[i]);
                } else {
                    EncoderLayer<T, DIM, DEP, DIM_HID, HEAD_SIZE>::forward(tmp[i - 1], tmp[i], p.layers_p[i]);
                }
            }
            for (int i = 0; i < DEP; ++i) {
                LayerNorm<T, DIM>::forward(tmp[LAYER_CNT - 1][i], output[i], p.norm_p);
            }
        }
    };
}
#endif //TRANSFORMER_ENCODER_H
