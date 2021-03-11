//
// Created by dianhsu on 2021/03/11.
//

#ifndef TRANSFORMER_DECODER_H
#define TRANSFORMER_DECODER_H

#include <array>
#include "decoder_layer.h"

namespace transformer {
    template<typename T, int DIM, int DIM_HID, int HEAD_SIZE, int LAYER_CNT>
    struct DecoderParameter {
        DecoderLayerParameter<T, DIM, DIM_HID, HEAD_SIZE> layers_p[LAYER_CNT];
        LayerNormParameter<T, DIM> norm_p;

        long long count() {
            return layers_p[0].count() * LAYER_CNT + norm_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int DIM_HID, int HEAD_SIZE, int LAYER_CNT>
    class Decoder {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &input,
                            std::array<std::array<T, DIM>, DEP> &enc_output,
                            std::array<std::array<T, DIM>, DEP> &output,
                            DecoderParameter<T, DIM, DIM_HID, HEAD_SIZE, LAYER_CNT> &p) {
            auto tmp = std::array<std::array<std::array<T, DIM>, DEP>, LAYER_CNT>{};
            for (int i = 0; i < LAYER_CNT; ++i) {
                if (i == 0) {
                    DecoderLayer<T, DIM, DEP, DIM_HID, HEAD_SIZE>::forward(input, enc_output, tmp[0], p.layers_p[i]);
                } else {
                    DecoderLayer<T, DIM, DEP, DIM_HID, HEAD_SIZE>::forward(tmp[i - 1], enc_output, tmp[0],
                                                                           p.layers_p[i]);
                }
            }
            for (int i = 0; i < DEP; ++i) {
                LayerNorm<T, DIM>::forward(tmp[LAYER_CNT - 1][i], output[i], p.norm_p);
            }
        }
    };
}
#endif //TRANSFORMER_DECODER_H
