//
// Created by dianhsu on 2021/03/11.
//

#ifndef TRANSFORMER_TRANSFORMER_H
#define TRANSFORMER_TRANSFORMER_H

#include <array>
#include "encoder.h"
#include "decoder.h"

namespace transformer {
    template<typename T, int DIM, int DIM_HID, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
    struct TransformerParameter {
        EncoderParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT> encoder_p;
        DecoderParameter<T, DIM, DIM_HID, HEAD_SIZE, DEC_LAYER_CNT> decoder_p;

        long long count() {
            return encoder_p.count() + decoder_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int DIM_HID, int HEAD_SIZE, int ENC_LAYER_CNT, int DEC_LAYER_CNT>
    class Transformer {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &input,
                            std::array<std::array<T, DIM>, DEP> &output,
                            TransformerParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, DEC_LAYER_CNT> &p) {
            auto tmp = std::array<std::array<T, DIM>, DEP>{};
            Encoder<T, DIM, DEP, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT>::forward(input, tmp, p.encoder_p);
            Decoder<T, DIM, DEP, DIM_HID, HEAD_SIZE, DEC_LAYER_CNT>::forward(tmp, tmp, output, p.decoder_p);
        }
    };
}
#endif //TRANSFORMER_TRANSFORMER_H
