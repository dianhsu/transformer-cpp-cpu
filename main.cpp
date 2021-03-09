#include "transformer.h"


int main() {
    TransformerParam<float, 512, 2048, 8, 6, 6> param;
    Transformer<float, 512, 20, 2048, 8, 6, 6> transformer;
    transformer.load_params(&param);
    return 0;
}