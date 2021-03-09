#include "transformer.h"
#include <iostream>

int main() {
    TransformerParam<float, 512, 2048, 8, 6, 6> param;
    std::cout << param.count() << std::endl;
    Transformer<float, 512, 20, 2048, 8, 6, 6> transformer;
    //transformer.load_params(&param);
    return 0;
}