#include "transformer.h"
#include <iostream>

int main() {
    auto *param = new TransformerParam<float, 512, 2048, 8, 6, 6>();
    std::cout << param->count() << std::endl;
    auto *transformer = new Transformer<float, 512, 20, 2048, 8, 6, 6>(*param);
    array<array<float, 512>, 20> input{}, output{};
    transformer->forward(input, output);
    return 0;
}