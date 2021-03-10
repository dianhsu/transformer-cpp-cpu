cmake_minimum_required(VERSION 3.16)

project(Transformer)
set(CMAKE_CXX_STANDARD 20)

include_directories("model")
include(CTest)
add_executable(transformer main.cpp)
add_test(transformer_test transformer)

