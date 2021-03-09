cmake_minimum_required(VERSION 3.17)

project(Transformer)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)
set(CMAKE_CXX_STANDARD 20)


include_directories("model")
add_executable(transformer main.cpp)

