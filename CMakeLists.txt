cmake_minimum_required(VERSION 3.5)

project(Transformer)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)

file(GLOB SRC "model/*.cpp")
include_directories(model)

add_executable(transformer ${SOURCE} main.cpp)