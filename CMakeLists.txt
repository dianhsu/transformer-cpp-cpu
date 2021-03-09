cmake_minimum_required(VERSION 3.5)

project(Transformer)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)
set(CMAKE_CXX_FLAGS "-std=c++2a")
file(GLOB SRC "model/src/*.cpp")
file(GLOB INC "model/include/*.h")
include_directories("model/include")

add_executable(transformer ${SRC} main.cpp ${INC})