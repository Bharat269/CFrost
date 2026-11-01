cmake_minimum_required(VERSION 3.20)

project(CFrost LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(CFrost
    Layers/Dense.cpp
    Activations/ActivationClasses.cpp
    Models/sequential.cpp
    Tensor/Tensor.cpp
    LossClasses/LossClasses.cpp
)

target_include_directories(CFrost
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)