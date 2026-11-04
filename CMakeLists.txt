cmake_minimum_required(VERSION 3.20)

project(CFrost LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(CFrost
    Tensor/tensor.cpp
    Layers/Dense.cpp
    MathModules/CPU/custommath.cpp
    Activations/ActivationClasses.cpp
    LossClasses/LossClasses.cpp
    Models/sequential.cpp
    Optimizers/SGD.cpp
)
add_executable(CFrostTests
    tests/testLR.cpp
)
target_link_libraries(CFrostTests
    PRIVATE CFrost
)
target_include_directories(CFrost
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)