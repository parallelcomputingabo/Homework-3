cmake_minimum_required(VERSION 3.18)
project(app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(app main.cu)
target_link_libraries(app PRIVATE CUDA::cudart)

set_target_properties(app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 60 70 75 80 86)
