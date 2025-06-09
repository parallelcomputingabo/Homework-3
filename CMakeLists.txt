cmake_minimum_required(VERSION 3.18)
project(app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please install it or set the CUDAToolkit_DIR variable.")
endif ()

add_executable(app main.cu)
target_link_libraries(app PRIVATE CUDA::cudart)

set_target_properties(app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 80 # Optimized for A100 (Ampere)
)