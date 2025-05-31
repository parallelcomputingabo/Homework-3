cmake_minimum_required(VERSION 3.18)
project(app LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Added for running on Puhti
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code=arch=compute_70,code=sm_70")

# Find CUDA package
find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please install it or set the CUDAToolkit_DIR variable.")
endif ()

# Add definitions
add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Add executable
add_executable(app main.cu)

# Link CUDA libraries
target_link_libraries(app PRIVATE CUDA::cudart)

# Enable separable compilation for CUDA (recommended for most CUDA projects)
set_target_properties(app PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 60 70 75 80 86 # Support common NVIDIA architectures (Pascal, Volta, Turing, Ampere, Ada)
)
