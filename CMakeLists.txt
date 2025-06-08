cmake_minimum_required(VERSION 3.18)
project(Homework3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
enable_language(CUDA)

add_executable(main main.cu)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code arch=compute_80,code=sm_80>)
target_link_libraries(main PRIVATE ${CUDA_LIBRARIES})