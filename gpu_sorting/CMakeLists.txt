cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu_sorting LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add executable
add_executable(gpu_sorting 
    src/main.cpp
    src/sort.cu
)

# Set CUDA architectures
set_target_properties(gpu_sorting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA architectures - modify these based on your GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80 86)
endif()

# Link libraries
target_link_libraries(gpu_sorting)

# Install
install(TARGETS gpu_sorting DESTINATION bin) 