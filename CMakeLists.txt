project(GrayscaleCUDA)
cmake_minimum_required(VERSION 3.22.1)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_75,code=sm_75)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS} "stb" "cpu" "gpu")

add_library("stb_lib" SHARED "stb/stb_image.cpp" "stb/stb_image_write.cpp")

cuda_add_executable(${PROJECT_NAME} main.cu cpu/grayscaleCPU.cpp gpu/grayscaleGPU.cu)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} "stb_lib")