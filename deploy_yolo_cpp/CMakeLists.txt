cmake_minimum_required(VERSION 2.6)

project(yolo)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)



# 添加eigen库
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})


include_directories(${PROJECT_SOURCE_DIR})
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/ubuntu/soft/TensorRT-8.2.5.1/include/)
link_directories(/home/ubuntu/soft/TensorRT-8.2.5.1/lib/)

include_directories(/home/ubuntu/soft/TensorRT-8.2.5.1/samples/common/)
#link_directories(/home/ubuntu/soft/TensorRT-8.2.5.1/lib/stubs/)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(VLPR ${PROJECT_SOURCE_DIR}/VLPRNET.cpp)
target_link_libraries(VLPR nvinfer)
target_link_libraries(VLPR cudart)
target_link_libraries(VLPR ${OpenCV_LIBS})
target_link_libraries(VLPR /home/ubuntu/soft/TensorRT-8.2.5.1/lib/stubs/libnvonnxparser.so)

add_definitions(-O2 -pthread)