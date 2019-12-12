cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ssd-app)

set(LIB_PATH "/home/dl/3rdparty/")

set (Torch_DIR "${LIB_PATH}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${Torch_INCLUDE_DIRS})
message("Torch_INCLUDE_DIRS " ${Torch_INCLUDE_DIRS})

set(OpenCV_DIR "/usr/share/OpenCV")
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV_DIR " ${OpenCV_INCLUDE_DIRS})

set(SSDInclude  ./)

include_directories(${SSDInclude})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ssd-app main.cpp ssd.cpp)
target_link_libraries(ssd-app "${TORCH_LIBRARIES}"  "${OpenCV_LIBRARIES}")
set_property(TARGET ssd-app PROPERTY CXX_STANDARD 11)


