cmake_minimum_required(VERSION 3.5)
project(ptextra_by_yolo)

set(CMAKE_CXX_STANDARD 17)

set(DARKNET_PATH /home/htk/src/darknet)
set(DARKNET_LIB_NAME libdark)
include_directories(${DARKNET_PATH}/include)
add_library(${DARKNET_LIB_NAME} SHARED IMPORTED)
set_property(
        TARGET ${DARKNET_LIB_NAME}
        PROPERTY IMPORTED_LOCATION 
        "${DARKNET_PATH}/libdark.so"
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.9.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(point_extr src/main.cpp )
target_link_libraries(point_extr ${OpenCV_LIBS})
target_link_libraries(point_extr ${CUDA_LIBS})
target_link_libraries(point_extr ${DARKNET_LIB_NAME})  
target_link_libraries(point_extr ${PCL_LIBRARIES})
