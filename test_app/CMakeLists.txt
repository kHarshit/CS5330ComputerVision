cmake_minimum_required(VERSION 3.0)
project(OpenCVTest)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# define the executable and its source file
add_executable(OpenCVTest main.cpp)
# link OpenCV libraries to your executable
target_link_libraries(OpenCVTest ${OpenCV_LIBS})