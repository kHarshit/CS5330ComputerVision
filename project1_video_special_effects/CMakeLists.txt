cmake_minimum_required(VERSION 3.5)
project(OpenCVTest)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
# define the executable and its source file
add_executable(project1_app main.cpp src/imgDisplay.cpp src/vidDisplay.cpp)
# link OpenCV libraries to your executable
target_link_libraries(project1_app ${OpenCV_LIBS})