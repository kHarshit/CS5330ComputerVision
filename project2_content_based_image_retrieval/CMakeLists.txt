cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
project(ImageBasedContentRetrieval)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
# define the executable and its source file
add_executable(project2_app main.cpp src/feature.cpp src/distance.cpp src/csv_util.cpp)
# link OpenCV libraries to your executable
target_link_libraries(project2_app ${OpenCV_LIBS})