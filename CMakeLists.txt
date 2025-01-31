cmake_minimum_required(VERSION 3.5.0)
project(Raw2RGB VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Current source directory
set(PROJECT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${PROJECT_FOLDER}/include)

# Add executable with source files
add_executable(Raw2RGB src/main.cpp src/image_process.cpp)

# Link libraries
target_link_libraries(Raw2RGB PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)