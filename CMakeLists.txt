cmake_minimum_required(VERSION 3.5.0)
project(Raw2RGB VERSION 0.2.0 LANGUAGES C CXX)
add_compile_definitions(PROJECT_VERSION="${PROJECT_VERSION}")

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

# Add OpenMP support
find_package(OpenMP)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/version.h
        @ONLY
)

# Include directories
include_directories(
        ${CMAKE_BINARY_DIR}
        ${PROJECT_FOLDER}/include)

# Add executable with source files
add_executable(Raw2RGB src/main.cpp src/image_process.cpp
        include/argparse.hpp
        include/lut_utils.h
        src/lut_utils.cpp
)

target_include_directories(Raw2RGB PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP, enabling support.")
    target_compile_options(Raw2RGB PRIVATE /openmp:experimental)
    target_link_libraries(Raw2RGB PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found, parallelization will be disabled.")
endif()


# Link libraries
target_link_libraries(Raw2RGB PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)