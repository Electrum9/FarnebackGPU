# Use a reasonably modern CMake version, 3.8+ has good CUDA support
cmake_minimum_required(VERSION 3.8)

# Define the project name and enable C++ and CUDA languages
project(PolynomialExpansion LANGUAGES CXX CUDA)

# --- OpenCV Configuration ---
# Tell CMake where to find your custom OpenCV build
# It looks for OpenCVConfig.cmake or opencv-config.cmake in this directory
set(OpenCV_DIR "/afs/ece.cmu.edu/usr/rtafresh/Private/build/" CACHE PATH "Path to custom OpenCV build directory")
message(STATUS "Looking for OpenCV in: ${OpenCV_DIR}")

# Find the OpenCV package. REQUIRED means CMake will stop with an error if not found.
# You can optionally list specific components you need, e.g., core imgproc
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
# Include OpenCV headers system-wide for this project (optional, but common)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- CUDA Configuration ---
# Find the CUDA toolkit. REQUIRED means CMake stops if not found.
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")
# Include CUDA headers (needed for cuda_runtime.h etc.)
# target_include_directories is preferred, see below

# --- Build Target: Your Shared Library ---
# Define the name of your shared library (e.g., PolyExp)
# List ALL source files needed (.cu, .cpp, .h if headers need processing)
add_library(PolyExp SHARED
    PolyExpansion_Test.cu # <-- Make sure this is your main CUDA file name!
    # Add any other .cpp or .cu files needed for this library here
)

# Set C++ Standard (optional but recommended)
set_target_properties(PolyExp PROPERTIES CXX_STANDARD 11) # Or 14, 17

# Set CUDA Standard (optional but recommended)
set_target_properties(PolyExp PROPERTIES CUDA_STANDARD 11) # Or 14, 17

# --- Linking ---
# Link your library against OpenCV and CUDA libraries
target_link_libraries(PolyExp PUBLIC # Use PUBLIC or PRIVATE as appropriate
    ${OpenCV_LIBS}      # Variable containing paths to required OpenCV libraries
    ${CUDA_LIBRARIES}   # Variable containing paths to CUDA runtime libraries (e.g., libcudart.so)
)

# --- Include Directories (Preferred Method) ---
# Specify include directories needed specifically for the PolyExp target
target_include_directories(PolyExp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # Include local headers
    ${OpenCV_INCLUDE_DIRS}      # Include dirs for OpenCV headers
    ${CUDA_INCLUDE_DIRS}        # Include dirs for CUDA headers (cuda_runtime.h)
)

# --- Optional: Compiler Flags ---
# CMake handles -fPIC automatically for SHARED libraries on most platforms.
# If you need other specific nvcc flags:
# target_compile_options(PolyExp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --your-nvcc-flag -O3 >)

# --- Installation (Optional) ---
# If you want to install the library and headers:
# install(TARGETS PolyExp DESTINATION lib)
# install(FILES your_header.h DESTINATION include)

message(STATUS "CMake configuration finished for PolyExp library.")