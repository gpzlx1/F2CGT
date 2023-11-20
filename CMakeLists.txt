cmake_minimum_required(VERSION 3.15.0)
project(pg LANGUAGES CUDA CXX)

# set flag
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

# find torch
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
list(APPEND PG_EXTERNAL_LIB ${TORCH_LIBRARIES})

# Set openmp
include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
endif(OPENMP_FOUND)
message(STATUS "Build with OpenMP")


# Define our library target
file(GLOB PG_SRC
  src/cuda/*.cu
  src/cuda/*.cc
  src/*.cc
)
add_library(pg SHARED ${PG_SRC})
set_target_properties(pg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(pg PROPERTIES CUDA_STANDARD 17)
set_target_properties(pg PROPERTIES POSITION_INDEPENDENT_CODE ON)
TARGET_COMPILE_OPTIONS(
  pg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Enable C++17
target_compile_features(pg PRIVATE cxx_std_17)

# Link against LibTorch
target_link_libraries(pg ${PG_EXTERNAL_LIB})