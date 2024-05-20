cmake_minimum_required(VERSION 3.26.4)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(F2CGTLib)

project(F2CGTLib LANGUAGES CUDA CXX)

# ------------- configure raft -----------------#

rapids_cpm_init()
include(cmake/thirdparty/get_raft.cmake)
#find_package(raft COMPONENTS compiled distributed)

# set flag
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

# find python
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_INTERP python3)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")

# find torch
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
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
# link against libtorch_python.so
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
list(APPEND F2CGT_EXTERNAL_LIB ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# find pybind11
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_pybind11.py FIND_PYBIND_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_PYBIND_PY}
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_pybind11.py output: ${PYBIND_CMAKE_DIR}")
set(pybind11_DIR ${PYBIND_CMAKE_DIR})
find_package(pybind11 REQUIRED)


# Define our library target
file(GLOB F2CGT_SRC
  src/*.cc
  src/*.cu
)
pybind11_add_module(F2CGTLib ${F2CGT_SRC})
set_target_properties(F2CGTLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(F2CGTLib PROPERTIES CUDA_STANDARD 17)
set_target_properties(F2CGTLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
TARGET_COMPILE_OPTIONS(
  F2CGTLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Enable C++17
target_compile_features(F2CGTLib PRIVATE cxx_std_17)

# Link against LibTorch
target_link_libraries(F2CGTLib PRIVATE ${F2CGT_EXTERNAL_LIB} raft::raft raft::compiled)