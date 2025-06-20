cmake_minimum_required(VERSION 3.10)
project(gpu_power LANGUAGES CXX)

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) Locate Python3 (interpreter, headers, libraries)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 2) Locate pybind11 (headers + CMake config)
# If pybind11 is in a custom prefix, pass -DCMAKE_PREFIX_PATH
find_package(pybind11 CONFIG REQUIRED)

# 3) Locate Intel Level-Zero loader (ze_loader)
find_library(ZE_LOADER_LIB
    NAMES ze_loader
    REQUIRED
)

# 4) Build the Python extension module
pybind11_add_module(gpu_power MODULE
    gpu_power_bindings.cpp
)

# 5) Link against pybind11 and Level-Zero loader
target_link_libraries(gpu_power PRIVATE
    pybind11::module
    ${ZE_LOADER_LIB}
)

# Optional: Install the module to Python site-packages
# install(TARGETS gpu_power DESTINATION ${Python3_SITEARCH})