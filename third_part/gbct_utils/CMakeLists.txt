cmake_minimum_required(VERSION 3.6)

project(gbct_utils LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off
execute_process(COMMAND python -m pybind11 --cmakedir RESULT_VARIABLE tmp OUTPUT_VARIABLE PYBIND11_CMAKE OUTPUT_STRIP_TRAILING_WHITESPACE) 
# MESSAGE("THIS is cmake dir ${PYBIND11_CMAKE}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PYBIND11_CMAKE})

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(pybind11_DIR ${PYBIND11_CMAKE})
set(root_path ".")
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED)
if (Python3_FOUND)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

include_directories(${root_path})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread ${PYBIND11_VAR}")

FILE(GLOB src_cpps "${root_path}/src/*.cpp" "include/*.hpp" "include/utils/*.hpp")
FILE(GLOB src_hs "include/*.h" "include/utils/*.h")

set(gbct_files ${src_cpps})

message("gbct files ${gbct_files}")
pybind11_add_module(gbct_utils MODULE ${gbct_files})
target_compile_definitions(gbct_utils 
                           PRIVATE VERSION=${EXAMPLE_VERSION_INFO})

