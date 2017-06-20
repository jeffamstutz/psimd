## ========================================================================== ##
## The MIT License (MIT)                                                      ##
##                                                                            ##
## Copyright (c) 2017 Jefferson Amstutz                                       ##
##                                                                            ##
## Permission is hereby granted, free of charge, to any person obtaining a    ##
## copy of this software and associated documentation files (the "Software"), ##
## to deal in the Software without restriction, including without limitation  ##
## the rights to use, copy, modify, merge, publish, distribute, sublicense,   ##
## and/or sell copies of the Software, and to permit persons to whom the      ##
## Software is furnished to do so, subject to the following conditions:       ##
##                                                                            ##
## The above copyright notice and this permission notice shall be included in ##
## in all copies or substantial portions of the Software.                     ##
##                                                                            ##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR ##
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   ##
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    ##
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER ##
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    ##
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        ##
## DEALINGS IN THE SOFTWARE.                                                  ##
## ========================================================================== ##

##############################################################
# CMake configuration
##############################################################

cmake_minimum_required(VERSION 2.8)

project(psimd)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/psimd.cmake)

psimd_setup_build_type()

# we are only building tests, should probably stick with 'Release'
mark_as_advanced(CMAKE_BUILD_TYPE)

install(DIRECTORY psimd DESTINATION include)

##############################################################
# Build tests
##############################################################

psimd_configure_compiler()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(PSIMD_SOURCES
  psimd/psimd.h
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/tests)

include_directories(${CMAKE_SOURCE_DIR})

enable_testing()

add_executable(test_pack
  tests/doctest.h
  tests/test_pack.cpp
  ${PSIMD_SOURCES}
)

add_test(arithmetic_operators
         tests/test_pack "--test-suite=\"arithmetic operators\"")

add_test(bitwise_operators
         tests/test_pack "--test-suite=\"bitwise operators\"")

add_test(logic_operators
         tests/test_pack "--test-suite=\"logic operators\"")

add_test(math_functions
         tests/test_pack "--test-suite=\"math functions\"")

add_test(algorithms
         tests/test_pack "--test-suite=\"algorithms\"")
