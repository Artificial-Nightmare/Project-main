cmake_minimum_required(VERSION 3.25)
project(software)

set(CMAKE_CXX_STANDARD 17)

add_library(software SHARED library.cpp)
