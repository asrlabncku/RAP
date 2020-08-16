cmake_minimum_required(VERSION 3.5.1)

project(cnet)

include_directories(core layer)

file(GLOB core "core/*.c")
file(GLOB layer "layer/*.c")

if (UNIX)
set(libs m)
endif()

add_library(cnet SHARED ${core} ${layer})


