cmake_minimum_required(VERSION 3.20)
project(vecsearch CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VS_SANITIZE "Build with ASan + UBSan" OFF)

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Werror)
if(VS_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# No -march=native: AVX2 kernels are compiled with target attributes and
# selected at runtime, so one binary is fast on AVX2 hosts and correct elsewhere.
add_library(vecsearch src/distance.cpp src/flat_index.cpp src/ivf_index.cpp)
target_include_directories(vecsearch PUBLIC include)
target_compile_features(vecsearch PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(vecsearch PUBLIC Threads::Threads)

add_executable(vs_bench bench/bench.cpp)
target_link_libraries(vs_bench PRIVATE vecsearch)
add_executable(vs_demo examples/demo.cpp)
target_link_libraries(vs_demo PRIVATE vecsearch)

enable_testing()
add_executable(vs_tests tests/test_vecsearch.cpp)
target_link_libraries(vs_tests PRIVATE vecsearch)
target_include_directories(vs_tests SYSTEM PRIVATE third_party)
add_test(NAME vs_tests COMMAND vs_tests)
