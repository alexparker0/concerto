cmake_minimum_required(VERSION 3.20)
project(concerto_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(concerto_core SHARED
    src/kernels.cpp
    src/c_api.cpp
)
target_include_directories(concerto_core PRIVATE include)
if(MSVC)
    target_compile_options(concerto_core PRIVATE /O2 /fp:precise /arch:AVX2 /GL)
    target_link_options(concerto_core PRIVATE /LTCG)
else()
    target_compile_options(concerto_core PRIVATE -O3 -ffp-contract=off -mavx2 -mfma -flto)
    # Self-contained DLL: no MinGW runtime DLLs needed at load time.
    target_link_options(concerto_core PRIVATE -static -static-libgcc -static-libstdc++ -flto)
endif()

set_target_properties(concerto_core PROPERTIES
    OUTPUT_NAME "concerto_core"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
