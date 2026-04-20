cmake_minimum_required(VERSION 3.16)
project(the_well_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release flags ──
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -DNDEBUG")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -mavx2 -mfma")

# ── Static linking (bare-metal philosophy) ──
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(BUILD_SHARED_LIBS OFF)

# ── Source ──
add_executable(the_well_cpp src/main.cpp)

target_include_directories(the_well_cpp PRIVATE src)

# ── Platform libs ──
if(WIN32)
    target_link_libraries(the_well_cpp PRIVATE -static -lpthread)
else()
    target_link_libraries(the_well_cpp PRIVATE pthread m)
endif()
