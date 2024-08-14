# NOT TESTED YET

cmake_minimum_required(VERSION 3.10)
project(RenderBallsSo CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the output name and type
add_library(render_balls_so SHARED render_balls_so.cpp)

# Set compiler options
target_compile_options(render_balls_so PRIVATE
    -O2
    -D_GLIBCXX_USE_CXX11_ABI=0
)

# Ensure position-independent code is generated
set_target_properties(render_balls_so PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
