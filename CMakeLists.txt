cmake_minimum_required(VERSION 4.1.1)

project(yavbr)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG latest
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

find_package(Vulkan REQUIRED)

file(COPY models
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY textures
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY shaders
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(yavbr
        src/main.cpp
        src/Application.cpp)

target_include_directories(yavbr
    PRIVATE
    ${CMAKE_SOURCE_DIR}/external)

target_link_libraries(yavbr
        PRIVATE Vulkan::Vulkan
        PRIVATE glfw
        PRIVATE glm)
