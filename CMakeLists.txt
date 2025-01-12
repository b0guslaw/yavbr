cmake_minimum_required(VERSION 3.20)

project(yavbr)

set(CMAKE_CXX_STANDARD 23)

find_package(Vulkan REQUIRED)
find_package(glfw3 3 REQUIRED)
find_package(glm REQUIRED)
find_package(stb REQUIRED)
find_package(tinyobjloader REQUIRED)
find_package(imgui REQUIRED)

file(COPY models
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY textures
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY shaders
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(yavbr
        src/main.cpp
        src/Application.cpp)

target_include_directories(yavbr PRIVATE ${CMAKE_SOURCE_DIR}/external)

target_link_libraries(yavbr
        PRIVATE Vulkan::Vulkan
        PRIVATE glfw
        PRIVATE glm::glm
        PRIVATE stb::stb
        PRIVATE tinyobjloader::tinyobjloader
        PRIVATE imgui::imgui)
