cmake_minimum_required(VERSION 3.20)

project(yavbr)

find_package(Vulkan REQUIRED)
find_package(glfw3 3.4 REQUIRED)
find_package(glm REQUIRED)

add_executable(yavbr src/main.cpp src/Application.cpp)
target_link_libraries(yavbr Vulkan::Vulkan glfw glm::glm)