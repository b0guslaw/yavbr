# Yet Another Vulkan Based Renderer

## Scope of this project

This repo follows the guides presented by the [vulkan-tutorial](https://vulkan-tutorial.com/) website and [vk-guide.dev](https://vkguide.dev/) and serves as a playing ground to get familier with the Vulkan Graphics API and more specifically its [C++ Bindings](https://github.com/KhronosGroup/Vulkan-Hpp).

As of writing this, the poject does nothing special or unusual that would not be already found in one of the aforementioned websites.

## Building and Running

To build and run this project you need

- C++20 compliant compiler
- CMake

Additionally the following dependencies must be installed:
- Vulkan SDK
     - This includes validation layers and Vulkan loaders
- glfw3
- glm

The **yavbr** target builds the project.

## Roadmap

This list contains a loose collection of items that I want to work on, that I have not specifically found to be mentioned on the vulkan-tutorial guide:

- [ ] Load shader source directly and compile to spir-v at runtime
- [ ] Switch to vk::raii namespace for simpler lifetime handling
- [ ] Use ctor of createInfo objects where possible
- [ ] Support iGPUs as well
- [ ] Modularize the architecture