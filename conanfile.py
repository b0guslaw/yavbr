from conan import ConanFile


class YavbrRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("glfw/3.3.8")
        self.requires("glm/cci.20230113")
        self.requires("stb/cci.20230920")
        self.requires("tinyobjloader/2.0.0-rc10")

    def build_requirements(self):
        self.tool_requires("cmake/3.22.6")
