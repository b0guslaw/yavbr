#pragma once

#ifdef _WIN32
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <optional>
#include <array>

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription() {
		vk::VertexInputBindingDescription bindingDescription
		{
			0,
			sizeof(Vertex),
			vk::VertexInputRate::eVertex
		};
		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescription() {
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions
		{{
			{0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)},
			{1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, color)}
		}};
		return attributeDescriptions;
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

inline VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

inline void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}

struct QueueFamilyIndices {
	std::optional<std::uint32_t> graphics_family;
	std::optional<std::uint32_t> present_family;
};

struct SwapChainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilties;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

class Application {

	const std::uint32_t WIDTH{ 800 };
	const std::uint32_t HEIGHT{ 600 };
	const std::uint32_t MAX_FRAMES_IN_FLIGHT{ 2 };

	const std::string application_name{ "Vulkan Application" };
	const std::string engine_name{ "Vulkan.hpp" };

public:
	void run();
private:
	void initWindow();
	void initVulkan();
	void mainLoop();
	void cleanUp();
	void createInstance();
	void createSurface();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createSwapChain();
	void createImageViews();
	void createRenderPass();
	void createGraphicsPipeline();
	void createDescriptorSetLayout();
	void createFramebuffers();
	void createCommandPool();
	void createCommandBuffers();
	void recordCommandBuffer(vk::CommandBuffer, std::uint32_t);
	void createSyncObjects();
	void recreateSwapChain();
	void cleanupSwapChain();
	void createBuffer(vk::DeviceSize, vk::BufferUsageFlags, vk::MemoryPropertyFlags, vk::Buffer&, vk::DeviceMemory&);
	void createVertexBuffer();
	void createIndexBuffer();
	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSet();
	void copyBuffer(vk::Buffer, vk::Buffer, vk::DeviceSize);
	std::vector<char> readFile(const std::string&);
	vk::ShaderModule createShaderModule(const std::vector<char>&);

	bool isSuitableDevice(vk::PhysicalDevice);
	bool deviceHasExtensionSupport(vk::PhysicalDevice);

	void drawFrame();
	void updateUniformBuffer(std::uint32_t);

	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice);
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>&);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>&);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR&);
	SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice);

	std::uint32_t findMemoryType(std::uint32_t, vk::MemoryPropertyFlags);

	void setupDebugCallback();
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT,
		VkDebugUtilsMessageTypeFlagsEXT,
		const VkDebugUtilsMessengerCallbackDataEXT*,
		void*
	);

	static void framebufferResizedCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
		app->m_framebufferResized = true;
	}

	std::vector<const char*> enableValidationLayers();
	std::vector<const char*> getRequiredExtensions();
	std::vector<vk::Framebuffer> m_swapChainFramebuffers;

	GLFWwindow* m_window;
	vk::Instance m_instance;
	vk::PhysicalDevice m_physicalDevice;
	vk::Device m_device;
	vk::Queue m_graphicsQueue;
	vk::Queue m_presentQueue;
	vk::SurfaceKHR m_surface;
	vk::SwapchainKHR m_swapchain;
	std::vector<vk::Image> m_swapchainImages;
	std::vector<vk::ImageView> m_swapchainImageViews;
	vk::Format m_swapchainImageFormat;
	vk::Extent2D m_swapchainExtent;
	vk::RenderPass m_renderPass;
	vk::DescriptorSetLayout m_descriptorSetLayout;
	vk::PipelineLayout m_pipelineLayout;
	vk::Pipeline m_graphicsPipeline;
	vk::CommandPool m_commandPool;
	std::vector<vk::CommandBuffer> m_commandBuffers;
	VkDebugUtilsMessengerEXT callback;
	std::vector<vk::Semaphore> m_imageAvailabeSemaphores;
	std::vector<vk::Semaphore> m_renderFinishedSemaphores;
	std::vector<vk::Fence> m_inFlightFences;
	bool m_framebufferResized{ false };
	bool m_enableValidationLayers{ true };
	float m_queuePriority{ 1.0f };
	std::uint32_t m_currentFrame{ 0 };

	vk::Buffer m_vertexBuffer;
	vk::DeviceMemory m_vertexBufferMemory;
	std::vector<Vertex> m_vertices;

	vk::Buffer m_indexBuffer;
	vk::DeviceMemory m_indexBufferMemory;
	std::vector<std::uint16_t> m_indices;
	
	std::vector<vk::Buffer> m_uniformBuffers;
	std::vector<vk::DeviceMemory> m_uniformBuffersMemory;
	std::vector<void*> m_uniformBuffersMapped;

	vk::DescriptorPool m_descriptorPool;
	std::vector<vk::DescriptorSet> m_descriptorSets;
};