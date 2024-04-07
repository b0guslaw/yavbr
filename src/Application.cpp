#include "Application.h"

#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <limits>
#include <array>
#include <fstream>
#include <chrono>

std::vector<const char*> requestedLayers = { "VK_LAYER_KHRONOS_validation" };
std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

void Application::run()
{
	m_vertices = {
		{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
		{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
		{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
	};

	m_indices = {
		0, 1, 2, 2, 3, 0
	};

	initWindow();
	initVulkan();
	mainLoop();
	cleanUp();
}

void Application::initWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	glfwSetWindowUserPointer(m_window, this);
	glfwSetFramebufferSizeCallback(m_window, framebufferResizedCallback);
}

void Application::initVulkan()
{
	createInstance();
	setupDebugCallback();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createRenderPass();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createFramebuffers();
	createCommandPool();
	createTextureImage();
	createTextureImageView();
	createTextureSampler();
	createVertexBuffer();
	createIndexBuffer();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSet();
	createCommandBuffers();
	createSyncObjects();
}

void Application::mainLoop()
{
	while (!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();
		drawFrame();
	}

	m_device.waitIdle();
}

void Application::cleanUp()
{
	cleanupSwapChain();

	m_device.destroyPipeline(m_graphicsPipeline);
	m_device.destroyPipelineLayout(m_pipelineLayout);
	m_device.destroyRenderPass(m_renderPass);

	for (std::size_t i{0}; i < MAX_FRAMES_IN_FLIGHT; i++) {
		m_device.destroyBuffer(m_uniformBuffers[i]);
		m_device.freeMemory(m_uniformBuffersMemory[i]);
	}

	m_device.destroyDescriptorPool(m_descriptorPool);

	m_device.destroySampler(m_textureSampler);
	m_device.destroyImageView(m_textureImageView);
	m_device.destroyImage(m_textureImage);
	m_device.freeMemory(m_textureImageMemory);

	m_device.destroyDescriptorSetLayout(m_descriptorSetLayout);

	m_device.destroyBuffer(m_indexBuffer);
	m_device.freeMemory(m_indexBufferMemory);
	m_device.destroyBuffer(m_vertexBuffer);
	m_device.freeMemory(m_vertexBufferMemory);

	for (std::size_t i{0}; i < MAX_FRAMES_IN_FLIGHT; i++) {
		m_device.destroySemaphore(m_imageAvailabeSemaphores[i]);
		m_device.destroySemaphore(m_renderFinishedSemaphores[i]);
		m_device.destroyFence(m_inFlightFences[i]);
	}

	m_device.destroyCommandPool(m_commandPool);

	m_device.destroy();

	if (m_enableValidationLayers) {
		DestroyDebugUtilsMessengerEXT(m_instance, callback, nullptr);
	}

	m_instance.destroySurfaceKHR(m_surface);
	m_instance.destroy();
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Application::createInstance()
{
	try {
		vk::ApplicationInfo info(application_name.c_str(), 1, engine_name.c_str(), 1, VK_API_VERSION_1_1);

		auto validationLayer = enableValidationLayers();
		auto extensions = getRequiredExtensions();

		vk::InstanceCreateInfo instanceCreateInfo({},
			&info,
			static_cast<uint32_t>(validationLayer.size()),
			validationLayer.size() > 0 ? validationLayer.data() : nullptr,
			static_cast<uint32_t>(extensions.size()),
			extensions.data());

		m_instance = vk::createInstance(instanceCreateInfo);
	}
	catch (vk::SystemError& err) {
		std::cout << "vk::SystemError: " << err.what() << std::endl;
		exit(-1);
	}

	std::uint32_t extensionCount{ 0 };
	auto extensions = vk::enumerateInstanceExtensionProperties();

	std::cout << "\nAvailable extensions:\n";

	for (const auto& ext : extensions) {
		std::cout << "\t" << ext.extensionName << "\n";
	}
}

void Application::createSurface()
{
	VkSurfaceKHR _surface;
	if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &_surface) != VK_SUCCESS) {
		exit(-1);
	}
	m_surface = vk::SurfaceKHR(_surface);
}

void Application::pickPhysicalDevice()
{
	auto devices = m_instance.enumeratePhysicalDevices();
	if (devices.empty()) {
		exit(-1);
	}
	for (const auto& device : devices) {
		if (isSuitableDevice(device)) {
			m_physicalDevice = device;
			break;
		}
	}
	std::cout << "Found device: " << m_physicalDevice.getProperties().deviceName << "\n";
}

void Application::createLogicalDevice()
{
	auto queueFamily = findQueueFamilies(m_physicalDevice);

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::set<std::uint32_t> uniqueQueueFamilies = {
		queueFamily.graphics_family.value(),
		queueFamily.present_family.value()
	};

	for (auto family : uniqueQueueFamilies) {
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, family, 1, &m_queuePriority);
		queueCreateInfos.push_back(deviceQueueCreateInfo);
	}

	vk::DeviceCreateInfo info;
	vk::PhysicalDeviceFeatures features;
	features.setSamplerAnisotropy(vk::True);

	info.queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size());
	info.pQueueCreateInfos = queueCreateInfos.data();
	info.pEnabledFeatures = &features;
	info.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
	info.ppEnabledExtensionNames = deviceExtensions.data();
	info.enabledLayerCount = static_cast<std::uint32_t>(requestedLayers.size());
	info.ppEnabledLayerNames = requestedLayers.data();

	m_device = m_physicalDevice.createDevice(info);

	m_device.getQueue(queueFamily.graphics_family.value(), 0, &m_graphicsQueue);
	m_device.getQueue(queueFamily.present_family.value(), 0, &m_presentQueue);
}

void Application::createImageViews()
{
	m_swapchainImageViews.resize(m_swapchainImages.size());
	for (std::size_t i{ 0 }; i < m_swapchainImages.size(); i++) {
		m_swapchainImageViews[i] = createImageView(m_swapchainImages[i], m_swapchainImageFormat);
	}
}

vk::ImageView Application::createImageView(vk::Image image, vk::Format format)
{
	vk::ImageViewCreateInfo createInfo;
	createInfo
		.setImage(image)
		.setFormat(format)
		.setViewType(vk::ImageViewType::e2D)
		.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
	return m_device.createImageView(createInfo);
}

void Application::createRenderPass()
{
	vk::AttachmentDescription colorAttachment({},
		m_swapchainImageFormat,
		vk::SampleCountFlagBits::e1,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::ePresentSrcKHR);

	vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);
	vk::SubpassDescription subpass;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	vk::RenderPassCreateInfo renderPassCreateInfo({},
		1, &colorAttachment,
		1, &subpass);
	m_renderPass = m_device.createRenderPass(renderPassCreateInfo);

	vk::SubpassDependency dependency;
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

	renderPassCreateInfo.dependencyCount = 1;
	renderPassCreateInfo.pDependencies = &dependency;
}

void Application::createGraphicsPipeline()
{
	auto vertShaderCode = readFile("shaders/vert.spv");
	auto fragShaderCode = readFile("shaders/frag.spv");

	auto vertShaderModule = createShaderModule(vertShaderCode);
	auto fragShaderModule = createShaderModule(fragShaderCode);

	vk::PipelineShaderStageCreateInfo vertShaderCreateInfo({}, vk::ShaderStageFlagBits::eVertex, vertShaderModule, "main");
	vk::PipelineShaderStageCreateInfo fragShaderCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");

	vk::PipelineShaderStageCreateInfo shaderStages[] = {
		vertShaderCreateInfo,
		fragShaderCreateInfo
	};

	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescription();

	vk::PipelineVertexInputStateCreateInfo vertexInputCreateInfo({},
		1, &bindingDescription,
		static_cast<std::uint32_t>(attributeDescriptions.size()),
		attributeDescriptions.data());
	vk::PipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo({}, vk::PrimitiveTopology::eTriangleList, vk::False);

	vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(m_swapchainExtent.width), static_cast<float>(m_swapchainExtent.height), 0.0f, 1.0f);
	vk::Rect2D scissor({ 0, 0 }, m_swapchainExtent);

	std::vector<vk::DynamicState> dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};

	vk::PipelineDynamicStateCreateInfo dynamicStateCreateInfo({}, static_cast<std::uint32_t>(dynamicStates.size()), dynamicStates.data());
	vk::PipelineViewportStateCreateInfo viewPortStateCreateInfo({}, viewport, scissor);

	vk::PipelineRasterizationStateCreateInfo rasterizerCreateInfo;
	rasterizerCreateInfo
		.setDepthClampEnable(vk::False)
		.setRasterizerDiscardEnable(vk::False)
		.setPolygonMode(vk::PolygonMode::eFill)
		.setLineWidth(1.0f)
		.setCullMode(vk::CullModeFlagBits::eBack)
		.setFrontFace(vk::FrontFace::eCounterClockwise)
		.setDepthBiasEnable(vk::False);


	vk::PipelineMultisampleStateCreateInfo multiSamplingCreateInfo;

	vk::PipelineColorBlendAttachmentState colorBlendAttachment(
		vk::False,
		vk::BlendFactor::eOne,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::BlendFactor::eOne,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::ColorComponentFlagBits::eR |
		vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA);

	vk::PipelineColorBlendStateCreateInfo colorBlendCreateInfo({},
		vk::False,
		vk::LogicOp::eCopy,
		1,
		&colorBlendAttachment,
		{ 0.0f, 0.0f, 0.0f, 0.0f });

	vk::PipelineLayoutCreateInfo layoutCreateInfo;
	layoutCreateInfo.setSetLayoutCount(1);
	layoutCreateInfo.setPSetLayouts(&m_descriptorSetLayout);

	try {
		m_pipelineLayout = m_device.createPipelineLayout(layoutCreateInfo);
	}
	catch (vk::SystemError err) {
		std::cout << "Unable to create pipeline layout\n";
		exit(-1);
	}

	vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo;
	graphicsPipelineCreateInfo.stageCount = 2;
	graphicsPipelineCreateInfo.pStages = shaderStages;
	graphicsPipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
	graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
	graphicsPipelineCreateInfo.pViewportState = &viewPortStateCreateInfo;
	graphicsPipelineCreateInfo.pRasterizationState = &rasterizerCreateInfo;
	graphicsPipelineCreateInfo.pMultisampleState = &multiSamplingCreateInfo;
	graphicsPipelineCreateInfo.pColorBlendState = &colorBlendCreateInfo;
	graphicsPipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
	graphicsPipelineCreateInfo.layout = m_pipelineLayout;
	graphicsPipelineCreateInfo.renderPass = m_renderPass;
	graphicsPipelineCreateInfo.subpass = 0;
	graphicsPipelineCreateInfo.basePipelineHandle = nullptr;

	try {
		m_graphicsPipeline = m_device.createGraphicsPipeline(nullptr, graphicsPipelineCreateInfo).value;
	}
	catch (vk::SystemError err) {
		std::cout << "Unable to create graphics pipeline\n";
		exit(-1);
	}

	m_device.destroyShaderModule(vertShaderModule);
	m_device.destroyShaderModule(fragShaderModule);
}

void Application::createDescriptorSetLayout()
{
	vk::DescriptorSetLayoutBinding uboLayoutBinding;
	uboLayoutBinding
		.setBinding(0)
		.setDescriptorCount(1)
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setPImmutableSamplers(nullptr)
		.setStageFlags(vk::ShaderStageFlagBits::eVertex);

	vk::DescriptorSetLayoutBinding samplerLayoutBinding;
	samplerLayoutBinding
		.setBinding(1)
		.setDescriptorCount(1)
		.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
		.setPImmutableSamplers(nullptr)
		.setStageFlags(vk::ShaderStageFlagBits::eFragment);

	std::array<vk::DescriptorSetLayoutBinding, 2> bindings =
		{ uboLayoutBinding , samplerLayoutBinding };

	vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
	descriptorSetLayoutCreateInfo
		.setBindingCount(static_cast<std::uint32_t>(bindings.size()))
		.setPBindings(bindings.data());
	m_descriptorSetLayout = m_device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
}

void Application::createFramebuffers()
{
	m_swapChainFramebuffers.resize(m_swapchainImageViews.size());

	for (std::size_t i{ 0 }; i < m_swapchainImageViews.size(); i++) {
		vk::ImageView attachments[] = {
			m_swapchainImageViews[i]
		};

		vk::FramebufferCreateInfo frameBufferCreateInfo({},
			m_renderPass,
			1,
			attachments,
			m_swapchainExtent.width,
			m_swapchainExtent.height,
			1);
		m_swapChainFramebuffers[i] = m_device.createFramebuffer(frameBufferCreateInfo);
	}
}

void Application::createCommandPool()
{
	auto queueFamilyIndices = findQueueFamilies(m_physicalDevice);

	vk::CommandPoolCreateInfo commandPoolCreateInfo(
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		queueFamilyIndices.graphics_family.value()
	);

	m_commandPool = m_device.createCommandPool(commandPoolCreateInfo);
}

void Application::createTextureImage()
{
	int textureWidth, textureHeight, textureChannels;
	auto pixels = stbi_load("textures/texture.jpg",
		&textureWidth, &textureHeight, &textureChannels,
		STBI_rgb_alpha
	);

	vk::DeviceSize imageSize = textureWidth * textureHeight * 4;

	if (!pixels) {
		throw std::runtime_error("Unable to locate texture\n");
	}

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	auto data = m_device.mapMemory(stagingBufferMemory, 0, imageSize);
	memcpy(data, pixels, static_cast<std::size_t>(imageSize));
	m_device.unmapMemory(stagingBufferMemory);

	stbi_image_free(pixels);

	createImage(textureWidth, textureHeight,
		vk::Format::eR8G8B8A8Srgb,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, 
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		m_textureImage,
		m_textureImageMemory);

	transitionImageLayout(m_textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(stagingBuffer, m_textureImage, static_cast<std::uint32_t>(textureWidth), static_cast<std::uint32_t>(textureHeight));
	transitionImageLayout(m_textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

	m_device.destroyBuffer(stagingBuffer);
	m_device.freeMemory(stagingBufferMemory);
}

void Application::createTextureImageView()
{
	m_textureImageView = createImageView(m_textureImage, vk::Format::eR8G8B8A8Srgb);
}

void Application::createTextureSampler()
{
	vk::SamplerCreateInfo samplerCreateInfo;
	samplerCreateInfo
		.setMagFilter(vk::Filter::eLinear)
		.setMinFilter(vk::Filter::eLinear)
		.setAddressModeU(vk::SamplerAddressMode::eRepeat)
		.setAddressModeV(vk::SamplerAddressMode::eRepeat)
		.setAddressModeW(vk::SamplerAddressMode::eRepeat)
		.setAnisotropyEnable(vk::True)
		.setMaxAnisotropy(m_physicalDevice.getProperties().limits.maxSamplerAnisotropy)
		.setBorderColor(vk::BorderColor::eIntOpaqueBlack)
		.setUnnormalizedCoordinates(vk::False)
		.setCompareEnable(vk::False)
		.setCompareOp(vk::CompareOp::eAlways)
		.setMipmapMode(vk::SamplerMipmapMode::eLinear)
		.setMipLodBias(0.0f)
		.setMinLod(0.0f)
		.setMaxLod(0.0f);

	m_textureSampler = m_device.createSampler(samplerCreateInfo);
}

void Application::createImage(
	std::uint32_t width, std::uint32_t height,
	vk::Format format, vk::ImageTiling tiling,
	vk::ImageUsageFlags usageFlagBits, vk::MemoryPropertyFlagBits memoryPropertyBits,
	vk::Image& textureImage, vk::DeviceMemory& imageMemory)
{
	vk::ImageCreateInfo imageCreateInfo;
	imageCreateInfo
		.setImageType(vk::ImageType::e2D)
		.setExtent({ static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1 })
		.setMipLevels(1)
		.setArrayLayers(1)
		.setFormat(format)
		.setTiling(tiling)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setUsage(usageFlagBits)
		.setSharingMode(vk::SharingMode::eExclusive)
		.setSamples(vk::SampleCountFlagBits::e1);

	textureImage = m_device.createImage(imageCreateInfo);

	auto memoryRequirements = m_device.getImageMemoryRequirements(textureImage);

	vk::MemoryAllocateInfo allocateInfo;
	allocateInfo
		.setAllocationSize(memoryRequirements.size)
		.setMemoryTypeIndex(findMemoryType(memoryRequirements.memoryTypeBits, memoryPropertyBits));
	imageMemory = m_device.allocateMemory(allocateInfo);
	m_device.bindImageMemory(textureImage, imageMemory, 0);
}

void Application::transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
	auto commandBuffer = beginOneTimeCommands();

	vk::ImageMemoryBarrier barrier;
	barrier
		.setOldLayout(oldLayout)
		.setNewLayout(newLayout)
		.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
		.setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
		.setImage(image)
		.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

	vk::PipelineStageFlags sourceStage, destinationStage;

	if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
		barrier
			.setSrcAccessMask(vk::AccessFlagBits::eNone)
			.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
		sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage = vk::PipelineStageFlagBits::eTransfer;

	} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
		barrier
			.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
			.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
		sourceStage = vk::PipelineStageFlagBits::eTransfer;
		destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
	} else {
		throw std::runtime_error("Unsupported layer transition\n");
	}

	commandBuffer.pipelineBarrier(
		sourceStage, 
		destinationStage,
		vk::DependencyFlagBits(0U),
		0, nullptr,
		0, nullptr,
		1, &barrier);

	endOneTimeCommands(commandBuffer);
}

void Application::createCommandBuffers()
{
	vk::CommandBufferAllocateInfo allocateInfo(
		m_commandPool,
		vk::CommandBufferLevel::ePrimary,
		MAX_FRAMES_IN_FLIGHT);
	m_commandBuffers = m_device.allocateCommandBuffers(allocateInfo);
}

void Application::recordCommandBuffer(vk::CommandBuffer commandBuffer, std::uint32_t imageIndex)
{
	vk::CommandBufferBeginInfo beginInfo;
	commandBuffer.begin(beginInfo);

	vk::ClearValue clearValue = vk::ClearValue{ vk::ClearColorValue{ std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f} } };
	vk::RenderPassBeginInfo renderPassBeginInfo(
		m_renderPass,
		m_swapChainFramebuffers[imageIndex],
		vk::Rect2D(vk::Offset2D(), m_swapchainExtent),
		1, &clearValue);

	commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);

	vk::Buffer vertexBuffers[] = { m_vertexBuffer };
	vk::DeviceSize offsets[] = { 0 };

	commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
	commandBuffer.bindIndexBuffer(m_indexBuffer, 0, vk::IndexType::eUint16);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, 1, &m_descriptorSets[m_currentFrame], 0, nullptr);

	vk::Viewport viewport(
		0.0f,
		0.0f,
		static_cast<float>(m_swapchainExtent.width),
		static_cast<float>(m_swapchainExtent.height),
		0.0f,
		1.0f
	);
	commandBuffer.setViewport(0, viewport);

	vk::Rect2D scissor({ 0,0 }, m_swapchainExtent);
	commandBuffer.setScissor(0, scissor);
	commandBuffer.drawIndexed(static_cast<std::uint32_t>(m_indices.size()), 1, 0, 0, 0);
	commandBuffer.endRenderPass();
	commandBuffer.end();
}

void Application::createSyncObjects()
{
	vk::SemaphoreCreateInfo createSemaphoreInfo;
	vk::FenceCreateInfo fenceCreateInfo(vk::FenceCreateFlagBits::eSignaled);

	for (std::size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; i++) {
		m_imageAvailabeSemaphores.push_back(m_device.createSemaphore(createSemaphoreInfo));
		m_renderFinishedSemaphores.push_back(m_device.createSemaphore(createSemaphoreInfo));
		m_inFlightFences.push_back(m_device.createFence(fenceCreateInfo));
	}
}

void Application::recreateSwapChain()
{
	int width{0}, height{0};
	glfwGetFramebufferSize(m_window, &width, &height);
	if (width == 0 || height == 0) {
		glfwGetFramebufferSize(m_window, &width, &height);
		glfwWaitEvents();
	}

	m_device.waitIdle();
	cleanupSwapChain();
	createSwapChain();
	createImageViews();
	createFramebuffers();
}

void Application::cleanupSwapChain()
{
	for (auto framebuffer : m_swapChainFramebuffers) {
		m_device.destroy(framebuffer);
	}

	for (auto imageView : m_swapchainImageViews) {
		m_device.destroyImageView(imageView);
	}

	m_device.destroySwapchainKHR(m_swapchain);
}

void Application::createBuffer(vk::DeviceSize size,
	vk::BufferUsageFlags usage,
	vk::MemoryPropertyFlags properties,
	vk::Buffer& buffer,
	vk::DeviceMemory& bufferMemory)
{
	vk::BufferCreateInfo bufferCreateInfo({},
		size,
		usage,
		vk::SharingMode::eExclusive);

	buffer = m_device.createBuffer(bufferCreateInfo);

	auto memoryRequirements = m_device.getBufferMemoryRequirements(buffer);

	vk::MemoryAllocateInfo allocateInfo(
		memoryRequirements.size,
		findMemoryType(memoryRequirements.memoryTypeBits,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
	bufferMemory = m_device.allocateMemory(allocateInfo);
	m_device.bindBufferMemory(buffer, bufferMemory, 0);
}

void Application::createVertexBuffer()
{
	vk::DeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;

	createBuffer(bufferSize, 
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer,
		stagingBufferMemory);

	void* data = m_device.mapMemory(stagingBufferMemory, 0, bufferSize);
	memcpy(data, m_vertices.data(), (std::size_t)bufferSize);
	m_device.unmapMemory(stagingBufferMemory);

	createBuffer(bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		m_vertexBuffer,
		m_vertexBufferMemory);

	copyBuffer(stagingBuffer, m_vertexBuffer, bufferSize);
	m_device.destroyBuffer(stagingBuffer);
	m_device.freeMemory(stagingBufferMemory);
}

void Application::createIndexBuffer()
{
	vk::DeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;

	createBuffer(bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer,
		stagingBufferMemory);

	void* data = m_device.mapMemory(stagingBufferMemory, 0, bufferSize);
	memcpy(data, m_indices.data(), (std::size_t)bufferSize);
	m_device.unmapMemory(stagingBufferMemory);

	createBuffer(bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		m_indexBuffer,
		m_indexBufferMemory);

	copyBuffer(stagingBuffer, m_indexBuffer, bufferSize);
	m_device.destroyBuffer(stagingBuffer);
	m_device.freeMemory(stagingBufferMemory);
}

void Application::createUniformBuffers()
{
	vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
	m_uniformBuffers.resize(bufferSize);
	m_uniformBuffersMemory.resize(bufferSize);
	m_uniformBuffersMapped.resize(bufferSize);

	for (std::size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; i++) {
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_uniformBuffers[i], m_uniformBuffersMemory[i]);
		m_uniformBuffersMapped[i] = m_device.mapMemory(m_uniformBuffersMemory[i], 0, bufferSize);
	}
}

void Application::createDescriptorPool()
{
	std::array<vk::DescriptorPoolSize, 2> poolSizes{};
	poolSizes[0]
		.setType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT));
	poolSizes[1]
		.setType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT));

	vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
	descriptorPoolCreateInfo
		.setPoolSizeCount(static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT))
		.setPPoolSizes(poolSizes.data())
		.setMaxSets(static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT));

	m_descriptorPool = m_device.createDescriptorPool(descriptorPoolCreateInfo);
}

void Application::createDescriptorSet()
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocateInfo(m_descriptorPool, layouts);
	m_descriptorSets = m_device.allocateDescriptorSets(allocateInfo);

	for (std::size_t i{0}; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::DescriptorBufferInfo bufferInfo;
		bufferInfo
			.setBuffer(m_uniformBuffers[i])
			.setOffset(0)
			.setRange(sizeof(UniformBufferObject));

		vk::DescriptorImageInfo imageInfo;
		imageInfo
			.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
			.setImageView(m_textureImageView)
			.setSampler(m_textureSampler);

		std::array<vk::WriteDescriptorSet, 2> descriptorWrites;
		descriptorWrites[0]
			.setDstSet(m_descriptorSets[i])
			.setDstBinding(0)
			.setDstArrayElement(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setPBufferInfo(&bufferInfo);
		descriptorWrites[1]
			.setDstSet(m_descriptorSets[i])
			.setDstBinding(1)
			.setDstArrayElement(0)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setDescriptorCount(1)
			.setPImageInfo(&imageInfo);
		m_device.updateDescriptorSets(descriptorWrites, nullptr);
	}
}

void Application::copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size)
{
	auto commandBuffer = beginOneTimeCommands();

	vk::BufferCopy copyRegion;
	copyRegion.setSize(size);

	commandBuffer.copyBuffer(src, dst, copyRegion);

	endOneTimeCommands(commandBuffer);
}

std::vector<char> Application::readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}
	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

void Application::copyBufferToImage(vk::Buffer buffer, vk::Image image, std::uint32_t width, std::uint32_t height)
{
	auto commandBuffer = beginOneTimeCommands();
	vk::BufferImageCopy region;
	region
		.setBufferOffset(0)
		.setBufferRowLength(0)
		.setBufferImageHeight(0)
		.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1))
		.setImageOffset({ 0, 0, 0 })
		.setImageExtent({ width, height, 1 });
	commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
	endOneTimeCommands(commandBuffer);
}

vk::ShaderModule Application::createShaderModule(const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo({}, code.size(), reinterpret_cast<const uint32_t*>(code.data()));
	return m_device.createShaderModule(createInfo);
}

bool Application::isSuitableDevice(vk::PhysicalDevice device)
{
	QueueFamilyIndices indices = findQueueFamilies(device);
	if (!indices.graphics_family.has_value()) {
		return false;
	}

	bool extensionSupported = deviceHasExtensionSupport(device);

	vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
	vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();

	bool swapChainAdequate{ false };
	if (extensionSupported) {
		auto details = querySwapChainSupport(device);
		swapChainAdequate = !details.formats.empty() && !details.presentModes.empty();
	}

	return deviceFeatures.geometryShader &&
		deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
		swapChainAdequate;
}

bool Application::deviceHasExtensionSupport(vk::PhysicalDevice device)
{
	auto availableExtensions = device.enumerateDeviceExtensionProperties();
	std::set<std::string> requiredExtensions{ deviceExtensions.begin(), deviceExtensions.end() };

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

QueueFamilyIndices Application::findQueueFamilies(vk::PhysicalDevice device)
{
	QueueFamilyIndices indices{ 0 };
	auto queueFamilies = device.getQueueFamilyProperties();
	std::uint32_t i{ 0 };
	for (const auto& queueFamily : queueFamilies) {

		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
			indices.graphics_family = i;
		}

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);

		if (presentSupport) {
			indices.present_family = i;
		}
		i++;
	}
	return indices;
}

vk::SurfaceFormatKHR Application::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
			availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
		{
			return availableFormat;
		}
	}
	return availableFormats.front();
}

vk::PresentModeKHR Application::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availableModes)
{
	for (const auto& availableMode : availableModes) {
		if (availableMode == vk::PresentModeKHR::eMailbox) {
			return availableMode;
		}
	}
	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Application::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != (std::numeric_limits<std::uint32_t>::max)()) {
		return capabilities.currentExtent;
	}
	else {
		int width, height;
		glfwGetFramebufferSize(m_window, &width, &height);

		vk::Extent2D actual{
			static_cast<std::uint32_t>(width),
			static_cast<std::uint32_t>(height)
		};

		actual.width = std::clamp(actual.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actual.height = std::clamp(actual.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
		return actual;
	}

	return {};
}

void Application::createSwapChain()
{
	auto swapChainSupportDetails = querySwapChainSupport(m_physicalDevice);
	auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupportDetails.formats);
	auto presentMode = chooseSwapPresentMode(swapChainSupportDetails.presentModes);
	auto extent = chooseSwapExtent(swapChainSupportDetails.capabilties);

	auto imageCount = swapChainSupportDetails.capabilties.minImageCount + 1;

	if (swapChainSupportDetails.capabilties.maxImageCount > 0 &&
		imageCount > swapChainSupportDetails.capabilties.maxImageCount)
	{
		imageCount = swapChainSupportDetails.capabilties.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR createInfo({},
		m_surface,
		imageCount,
		surfaceFormat.format,
		surfaceFormat.colorSpace,
		extent,
		1,
		vk::ImageUsageFlagBits::eColorAttachment);

	auto indices = findQueueFamilies(m_physicalDevice);
	std::array<std::uint32_t, 2> queueFamilyIndices = { indices.graphics_family.value(), indices.present_family.value() };
	if (indices.graphics_family != indices.present_family) {
		createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
	}
	else {
		createInfo.imageSharingMode = vk::SharingMode::eExclusive;
		createInfo.queueFamilyIndexCount = 0;
		createInfo.pQueueFamilyIndices = nullptr;
	}

	createInfo.preTransform = swapChainSupportDetails.capabilties.currentTransform;
	createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	createInfo.presentMode = presentMode;
	createInfo.clipped = vk::True;
	createInfo.oldSwapchain = nullptr;

	m_swapchain = m_device.createSwapchainKHR(createInfo);
	m_swapchainImageFormat = surfaceFormat.format;
	m_swapchainExtent = extent;
	m_swapchainImages = m_device.getSwapchainImagesKHR(m_swapchain);
}

void Application::drawFrame()
{
	auto result = m_device.waitForFences(m_inFlightFences[m_currentFrame], vk::True, (std::numeric_limits<std::uint64_t>::max)());
	auto aquireNextImageResult = m_device.acquireNextImageKHR(
		m_swapchain,
		(std::numeric_limits<std::uint64_t>::max)(),
		m_imageAvailabeSemaphores[m_currentFrame],
		nullptr
	);

	if (aquireNextImageResult.result == vk::Result::eErrorOutOfDateKHR) {
		recreateSwapChain();
		return;
	} else if (aquireNextImageResult.result != vk::Result::eSuccess &&
				aquireNextImageResult.result != vk::Result::eSuboptimalKHR)
	{
		throw std::runtime_error("Failed to aquire swap chain image\n");
	}

	m_device.resetFences(m_inFlightFences[m_currentFrame]);

	auto imageIndex = aquireNextImageResult.value;
	m_commandBuffers[m_currentFrame].reset();
	recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

	vk::Semaphore waitSemaphores[]{ m_imageAvailabeSemaphores[m_currentFrame] };
	vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	vk::Semaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrame] };

	updateUniformBuffer(m_currentFrame);

	vk::SubmitInfo submitInfo(
		1,
		waitSemaphores,
		waitStages,
		1,
		&m_commandBuffers[m_currentFrame],
		1,
		signalSemaphores
	);

	m_graphicsQueue.submit(submitInfo, m_inFlightFences[m_currentFrame]);

	vk::SwapchainKHR swapchains[] = { m_swapchain };
	vk::PresentInfoKHR presentInfo(
		1, signalSemaphores,
		1, swapchains,
		&imageIndex
	);

	auto presentResult = m_presentQueue.presentKHR(&presentInfo);

	if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || m_framebufferResized) {
		recreateSwapChain();
		m_framebufferResized = false;
	}

	m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Application::updateUniformBuffer(std::uint32_t currentImage)
{
	static auto startTime = std::chrono::high_resolution_clock::now();
	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	UniformBufferObject ubo;
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f), m_swapchainExtent.width / (float)m_swapchainExtent.height, 0.1f, 10.0f);

	ubo.proj[1][1] *= -1;
	memcpy(m_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

vk::CommandBuffer Application::beginOneTimeCommands()
{
	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo
		.setLevel(vk::CommandBufferLevel::ePrimary)
		.setCommandPool(m_commandPool)
		.setCommandBufferCount(1);
	auto commandBuffer = m_device.allocateCommandBuffers(allocInfo).front();
	vk::CommandBufferBeginInfo beginInfo;
	beginInfo
		.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	commandBuffer.begin(beginInfo);
	return commandBuffer;
}

void Application::endOneTimeCommands(vk::CommandBuffer commandBuffer)
{
	commandBuffer.end();

	vk::SubmitInfo submitInfo;
	submitInfo
		.setCommandBufferCount(1)
		.setPCommandBuffers(&commandBuffer);
	m_graphicsQueue.submit(submitInfo);
	m_graphicsQueue.waitIdle();
	m_device.freeCommandBuffers(m_commandPool, commandBuffer);
}

SwapChainSupportDetails Application::querySwapChainSupport(vk::PhysicalDevice device)
{
	SwapChainSupportDetails details;
	if (device.getSurfaceCapabilitiesKHR(m_surface, &details.capabilties) != vk::Result::eSuccess) { exit(-1); }
	details.presentModes = device.getSurfacePresentModesKHR(m_surface);
	details.formats = device.getSurfaceFormatsKHR(m_surface);
	return details;
}

std::uint32_t Application::findMemoryType(std::uint32_t filter, vk::MemoryPropertyFlags properties)
{
	auto physicalDeviceMemoryProperties = m_physicalDevice.getMemoryProperties();
	for (std::uint32_t i{ 0 }; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
		if (filter & (1 << i) &&
			physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & properties)
		{
			return i;
		}
	}

	throw std::runtime_error("Unable to find suitable memory type!\n");
}

void Application::setupDebugCallback()
{
	if (!m_enableValidationLayers) return;

	auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT(
		vk::DebugUtilsMessengerCreateFlagsEXT(),
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
		debugCallback,
		nullptr
	);

	if (CreateDebugUtilsMessengerEXT(m_instance,
		reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&createInfo),
		nullptr,
		&callback) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to set up debug callback!");
	}
}

VKAPI_ATTR VkBool32 VKAPI_CALL Application::debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	std::cout << "validation layer: " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

std::vector<const char*> Application::enableValidationLayers()
{
	if (!m_enableValidationLayers) {
		return {};
	}

	auto instanceLayerProperties = vk::enumerateInstanceLayerProperties();

	if (instanceLayerProperties.empty()) {
		std::cout << "Validation Layers requested but none available.\n";
		return {};
	}

	for (const auto& layerName : requestedLayers) {
		bool found{ false };
		for (const auto& layerProperty : instanceLayerProperties) {
			if (strcmp(layerName, layerProperty.layerName) == 0) {
				found = true;
				break;
			}
		}
		if (found) { break; }
	}

	return requestedLayers;
}

std::vector<const char*> Application::getRequiredExtensions()
{
	std::uint32_t glfwExtensionCount{ 0 };
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (m_enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return extensions;
}