set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

# EppoCore locates Vulkan, shaderc and SPIRV-Cross through the SDK
set(VCPKG_ENV_PASSTHROUGH VULKAN_SDK)
