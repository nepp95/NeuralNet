set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# EppoCore locates Vulkan, shaderc and SPIRV-Cross through the SDK when VULKAN_SDK is set
set(VCPKG_ENV_PASSTHROUGH VULKAN_SDK)
