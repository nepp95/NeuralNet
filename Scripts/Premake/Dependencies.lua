local system = os.host()
local vcpkgTriplet = system == "windows" and "x64-windows" or "x64-linux"
local vcpkgInstalled = Root .. "/build/vcpkg_installed/" .. system .. "/" .. vcpkgTriplet

local vulkanSdk = os.getenv("VULKAN_SDK")
if not vulkanSdk or vulkanSdk == "" then
    error("VULKAN_SDK is not set. Install the Vulkan SDK and reopen the terminal.")
end

local vulkanInclude = vulkanSdk .. (system == "windows" and "/Include" or "/include")
local vulkanLibrary = vulkanSdk .. (system == "windows" and "/Lib" or "/lib")

Dependencies = {
    IncludeDirectories = {
        Implot = Root .. "/Vendor/implot/include",
        Library = Root .. "/NeuralNetwork/Source",
        Tracy = vcpkgInstalled .. "/include/tracy",
        Vcpkg = vcpkgInstalled .. "/include",
        VulkanSDK = vulkanInclude,
    },
    LibraryDirectories = {
        Debug = vcpkgInstalled .. "/debug/lib",
        DebugTracy = vcpkgInstalled .. "/debug/lib/Debug",
        Release = vcpkgInstalled .. "/lib",
        VulkanSDK = vulkanLibrary,
    },
    RuntimeDirectories = {
        Debug = vcpkgInstalled .. "/debug/bin",
        DebugTracy = vcpkgInstalled .. "/debug/bin/Debug",
        Release = vcpkgInstalled .. "/bin",
    },
}
