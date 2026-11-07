project "NeuralNetworkUI"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"
    staticruntime "off"

    -- See NeuralNetwork/premake5.lua: the ninja module needs the project location at the workspace root for pch paths
    if _ACTION == "ninja" then
        location (Root)
    end

    debugdir "%{wks.location}"
    targetdir ("%{wks.location}/build/bin/" .. OutputDir .. "/%{prj.name}")
    objdir ("%{wks.location}/build/obj/" .. OutputDir .. "/%{prj.name}")

    files
    {
        "Source/**.h",
        "Source/**.cpp",
    }

    pchheader "pch.h"
    pchsource "Source/pch.cpp"

    includedirs
    {
        "Source",
        Dependencies.IncludeDirectories.Implot,
        Dependencies.IncludeDirectories.Library,
        Dependencies.IncludeDirectories.Tracy,
        Dependencies.IncludeDirectories.Vcpkg,
        Dependencies.IncludeDirectories.VulkanSDK,
    }

    defines
    {
        "EP_GUI",
        "TRACY_ENABLE",
        "TRACY_ON_DEMAND",
    }

    links
    {
        "NeuralNetwork",
        "implot",
    }

    filter { "system:windows", "configurations:Debug" }
        libdirs
        {
            Dependencies.LibraryDirectories.Debug,
            Dependencies.LibraryDirectories.DebugTracy,
            Dependencies.LibraryDirectories.VulkanSDK,
        }

        links
        {
            "EppoCore_GUI",
            "glfw3dll",
            "imguid",
            "spdlogd",
            "fmtd",
            "TracyClient",
            "vulkan-1",
            "shaderc_combinedd",
            "spirv-cross-cored",
        }

        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Debug .. '/glfw3.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.Debug .. '/spdlogd.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.Debug .. '/fmtd.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.DebugTracy .. '/TracyClient.dll" "%{cfg.targetdir}"',
        }

    filter { "system:windows", "configurations:not Debug" }
        libdirs
        {
            Dependencies.LibraryDirectories.Release,
            Dependencies.LibraryDirectories.VulkanSDK,
        }

        links
        {
            "EppoCore_GUI",
            "glfw3dll",
            "imgui",
            "spdlog",
            "fmt",
            "TracyClient",
            "vulkan-1",
            "shaderc_combined",
            "spirv-cross-core",
        }

        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/glfw3.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/spdlog.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/fmt.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/TracyClient.dll" "%{cfg.targetdir}"',
        }

    -- Static archives: GNU ld resolves left to right, so each library comes before the ones it uses (imgui's GLFW
    -- backend needs glfw3 after it). Debug names match Windows: imgui/spdlog/fmt add a "d" postfix and Tracy installs
    -- into debug/lib/Debug on every platform.
    filter { "system:linux", "configurations:Debug" }
        libdirs
        {
            Dependencies.LibraryDirectories.Debug,
            Dependencies.LibraryDirectories.DebugTracy,
            Dependencies.LibraryDirectories.VulkanSDK,
        }

        links
        {
            "EppoCore_GUI",
            "imguid",
            "glfw3",
            "spdlogd",
            "fmtd",
            "TracyClient",
            "vulkan",
            "shaderc_combined",
            "spirv-cross-core",
        }

    filter { "system:linux", "configurations:not Debug" }
        libdirs
        {
            Dependencies.LibraryDirectories.Release,
            Dependencies.LibraryDirectories.VulkanSDK,
        }

        links
        {
            "EppoCore_GUI",
            "imgui",
            "glfw3",
            "spdlog",
            "fmt",
            "TracyClient",
            "vulkan",
            "shaderc_combined",
            "spirv-cross-core",
        }
