project "NeuralNetwork"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "off"

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
