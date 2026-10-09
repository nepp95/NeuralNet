project "NeuralNetworkUI"
    kind "ConsoleApp"
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
        Dependencies.IncludeDirectories.Glad,
        Dependencies.IncludeDirectories.Stb,
        Dependencies.IncludeDirectories.Implot,
        Dependencies.IncludeDirectories.Library,
        Dependencies.IncludeDirectories.Vcpkg,
    }

    links
    {
        "glad",
        "implot",
        "NeuralNetwork"
    }

    filter { "system:windows", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)
        links { "glfw3dll", "imguid" }
        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Debug .. '/glfw3.dll" "%{cfg.targetdir}"',
        }

    filter { "system:windows", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
        links { "glfw3dll", "imgui" }
        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/glfw3.dll" "%{cfg.targetdir}"',
        }

    filter { "system:linux", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)
        links { "glfw3", "imgui" }

    filter { "system:linux", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
        links { "glfw3", "imgui" }
