project "NeuralNetworkUI"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"
    staticruntime "off"

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
        Dependencies.IncludeDirectories.Glad,
        Dependencies.IncludeDirectories.Implot,
        Dependencies.IncludeDirectories.Stb,
        Dependencies.IncludeDirectories.Library,
        Dependencies.IncludeDirectories.Tracy,
        Dependencies.IncludeDirectories.Vcpkg,
    }

    defines {
        "TRACY_ENABLE"
    }

    links
    {
        "glad",
        "NeuralNetwork"
    }

    filter { "system:windows", "configurations:Debug" }
        libdirs { Dependencies.LibraryDirectories.Debug, Dependencies.LibraryDirectories.DebugTracy }
        links { "glfw3dll", "imguid", "TracyClient" }
        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Debug .. '/glfw3.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.DebugTracy .. '/TracyClient.dll" "%{cfg.targetdir}"',
        }

    filter { "system:windows", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
        links { "glfw3dll", "imgui", "TracyClient" }
        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/glfw3.dll" "%{cfg.targetdir}"',
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/TracyClient.dll" "%{cfg.targetdir}"',
        }

    filter { "system:linux", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)
        links { "glfw3", "imgui", "TracyClient" }

    filter { "system:linux", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
        links { "glfw3", "imgui", "TracyClient" }
