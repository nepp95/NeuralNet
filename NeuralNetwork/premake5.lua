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
        Dependencies.IncludeDirectories.Glad,
        Dependencies.IncludeDirectories.Implot,
        Dependencies.IncludeDirectories.Library,
        Dependencies.IncludeDirectories.Stb,
        Dependencies.IncludeDirectories.Tracy,
        Dependencies.IncludeDirectories.Vcpkg,
    }

    defines {
        "TRACY_ENABLE",
        "TRACY_ON_DEMAND"
    }

    links {
        "glad",
        "implot"
    }

    filter { "system:windows", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)

    filter { "system:windows", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)

    filter { "system:linux", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)

    filter { "system:linux", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
