project "NeuralNetwork"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"

    targetdir ("%{wks.location}/build/bin/" .. OutputDir .. "/%{prj.name}")
    objdir ("%{wks.location}/build/obj/" .. OutputDir .. "/%{prj.name}")

    files
    {
        "Source/**.hpp",
        "Source/**.cpp",
    }

    includedirs
    {
        Dependencies.IncludeDirectories.Library,
        Dependencies.IncludeDirectories.Vcpkg,
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines "NN_DEBUG"
        symbols "on"

    filter "configurations:not Debug"
        defines "NN_RELEASE"
        optimize "on"

    filter { "system:windows", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)
        links (Dependencies.WindowsDebugLinks)

    filter { "system:windows", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
        links (Dependencies.WindowsReleaseLinks)

    filter { "system:linux", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)
        links (Dependencies.LinuxLinks)

    filter { "system:linux", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
        links (Dependencies.LinuxLinks)
