project "NeuralNetworkUI"
    kind "ConsoleApp"
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

    links "NeuralNetwork"

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        symbols "on"

    filter "configurations:not Debug"
        optimize "on"

    -- The vcpkg libraries resolve at final link time, so the import library
    -- search paths and GLFW's runtime DLL must follow the executable.
    filter { "system:windows", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)
        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Debug .. '/glfw3.dll" "%{cfg.targetdir}"',
        }

    filter { "system:windows", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
        postbuildcommands
        {
            '{COPY} "' .. Dependencies.RuntimeDirectories.Release .. '/glfw3.dll" "%{cfg.targetdir}"',
        }

    filter { "system:linux", "configurations:Debug" }
        libdirs (Dependencies.LibraryDirectories.Debug)

    filter { "system:linux", "configurations:not Debug" }
        libdirs (Dependencies.LibraryDirectories.Release)
