project "implot"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "off"

    targetdir ("%{wks.location}/build/bin/" .. OutputDir .. "/%{prj.name}")
    objdir ("%{wks.location}/build/obj/" .. OutputDir .. "/%{prj.name}")

    files
    {
        "src/**.cpp",
        "include/**.h",
    }

    includedirs
    {
        "include",
        Dependencies.IncludeDirectories.Vcpkg,
    }
