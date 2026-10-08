project "glad"
    kind "StaticLib"
    language "C"
    staticruntime "off"

    targetdir ("%{wks.location}/build/bin/" .. OutputDir .. "/%{prj.name}")
    objdir ("%{wks.location}/build/obj/" .. OutputDir .. "/%{prj.name}")

    files
    {
        "src/**.c",
        "include/**.h",
    }

    includedirs
    {
        "include",
    }
