Root = path.getabsolute(".")
dofile("Scripts/Premake/Dependencies.lua")

workspace "NeuralNetwork"
    configurations { "Debug", "Release" }
    platforms "x64"
    startproject "NeuralNetworkUI"

    OutputDir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

    filter { "configurations:Debug" }
        symbols "On"

    filter { "configurations:Release" }
        optimize "Speed"
        symbols "On"

    filter {}

    -- spdlog/fmt usage requirements from vcpkg's CMake targets (premake can't read them). Without these, code that
    -- logs compiles spdlog header-only and collides with the prebuilt spdlog library at link time.
    defines { "SPDLOG_COMPILED_LIB", "SPDLOG_FMT_EXTERNAL" }

    filter { "system:windows" }
        buildoptions { "/utf-8" }
        defines { "SPDLOG_SHARED_LIB", "FMT_SHARED" } -- x64-windows triplet links dynamically
        disablewarnings { "4251", "4275" } -- STL members/bases in spdlog's dll-exported classes

    filter {}

    include "Vendor/implot"
    include "NeuralNetwork"
    include "NeuralNetworkUI"
