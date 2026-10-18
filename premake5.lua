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

    include "Vendor/glad"
    include "Vendor/implot"
    include "NeuralNetwork"
    include "NeuralNetworkUI"
