Root = path.getabsolute(".")
dofile("Scripts/Premake/Dependencies.lua")

workspace "NeuralNetwork"
    configurations { "Debug", "Release" }
    platforms "x64"
    startproject "NeuralNetworkUI"

    OutputDir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

    include "Vendor/glad"
    include "NeuralNetwork"
    include "NeuralNetworkUI"
