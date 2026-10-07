NeuralNetworkRoot = path.getabsolute(".")
dofile("Scripts/Premake/Dependencies.lua")

workspace "NeuralNetwork"
    configurations { "Debug", "Release", "Dist" }
    platforms "x64"
    startproject "NeuralNetworkUI"

    OutputDir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

    include "NeuralNetwork"
    include "NeuralNetworkUI"
