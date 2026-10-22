local vcpkgTriplet = os.host() == "windows" and "x64-windows" or "x64-linux"
local vcpkgInstalled = Root .. "/build/vcpkg_installed/" .. os.host() .. "/" .. vcpkgTriplet

Dependencies = {
    IncludeDirectories = {
        Glad = Root .. "/Vendor/glad/include",
        Stb = Root .. "/Vendor/stb",
        Implot = Root .. "/Vendor/implot/include",
        Library = Root .. "/NeuralNetwork/Source",
        Tracy = vcpkgInstalled .. "/include/tracy",
        Vcpkg = vcpkgInstalled .. "/include",
    },
    LibraryDirectories = {
        Debug = vcpkgInstalled .. "/debug/lib",
        DebugTracy = vcpkgInstalled .. "/debug/lib/Debug",
        Release = vcpkgInstalled .. "/lib",
    },
    RuntimeDirectories = {
        Debug = vcpkgInstalled .. "/debug/bin",
        DebugTracy = vcpkgInstalled .. "/debug/bin/Debug",
        Release = vcpkgInstalled .. "/bin",
    },
}
