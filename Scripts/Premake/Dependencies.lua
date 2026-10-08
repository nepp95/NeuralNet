local vcpkgTriplet = os.host() == "windows" and "x64-windows" or "x64-linux"
local vcpkgInstalled = Root .. "/build/vcpkg_installed/" .. os.host() .. "/" .. vcpkgTriplet

Dependencies = {
    IncludeDirectories = {
        Glad = Root .. "/Vendor/glad/include",
        Library = Root .. "/NeuralNetwork/Source",
        Vcpkg = vcpkgInstalled .. "/include",
    },
    LibraryDirectories = {
        Debug = vcpkgInstalled .. "/debug/lib",
        Release = vcpkgInstalled .. "/lib",
    },
    RuntimeDirectories = {
        Debug = vcpkgInstalled .. "/debug/bin",
        Release = vcpkgInstalled .. "/bin",
    },
}
