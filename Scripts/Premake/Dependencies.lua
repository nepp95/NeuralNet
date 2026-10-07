local vcpkgTriplet = os.host() == "windows" and "x64-windows" or "x64-linux"
local vcpkgInstalled = NeuralNetworkRoot .. "/build/vcpkg_installed/" .. os.host() .. "/" .. vcpkgTriplet

Dependencies = {
    IncludeDirectories = {
        Library = NeuralNetworkRoot .. "/NeuralNetwork/Source",
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
    WindowsDebugLinks = {
        "glfw3dll", "glm", "imguid",
        "opengl32", "gdi32", "user32", "shell32",
    },
    WindowsReleaseLinks = {
        "glfw3dll", "glm", "imgui",
        "opengl32", "gdi32", "user32", "shell32",
    },
    LinuxLinks = {
        "glfw3", "glm", "imgui",
        "X11", "Xrandr", "Xinerama", "Xcursor", "Xi", "pthread", "dl",
    },
}
