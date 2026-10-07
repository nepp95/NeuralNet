#pragma once

#include <GLFW/glfw3.h>

namespace NN
{
    class Window
    {
    public:
        Window(uint32_t width, uint32_t height);
        ~Window();

        auto ProcessEvents() -> void;

        auto Valid() const -> bool { return m_Window != nullptr; }

    private:
        GLFWwindow* m_Window = nullptr;

        uint32_t m_Width = 0;
        uint32_t m_Height = 0;
    };
}
