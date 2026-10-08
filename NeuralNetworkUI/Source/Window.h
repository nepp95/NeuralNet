#pragma once

#include "Event/Event.h"

#include <GLFW/glfw3.h>

namespace NN
{
    class Window
    {
    public:
        using EventCallbackFn = std::function<void(Event&)>;

        Window(uint32_t width, uint32_t height);
        ~Window();

        auto ProcessEvents() -> void;
        auto SetEventCallback(const EventCallbackFn& callback) { m_EventCallback = callback; }

        auto Valid() const -> bool { return m_Window != nullptr; }

    private:
        GLFWwindow* m_Window = nullptr;
        EventCallbackFn m_EventCallback;

        uint32_t m_Width = 0;
        uint32_t m_Height = 0;
    };
}
