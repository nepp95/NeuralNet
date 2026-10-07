#include "pch.h"
#include "Application.h"

#include <GLFW/glfw3.h>

namespace NN
{
    Application::Application() {}

    Application::~Application() {}

    auto Application::Init() -> bool
    {
        m_Window = std::make_unique<Window>(1280, 720);
        if (!m_Window || !m_Window->Valid())
            return false;

        m_IsRunning = true;

        return true;
    }

    auto Application::Run() -> bool
    {
        while (m_IsRunning)
        {
            const auto time = static_cast<float>(glfwGetTime());
            const float timestep = time - m_LastFrameTime;
            m_LastFrameTime = time;

            m_Window->ProcessEvents();
        }

        return true;
    }

    auto Application::Close() -> void
    {
        m_IsRunning = false;
    }
}
