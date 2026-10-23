#include "pch.h"
#include "Window.h"

#include "Event/ApplicationEvent.h"

namespace NN
{
    namespace
    {
        auto GLFWErrorCallback(int error, const char* description) -> void
        {
            std::cout << "GLFW Error: (" << error << ") " << description << "\n";
        }
    }

    Window::Window(uint32_t width, uint32_t height)
        : m_Width(width), m_Height(height)
    {
        auto success = glfwInit();
        if (success != GLFW_TRUE)
        {
            std::cout << "Failed to initialize glfw!\n";
            return;
        }

        glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_FALSE);

        // Compute shaders need 4.6
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

        glfwSetErrorCallback(GLFWErrorCallback);

        m_Window = glfwCreateWindow(m_Width, m_Height, "Neural Network", nullptr, nullptr);
        if (!m_Window)
        {
            glfwTerminate();
            m_Window = nullptr;
            return;
        }

        // Set callbacks
        glfwSetWindowUserPointer(m_Window, this);

        glfwSetWindowCloseCallback(
            m_Window,
            [](GLFWwindow* window) -> void
            {
                const auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
                WindowCloseEvent e;
                self->m_EventCallback(e);
            }
        );

        glfwSetWindowSizeCallback(
            m_Window,
            [](GLFWwindow* window, const int width, const int height) -> void
            {
                auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
                self->m_Width = static_cast<uint32_t>(width);
                self->m_Height = static_cast<uint32_t>(height);
                WindowResizeEvent e(width, height);
                self->m_EventCallback(e);
            }
        );
    }

    Window::~Window()
    {
        glfwDestroyWindow(m_Window);
        glfwTerminate();
    }

    auto Window::ProcessEvents() -> void
    {
        glfwPollEvents();
    }
}
