#include "pch.h"
#include "Application.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

namespace NN
{
    Application::Application() {}

    Application::~Application()
    {
        ImGui::DestroyContext();
        m_Window = nullptr;
    }

    auto Application::Init() -> bool
    {
        // Create window
        m_Window = std::make_unique<Window>(1280, 720);
        if (!m_Window || !m_Window->Valid())
            return false;

        m_Window->SetEventCallback(
            [this](Event& e) -> void
            {
                OnEvent(e);
            }
        );

        // Initialize imgui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        io.FontDefault = io.Fonts->AddFontDefault();

        ImGui::StyleColorsDark();

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
            OnUpdate();
        }

        return true;
    }

    auto Application::Close() -> void
    {
        m_IsRunning = false;
    }

    auto Application::OnUpdate() -> void {}

    auto Application::OnEvent(Event& e) -> void
    {
        EventDispatcher dispatcher(e);
        dispatcher.Dispatch<WindowCloseEvent>(std::bind_front(&Application::OnWindowClose, this));
        dispatcher.Dispatch<WindowResizeEvent>(std::bind_front(&Application::OnWindowResize, this));
    }

    auto Application::OnWindowClose(const WindowCloseEvent& e) -> bool
    {
        Close();
        return true;
    }

    auto Application::OnWindowResize(const WindowResizeEvent& e) -> bool
    {
        const uint32_t width = e.GetWidth();
        const uint32_t height = e.GetHeight(); // NOLINT

        if (width == 0 || height == 0)
            return false;

        return false;
    }
}
