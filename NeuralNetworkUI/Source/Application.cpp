#include "pch.h"
#include "Application.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace NN
{
    Application::Application() {}

    Application::~Application()
    {
        Log::Info("Closing application...");

        m_Image.reset();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_Window.reset();
    }

    auto Application::Init() -> bool
    {
        // Create window
        Log::Info("Creating window imgui");

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
        Log::Info("Initializing imgui");

        glfwMakeContextCurrent(m_Window->GetNative());
        gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress));

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        io.FontDefault = io.Fonts->AddFontDefault();

        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(m_Window->GetNative(), true);
        ImGui_ImplOpenGL3_Init();

        m_Image = std::make_unique<Image>(ImageParams{ 1280, 720 });

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

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            OnUpdate();

            ImGui::Render();
            int width, height;
            glfwGetFramebufferSize(m_Window->GetNative(), &width, &height);
            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT);
            glBlitNamedFramebuffer(
                m_Image->GetFramebuffer(), 0, 0, 0, m_Image->GetWidth(), m_Image->GetHeight(), 0, 0, width, height, GL_COLOR_BUFFER_BIT,
                GL_LINEAR
            );
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            GLFWwindow* context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(context);

            glfwSwapBuffers(m_Window->GetNative());
        }

        return true;
    }

    auto Application::Close() -> void
    {
        m_IsRunning = false;
    }

    auto Application::OnUpdate() -> void
    {
        // Prep image
        const auto width = m_Image->GetWidth();
        const auto height = m_Image->GetHeight();
        const auto imageSize = width * height * 4;

        if (imageSize != m_ImageData.size())
            m_ImageData.resize(imageSize);

        // Write image
        for (uint32_t y = 0; y < height; y++)
        {
            for (uint32_t x = 0; x < width; x++)
            {
                m_ImageData[width * y + x] = static_cast<uint32_t>(rand() * 255);
            }
        }

        // Upload image
        m_Image->SetData(m_ImageData.data());
    }

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
