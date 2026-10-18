#include "pch.h"
#include "Application.h"

#include <Utility/Random.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

namespace NN
{
    Application::~Application()
    {
        Log::Info("Closing application...");

        ImPlot::DestroyContext();
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
        ImPlot::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        io.FontDefault = io.Fonts->AddFontDefault();

        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(m_Window->GetNative(), true);
        ImGui_ImplOpenGL3_Init();

        m_IsRunning = true;
        return true;
    }

    auto Application::Run() -> bool
    {
        m_Example = std::make_unique<MnistExample>();

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
        // Docking host
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        // Run NN
        constexpr uint32_t epochs = 1;
        constexpr float learningRate = 0.5f;
        m_Example->Run(epochs, learningRate);

        // Settings
        ImGui::Begin("Settings");
        ImGui::SliderInt("Max neurons per layer", &m_Example->GetGraphOptions().MaxNeuronsPerLayer, 2, 64);
        ImGui::End(); // Settings

        ImGui::Begin("Graph");
        m_Example->RenderVisualisation();
        ImGui::End(); // Graph
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
