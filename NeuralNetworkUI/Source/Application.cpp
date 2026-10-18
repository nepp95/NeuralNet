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

        m_TestImage.reset();

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
        m_Example->SetTestImageDrawer(
            [this](std::span<const float> pixels, uint32_t width, uint32_t height) -> void
            {
                if (!m_TestImage || m_TestImage->GetWidth() != width || m_TestImage->GetHeight() != height)
                    m_TestImage = std::make_unique<Image>(ImageParams{ width, height, nullptr });

                m_TestImagePixels.resize(static_cast<size_t>(width) * height * 4);
                for (size_t i = 0; i < pixels.size(); i++)
                {
                    const auto value = static_cast<uint8_t>(std::lround(std::clamp(pixels[i], 0.0f, 1.0f) * 255.0f));

                    m_TestImagePixels[i * 4 + 0] = value;
                    m_TestImagePixels[i * 4 + 1] = value;
                    m_TestImagePixels[i * 4 + 2] = value;
                    m_TestImagePixels[i * 4 + 3] = 255;
                }

                m_TestImage->SetData(m_TestImagePixels.data());

                const float side = ImGui::GetContentRegionAvail().x;
                ImGui::Image(static_cast<ImTextureID>(m_TestImage->GetID()), ImVec2(side, side));
            }
        );

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

        auto& options = m_Example->GetGraphOptions();

        if (m_IsTraining)
            m_Example->Run(epochs, learningRate);

        if (options.TestAutoEval &&
            m_Example->GetTotalEpochs() - m_Example->GetLastTestEvalEpoch() >= static_cast<uint64_t>(options.TestEvalInterval))
        {
            m_Example->EvaluateTestSet();
        }

        if (m_IsTraining)
        {
            const bool lossReached = m_StopAtLoss && m_Example->GetLastLoss() <= m_TargetLoss;
            const bool accuracyReached =
                m_StopAtAccuracy && !m_Example->GetTestPredictions().empty() && m_Example->GetTestAccuracy() >= m_TargetAccuracy;

            if (lossReached || accuracyReached)
                m_IsTraining = false;
        }

        // Settings
        ImGui::Begin("Settings");
        ImGui::SliderInt("Max neurons per layer", &options.MaxNeuronsPerLayer, 2, 64);

        ImGui::SeparatorText("Training");

        if (ImGui::Button(m_IsTraining ? "Pause" : "Resume"))
            m_IsTraining = !m_IsTraining;

        ImGui::SameLine();
        ImGui::Text("Epoch %llu | Loss %.6f", static_cast<unsigned long long>(m_Example->GetTotalEpochs()), m_Example->GetLastLoss());

        ImGui::Checkbox("Stop at loss", &m_StopAtLoss);
        ImGui::BeginDisabled(!m_StopAtLoss);
        ImGui::SetNextItemWidth(160.0f);
        ImGui::DragFloat("Target loss", &m_TargetLoss, 0.0005f, 0.0f, 1.0f, "%.4f");
        ImGui::EndDisabled();

        ImGui::Checkbox("Stop at accuracy", &m_StopAtAccuracy);
        ImGui::BeginDisabled(!m_StopAtAccuracy);
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderFloat("Target accuracy", &m_TargetAccuracy, 0.0f, 1.0f, "%.2f");
        ImGui::EndDisabled();

        if (m_StopAtAccuracy && !options.TestAutoEval)
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "Enable 'Auto evaluate' in the Test tab to update accuracy");

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
