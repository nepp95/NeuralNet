#include "pch.h"
#include "Application.h"

#include <Random.h>

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

        m_Example.reset();

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

        // Initialize neural network
        // m_Example = std::make_unique<SinglePerceptronExample>();
        m_Example = std::make_unique<MultiNeuronExample>();

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

        // NN
        m_Example->Run();

        // Stats
        ImGui::Begin("Stats");

        if (m_Example)
        {
            const auto& inputs = m_Example->GetInputs();
            const uint32_t epochs = m_Example->GetEpochs();
            const uint32_t lastMistakes = m_Example->GetLastMistakes();

            const uint32_t samples = static_cast<uint32_t>(inputs.size() / 3);
            const float accuracy = 100.0f * static_cast<float>(samples - lastMistakes) / static_cast<float>(samples);

            const uint32_t previousExample = m_ActiveExample;
            ImGui::RadioButton("Example 1", reinterpret_cast<int*>(&m_ActiveExample), 0);
            ImGui::RadioButton("Example 2", reinterpret_cast<int*>(&m_ActiveExample), 1);
            if (ImGui::Button("Reset"))
                Reset();
            ImGui::SameLine();
            if (ImGui::Button("Test"))
                Test();

            ImGui::SeparatorText("Stats");

            ImGui::Text("Accuracy: %.2f%%", accuracy);
            ImGui::Text("Mistakes: %u", lastMistakes);
            ImGui::Text("Samples: %u", samples);
            ImGui::Text("Epochs: %u", epochs);

            ImGui::SeparatorText("Tests");

            ImGui::Text("Test sample size: %u", m_TestSampleSize);
            ImGui::Text("Tests run: %u", m_TestsRun);
            if (m_LastTestAccuracy >= 0.0f)
                ImGui::Text("Last test accuracy: %.2f%%", m_LastTestAccuracy);
            if (m_OverallTestAccuracy >= 0.0f)
                ImGui::Text("Overall test accuracy: %.2f%%", m_OverallTestAccuracy);

            if (m_ActiveExample != previousExample)
                Reset();

            ImGui::SeparatorText("Neural Network Structure");

            ImGui::TextUnformatted(
                "INPUT:   price, shipping, bias (1.0)\n"
                "  |\n"
                "  v\n"
                "LAYER 0 (2 neurons, tanh)\n"
                "  N0:  w[price, shipping]  + bias\n"
                "  N1:  w[price, shipping]  + bias\n"
                "  |\n"
                "  v\n"
                "LAYER 1 (4 neurons, tanh)\n"
                "  N0:  w[L0.N0, L0.N1]  + bias\n"
                "  N1:  w[L0.N0, L0.N1]  + bias\n"
                "  N2:  w[L0.N0, L0.N1]  + bias\n"
                "  N3:  w[L0.N0, L0.N1]  + bias\n"
                "  |\n"
                "  v\n"
                "LAYER 2 (1 neuron, tanh)\n"
                "  N0:  w[L1.N0, L1.N1, L1.N2, L1.N3]  + bias\n"
                "  |\n"
                "  v\n"
                "OUTPUT:  sign > 0 = within budget\n"
                "         sign < 0 = over budget"
            );
        }

        ImGui::End(); // Stats

        if (m_Example)
        {
            ImGui::Begin("Graph");
            m_Example->RenderVisualisation();
            ImGui::End(); // Graph
        }
    }

    auto Application::OnEvent(Event& e) -> void
    {
        EventDispatcher dispatcher(e);
        dispatcher.Dispatch<WindowCloseEvent>(std::bind_front(&Application::OnWindowClose, this));
        dispatcher.Dispatch<WindowResizeEvent>(std::bind_front(&Application::OnWindowResize, this));
    }

    auto Application::Reset() -> void
    {
        switch (m_ActiveExample)
        {
            case 0:
            {
                m_Example = std::make_unique<SinglePerceptronExample>();
                break;
            }

            case 1:
            {
                m_Example = std::make_unique<MultiNeuronExample>();
                break;
            }

            default:
            {
                m_Example = std::make_unique<SinglePerceptronExample>();
                break;
            }
        }
    }

    auto Application::Test() -> void
    {
        m_TestsRun++;

        const uint32_t correct = m_Example->Test(m_TestSampleSize);

        m_TestResults.emplace_back(static_cast<float>(correct));
        m_LastTestAccuracy = 100.0f * static_cast<float>(correct) / static_cast<float>(m_TestSampleSize);
        m_OverallTestAccuracy =
            100.0f * std::accumulate(m_TestResults.begin(), m_TestResults.end(), 0.0f) / (m_TestSampleSize * m_TestResults.size());
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
