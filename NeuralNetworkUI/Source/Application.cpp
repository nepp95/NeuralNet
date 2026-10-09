#include "pch.h"
#include "Application.h"

#include <Random.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

namespace NN
{
    namespace
    {
        constexpr size_t historySize = 300;

        uint32_t s_EpochOffset = 0;

        auto FormatEpochTick(const double value, char* buffer, const int size, void* userData) -> int
        {
            const auto offset = *static_cast<uint32_t*>(userData);
            const auto clamped = value < 0.0 ? 0u : static_cast<uint32_t>(value);
            return snprintf(buffer, size, "%u", clamped + offset);
        }
    }

    Application::~Application()
    {
        Log::Info("Closing application...");

        m_Image.reset();
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

        m_Image = std::make_unique<Image>(ImageParams{ 1280, 720 });

        // Initialize neural network
        constexpr uint32_t inputCount = 2000;
        constexpr uint32_t budget = 100;
        m_Inputs.resize(inputCount * 3); // 0 = price, 1 = shipping, 2 = desired result
        for (size_t i = 0; i < m_Inputs.size(); i += 3)
        {
            m_Inputs[i + 0] = Utils::GenerateRandomFloat(20.0f, 130.0f);
            m_Inputs[i + 1] = Utils::GenerateRandomFloat(5.0f, 40.0f);
            m_Inputs[i + 2] = m_Inputs[i + 0] + m_Inputs[i + 1] <= budget ? 1.0f : -1.0f;
        }

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
        // Docking host
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        // Prep image
        const auto width = m_Image->GetWidth();
        const auto height = m_Image->GetHeight();
        const auto imageSize = width * height;

        if (imageSize != m_ImageData.size())
        {
            m_ImageData.resize(imageSize);
            std::fill(m_ImageData.begin(), m_ImageData.end(), 0xFF202020);
        }

        // NN
        if (m_LastMistakes > 0)
            RunNN();

        // Stats
        const uint32_t samples = static_cast<uint32_t>(m_Inputs.size() / 3);
        const float accuracy = 100.0f * static_cast<float>(samples - m_LastMistakes) / static_cast<float>(samples);

        ImGui::Begin("Stats");
        ImGui::Text("Accuracy: %.1f%%", accuracy);
        ImGui::Separator();
        ImGui::Text("Mistakes: %u", m_LastMistakes);
        ImGui::Text("Samples: %u", samples);
        ImGui::Text("Epochs: %u", m_Epochs);
        ImGui::End();

        ImGui::Begin("Graph");
        if (ImPlot::BeginPlot("Mistakes", ImVec2(-1.0f, -1.0f)))
        {
            ImPlot::SetupAxes("Iteration", "Mistakes");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, static_cast<double>(historySize), ImPlotCond_Once);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -0.5, 2000.0, ImPlotCond_Once);

            s_EpochOffset = m_Epochs - static_cast<uint32_t>(m_Mistakes.size()) + 1;
            ImPlot::SetupAxisFormat(ImAxis_X1, &FormatEpochTick, &s_EpochOffset);
            ImPlot::PlotLine("Mistakes", m_Mistakes.data(), static_cast<int>(m_Mistakes.size()));
            ImPlot::EndPlot();
        }
        ImGui::End();

        // Upload image
        m_Image->SetData(m_ImageData.data());
    }

    auto Application::OnEvent(Event& e) -> void
    {
        EventDispatcher dispatcher(e);
        dispatcher.Dispatch<WindowCloseEvent>(std::bind_front(&Application::OnWindowClose, this));
        dispatcher.Dispatch<WindowResizeEvent>(std::bind_front(&Application::OnWindowResize, this));
    }

    auto Application::RunNN() -> void
    { // Run neural network
        //
        // Example 1:
        // We have a set money budget: 100
        // We have different products with different prices
        // We have a shipping cost: 10-40
        // Is the item within budget?

        m_Perceptron.SetActivationFn(
            [](float sum) -> float
            {
                return sum >= 0.0f ? 1.0f : -1.0f;
            }
        );

        uint32_t mistakes = 0;
        for (size_t i = 0; i < m_Inputs.size() / 3; i++)
        {
            const size_t index = i * 3;
            const auto& price = m_Inputs.at(index);
            const auto& shipping = m_Inputs.at(index + 1);
            m_Perceptron.SetInputs({ price, shipping, 1.0f });

            const float desired = m_Inputs.at(index + 2);
            if (m_Perceptron.Guess() != desired)
                mistakes++;
            m_Perceptron.Train(desired);
        }

        m_LastMistakes = mistakes;
        m_Mistakes.emplace_back(static_cast<float>(mistakes));
        if (m_Mistakes.size() > historySize)
            m_Mistakes.erase(m_Mistakes.begin());
        m_Epochs++;
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
