#include "pch.h"
#include "Application.h"

#include <Random.h>

#include <glm/glm.hpp>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

namespace NN
{
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
        switch (m_ActiveExample)
        {
            case 0:
                RunExample1();
                break;
        }

        // Stats
        const uint32_t samples = static_cast<uint32_t>(m_Inputs.size() / 3);
        const float accuracy = 100.0f * static_cast<float>(samples - m_LastMistakes) / static_cast<float>(samples);

        ImGui::Begin("Stats");
        const uint32_t previousExample = m_ActiveExample;
        ImGui::RadioButton("Example 1", reinterpret_cast<int*>(&m_ActiveExample), 0);
        if (ImGui::Button("Reset"))
            Reset();
        ImGui::SameLine();
        if (ImGui::Button("Test"))
            Test();

        ImGui::SeparatorText("Stats");

        ImGui::Text("Accuracy: %.2f%%", accuracy);
        ImGui::Text("Mistakes: %u", m_LastMistakes);
        ImGui::Text("Samples: %u", samples);
        ImGui::Text("Epochs: %u", m_Epochs);

        ImGui::SeparatorText("Tests");

        ImGui::Text("Test sample size: 100.000");
        ImGui::Text("Tests run: %u", m_TestsRun);
        if (m_LastTestAccuracy >= 0.0f)
            ImGui::Text("Last test accuracy: %.2f%%", m_LastTestAccuracy);
        if (m_OverallTestAccuracy >= 0.0f)
            ImGui::Text("Last test accuracy: %.2f%%", m_OverallTestAccuracy);
        ImGui::End();

        if (m_ActiveExample != previousExample)
            Reset();

        ImGui::Begin("Graph");
        switch (m_ActiveExample)
        {
            case 0:
                DrawExample1Graph();
                break;
        }
        ImGui::End();
    }

    auto Application::OnEvent(Event& e) -> void
    {
        EventDispatcher dispatcher(e);
        dispatcher.Dispatch<WindowCloseEvent>(std::bind_front(&Application::OnWindowClose, this));
        dispatcher.Dispatch<WindowResizeEvent>(std::bind_front(&Application::OnWindowResize, this));
    }

    auto Application::Reset() -> void
    {
        m_LastMistakes = std::numeric_limits<uint32_t>::max();
        m_Mistakes.clear();
        m_Epochs = 0;
        m_Inputs.clear();
        m_TestsRun = 0;
        m_LastTestAccuracy = -1.0f;
        m_OverallTestAccuracy = -1.0f;
        m_Perceptron = Perceptron();
    }

    auto Application::Test() -> void
    {
        m_TestsRun++;

        constexpr uint32_t testCount = 100000;

        float correct = 0.0f;
        switch (m_ActiveExample)
        {
            case 0:
                correct = TestExample1(testCount);
                break;
        }

        m_TestResults.emplace_back(correct);
        m_LastTestAccuracy = 100.0f * static_cast<float>(correct) / static_cast<float>(testCount);
        m_OverallTestAccuracy =
            100.0f * std::accumulate(m_TestResults.begin(), m_TestResults.end(), 0.0f) / (testCount * m_TestResults.size());
    }

    auto Application::RunExample1() -> void
    {
        // Example 1:
        // We have a set money budget: 100
        // We have different products with different prices
        // We have a shipping cost: 10-40
        // Is the item within budget?

        if (m_Inputs.empty())
        {
            constexpr uint32_t inputCount = 2000;
            constexpr uint32_t budget = 100;
            m_Inputs.resize(inputCount * 3); // 0 = price, 1 = shipping, 2 = desired result
            for (size_t i = 0; i < m_Inputs.size(); i += 3)
            {
                m_Inputs[i + 0] = Utils::GenerateRandomFloat(20.0f, 130.0f);
                m_Inputs[i + 1] = Utils::GenerateRandomFloat(5.0f, 40.0f);
                m_Inputs[i + 2] = m_Inputs[i + 0] + m_Inputs[i + 1] <= budget ? 1.0f : -1.0f;
            }
        }

        if (m_LastMistakes == 0)
            return;

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
            // Normalized so every sample is a unit vector
            const glm::vec3 input = glm::normalize(glm::vec3(price, shipping, 1.0f));
            m_Perceptron.SetInputs({ input.x, input.y, input.z });

            const float desired = m_Inputs.at(index + 2);
            if (m_Perceptron.Guess() != desired)
                mistakes++;
            m_Perceptron.Train(desired);
        }

        m_LastMistakes = mistakes;
        m_Mistakes.emplace_back(static_cast<float>(mistakes));

        m_Epochs++;
    }

    auto Application::DrawExample1Graph() -> void
    {
        // Split sample points by desired label (x = shipping, y = price)
        std::vector<float> withinX, withinY, overX, overY;
        for (size_t i = 0; i < m_Inputs.size(); i += 3)
        {
            if (m_Inputs[i + 2] > 0.0f)
            {
                withinX.push_back(m_Inputs[i + 1]);
                withinY.push_back(m_Inputs[i]);
            }
            else
            {
                overX.push_back(m_Inputs[i + 1]);
                overY.push_back(m_Inputs[i]);
            }
        }

        // Learned boundary: w0*price + w1*shipping + w2 = 0  ->  price = -(w1*shipping + w2) / w0
        const std::vector<float>& weights = m_Perceptron.GetWeights();
        const float wPrice = weights[0];
        const float wShipping = weights[1];
        const float wBias = weights[2];

        if (ImPlot::BeginPlot("Decision boundary", ImVec2(-1.0f, -1.0f)))
        {
            ImPlot::SetupAxes("Shipping", "Price");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 45.0, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 140.0, ImPlotCond_Always);

            // Sample points
            ImPlot::PlotScatter("Within budget", withinX.data(), withinY.data(), static_cast<int>(withinX.size()));
            ImPlot::PlotScatter("Over budget", overX.data(), overY.data(), static_cast<int>(overX.size()));

            // Reference line: price = 100 - shipping
            constexpr float referenceX[2] = { 5.0f, 40.0f };
            constexpr float referenceY[2] = { 95.0f, 60.0f };
            ImPlot::PlotLine("Reference", referenceX, referenceY, 2);

            // Learned line at shipping = 5 and 40
            const float learnedX[2] = { 5.0f, 40.0f };
            const float learnedY[2] = { -(wShipping * 5.0f + wBias) / wPrice, -(wShipping * 40.0f + wBias) / wPrice };
            ImPlot::PlotLine("Learned", learnedX, learnedY, 2);

            ImPlot::EndPlot();
        }
    }

    auto Application::TestExample1(uint32_t testCount) -> uint32_t
    {
        // 10000 unseen samples, guess only - the weights are not touched
        constexpr uint32_t budget = 100;

        uint32_t correct = 0;
        for (uint32_t i = 0; i < testCount; i++)
        {
            const float price = Utils::GenerateRandomFloat(20.0f, 130.0f);
            const float shipping = Utils::GenerateRandomFloat(5.0f, 40.0f);
            const float desired = price + shipping <= budget ? 1.0f : -1.0f;

            const glm::vec3 input = glm::normalize(glm::vec3(price, shipping, 1.0f));
            m_Perceptron.SetInputs({ input.x, input.y, input.z });

            if (m_Perceptron.Guess() == desired)
                correct++;
        }

        return correct;
    }

    auto Application::RunExample2() -> void {}

    auto Application::DrawExample2Graph() -> void {}

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
