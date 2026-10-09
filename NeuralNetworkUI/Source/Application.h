#pragma once

#include "Event/ApplicationEvent.h"
#include "Image.h"
#include "Window.h"

#include <NeuralNetwork.h>

namespace NN
{
    class Application
    {
    public:
        Application() = default;
        ~Application();

        auto Init() -> bool;
        auto Run() -> bool;
        auto Close() -> void;

        auto OnUpdate() -> void;
        auto OnEvent(Event& e) -> void;

    private:
        auto Reset() -> void;
        auto Test() -> void;

        auto RunExample1() -> void;
        auto DrawExample1Graph() -> void;
        auto TestExample1(uint32_t testCount) -> uint32_t;
        auto RunExample2() -> void;
        auto DrawExample2Graph() -> void;

        auto OnWindowClose(const WindowCloseEvent& e) -> bool;
        auto OnWindowResize(const WindowResizeEvent& e) -> bool;

    private:
        std::unique_ptr<Window> m_Window = nullptr;

        std::vector<uint32_t> m_ImageData;
        std::unique_ptr<Image> m_Image = nullptr;

        bool m_IsRunning = false;
        float m_LastFrameTime = 0.0f;

        uint32_t m_ActiveExample = 0;

        // NN related
        Perceptron m_Perceptron;
        std::vector<float> m_Inputs;
        std::vector<float> m_Mistakes;
        uint32_t m_LastMistakes = std::numeric_limits<uint32_t>::max();
        uint32_t m_Epochs = 0;
        uint32_t m_TestsRun = 0;
        float m_LastTestAccuracy = -1.0f;
        std::vector<float> m_TestResults;
        float m_OverallTestAccuracy = -1.0f;
    };
}
