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

        auto OnWindowClose(const WindowCloseEvent& e) -> bool;
        auto OnWindowResize(const WindowResizeEvent& e) -> bool;

    private:
        std::unique_ptr<Window> m_Window = nullptr;

        bool m_IsRunning = false;
        float m_LastFrameTime = 0.0f;

        uint32_t m_ActiveExample = 1;
        std::unique_ptr<ExampleBase> m_Example = nullptr;

        uint32_t m_TestSampleSize = 100000;
        uint32_t m_TestsRun = 0;
        std::vector<float> m_TestResults;
        float m_LastTestAccuracy = -1.0f;
        float m_OverallTestAccuracy = -1.0f;
    };
}
