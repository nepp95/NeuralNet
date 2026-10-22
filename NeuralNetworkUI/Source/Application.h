#pragma once

#include "Event/ApplicationEvent.h"
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
        auto OnWindowClose(const WindowCloseEvent& e) -> bool;
        auto OnWindowResize(const WindowResizeEvent& e) -> bool;

    private:
        std::unique_ptr<Window> m_Window = nullptr;

        bool m_IsRunning = false;
        float m_LastFrameTime = 0.0f;

        std::unique_ptr<ExampleBase> m_Example = nullptr;
        std::unique_ptr<Image> m_TestImage = nullptr;
        std::vector<uint8_t> m_TestImagePixels;

        bool m_IsTraining = true;
        bool m_StopAtLoss = true;
        bool m_StopAtAccuracy = true;
        float m_TargetLoss = 0.05f;
        float m_TargetAccuracy = 0.95f;
    };
}
