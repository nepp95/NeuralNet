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
        auto OnWindowClose(const WindowCloseEvent& e) -> bool;
        auto OnWindowResize(const WindowResizeEvent& e) -> bool;

    private:
        std::unique_ptr<Window> m_Window = nullptr;

        bool m_IsRunning = false;
        float m_LastFrameTime = 0.0f;

        std::unique_ptr<ExampleBase> m_Example = nullptr;
    };
}
