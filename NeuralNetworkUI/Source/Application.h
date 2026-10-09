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
        auto RunNN() -> void;

        auto OnWindowClose(const WindowCloseEvent& e) -> bool;
        auto OnWindowResize(const WindowResizeEvent& e) -> bool;

    private:
        std::unique_ptr<Window> m_Window = nullptr;

        std::vector<uint32_t> m_ImageData;
        std::unique_ptr<Image> m_Image = nullptr;

        bool m_IsRunning = false;
        float m_LastFrameTime = 0.0f;

        // NN related
        Perceptron m_Perceptron;
        std::vector<float> m_Inputs;
        std::vector<float> m_Mistakes;
        uint32_t m_LastMistakes = std::numeric_limits<uint32_t>::max();
        uint32_t m_Epochs = 0;
    };
}
