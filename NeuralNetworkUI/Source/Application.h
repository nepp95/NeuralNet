#pragma once

#include "Window.h"

namespace NN
{
    class Application
    {
    public:
        Application();
        ~Application();

        auto Init() -> bool;
        auto Run() -> bool;
        auto Close() -> void;

    private:
        std::unique_ptr<Window> m_Window = nullptr;

        bool m_IsRunning = false;
        float m_LastFrameTime = 0.0f;
    };
}
