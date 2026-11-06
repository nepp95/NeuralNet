#pragma once

#include <NeuralNetwork.h>

namespace NN
{
    class ExampleBase
    {
    public:
        virtual ~ExampleBase() = default;

        virtual auto Train() -> void = 0;

        virtual auto OnSettingsRender() -> void {}
        virtual auto OnUIRender() -> void {}

    protected:
        std::unique_ptr<Network> m_Network = nullptr;
    };
}
