#pragma once

#include "NeuralNetworks/ExampleBase.h"
#include "NeuralNetworks/Perceptron.h"

namespace NN
{
    class SinglePerceptronExample : public ExampleBase
    {
    public:
        ~SinglePerceptronExample() override = default;

        auto Run() -> void override;
        auto Test(uint32_t testCount) -> uint32_t override;
        auto RenderVisualisation() -> void override;

    private:
        Perceptron m_Perceptron;
    };
}
