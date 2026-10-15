#pragma once

#include "NeuralNetworks/ExampleBase.h"

namespace NN
{
    class SinglePerceptron
    {
    public:
        auto Guess() const -> float;
        auto Train(float desired) -> void;

        auto SetInputs(const std::vector<float>& inputs) -> void;
        auto SetActivationFn(const std::function<float(float)>& activationFn) -> void { m_ActivationFn = activationFn; }
        auto GetWeights() const -> const std::vector<float>& { return m_Weights; }

    private:
        std::vector<float> m_Data;
        std::vector<float> m_Weights;
        float m_LearningConstant = 0.01f;
        std::function<float(float)> m_ActivationFn;
    };

    class SinglePerceptronExample : public ExampleBase
    {
    public:
        ~SinglePerceptronExample() override = default;

        auto Run() -> void override;
        auto Test(uint32_t testCount) -> uint32_t override;
        auto RenderVisualisation() -> void override;

    private:
        SinglePerceptron m_Perceptron;
    };
}
