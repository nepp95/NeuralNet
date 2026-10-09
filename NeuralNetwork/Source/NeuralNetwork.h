#pragma once

namespace NN
{
    class Perceptron
    {
    public:
        Perceptron() = default;
        ~Perceptron() = default;

        auto SetInputs(std::vector<float>&& weights) -> void;
        auto SetActivationFn(const std::function<float(float)>& activationFn) -> void;

        auto Guess() -> float;
        auto Train(float desired) -> void;

    private:
        std::vector<float> m_Data;
        std::vector<float> m_Weights;
        float m_LearningConstant = 0.05f;
        std::function<float(float)> m_ActivationFn;
    };
}
